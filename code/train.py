import torch
from torch.utils.data import DataLoader,Dataset,ConcatDataset
import numpy as np
from data import prepare_data_loader
import torch.nn as nn

def mix_generate(loader,fraction,model_for_generation):
  with torch.no_grad():
    quantity=round(len(loader.dataset)*fraction)
    maxpix=loader.dataset[:][0].max()
    minpix=loader.dataset[:][0].min()
    data=[]
    for i in range(quantity):

      if(model_for_generation['type']=='encoder_decoder'):
        z=torch.from_numpy(np.random.multivariate_normal(model_for_generation['mean'],model_for_generation['cov']).astype(np.float32))
        z=z.view((1,128,1,1))
        z=z.to('cuda')
        sample=model_for_generation['model'](z)
      else:
        sample=model_for_generation['model'](torch.randn(1, model_for_generation['Nz'], 1, 1).to('cuda'))
        
      sample-=sample.min()
      sample=sample/sample.max()
      
      sample=sample*(maxpix-minpix)+minpix
      data.append((10,sample.to('cpu')))
    loaders=prepare_data_loader(data,transform=None,validation_fraction=0)[0]
    dataset=ConcatDataset((loaders.dataset,loader.dataset))
    del loaders
  return DataLoader(dataset,batch_size=256,shuffle=True,num_workers=4)

def train(model, train_loader, optimizer, loss_fun, device):
  model.to(device)
  model.train()
  num_data = 0
  correct = 0
  loss_all = 0
  train_iter = iter(train_loader)
  for step in range(len(train_iter)):
      optimizer.zero_grad()
      x, y = next(train_iter)
      num_data += y.size(0)
      x = x.to(device).float()
      y = y.to(device).long()
      output = model(x)

      loss = loss_fun(output, y)
      loss.backward()
      loss_all += loss.item()
      optimizer.step()

      pred = output.data.max(1)[1]
      correct += pred.eq(y.view(-1)).sum().item()
  return loss_all/len(train_iter), correct/num_data


def train_EXT(model, train_loader, optimizer,wr,fr ,device,generation_model,cycle=True):
  model.to(device)
  model.train()
  num_data = 0
  correct = 0
  loss_all = 0
  if(cycle):
    loader_train=mix_generate(train_loader,fr,generation_model)
    train_iter = iter(loader_train)
  else:
    train_iter = iter(train_loader)
  w=np.ones((10))
  w=np.hstack((w,wr))
  w=torch.from_numpy(w)
  w=w.to(device).float()
  loss_fun=nn.CrossEntropyLoss(weight=w)
  for step in range(len(train_iter)):
      optimizer.zero_grad()
      x, y = next(train_iter)
      num_data += y.size(0)
      x = x.to(device).float()
      y = y.to(device).long()
      output = model(x)

      loss = loss_fun(output, y)
      loss.backward()
      loss_all += loss.item()
      optimizer.step()

      pred = output.data.max(1)[1]
      correct += pred.eq(y.view(-1)).sum().item()
  if(cycle):
    del loader_train
  return loss_all/len(train_iter), correct/num_data


def train_IR(model, train_loader, optimizer, loss_fun,p, device):
  model.to(device)
  model.train()
  num_data = 0
  correct = 0
  loss_all = 0
  train_iter = iter(train_loader)
  for step in range(len(train_iter)):
    optimizer.zero_grad()
    x, y = next(train_iter)
    num_data += y.size(0)
    x = x.to(device).float()
    y = y.to(device).long()
    output = model(x)
    
    with torch.no_grad():
      local_batch=[0]*len(p)
      for i in y:
        local_batch[i]+=1
      local_batch=np.array(local_batch)
      local_batch=local_batch/local_batch.sum() +0.00001
      weights=np.array(p)/local_batch
    tot_loss=torch.tensor(0.).to(device)
    n=0
    for i,j in zip(output,y):
      loss = loss_fun(i.view(1,-1), j.view(-1))
      tot_loss+=loss*(weights[j])
      n+=weights[j]
    tot_loss/=n

    tot_loss.backward()
    loss_all += loss.item()
    optimizer.step()

    pred = output.data.max(1)[1]
    correct += pred.eq(y.view(-1)).sum().item()
  return loss_all/len(train_iter), correct/num_data


def train_fedprox(server_model, model, train_loader, optimizer, loss_fun,mu, device):
  model.to(device)
  model.train()
  num_data = 0
  correct = 0
  loss_all = 0
  train_iter = iter(train_loader)
  for step in range(len(train_iter)):
    optimizer.zero_grad()
    x, y = next(train_iter)
    num_data += y.size(0)
    x = x.to(device).float()
    y = y.to(device).long()
    output = model(x)

    loss = loss_fun(output, y)

    #########################we implement FedProx Here###########################
    # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
    if step>0:
        w_diff = torch.tensor(0., device=device)
        for w, w_t in zip(server_model.parameters(), model.parameters()):
            w_diff += torch.pow(torch.norm(w - w_t), 2)
        loss += (mu / 2.0) * w_diff
    #############################################################################

    loss.backward()
    loss_all += loss.item()
    optimizer.step()

    pred = output.data.max(1)[1]
    correct += pred.eq(y.view(-1)).sum().item()
  return loss_all/len(train_iter), correct/num_data

def communication(server_model, models,client_weights,nb_client): # takes server model and the selected model clients with their weights and avg them
    with torch.no_grad():
      for key in server_model.state_dict().keys():
              temp = torch.zeros_like(server_model.state_dict()[key])
              z=0
              for client_idx in range(nb_client):
                  if('num_batches_tracked' in key):
                    z+=client_weights[client_idx] * models[client_idx].state_dict()[key].item()
                  else:
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
              if('num_batches_tracked' in key):
                temp+=torch.tensor(round(z))
              server_model.state_dict()[key].data.copy_(temp)
    return server_model.state_dict()