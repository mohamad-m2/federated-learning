import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
class Encoder(nn.Module):
    def __init__(self,inp):
        super().__init__()
        self.encod1 = nn.Conv2d(inp,8,(5,5),stride=2,padding=1)
        self.encod2 = nn.Conv2d(8,16,(3,3),stride=2,padding=0)
        self.encod3 = nn.Conv2d(16,32,(3,3),stride=1,padding=0)
        self.encod4 = nn.Conv2d(32,64,(3,3),stride=1,padding=0)
        self.encod5 = nn.Conv2d(64,128,(3,3),stride=1,padding=0)
        self.act=nn.ReLU()
    def forward(self,x):
      x=self.act(self.encod1(x))
      x=self.act(self.encod2(x))
      x=self.act(self.encod3(x))
      x=self.act(self.encod4(x))
      x=self.act(self.encod5(x))
      return x

        
class Decoder(nn.Module):
    def __init__(self,out):
        super().__init__()
        self.decod1 = nn.ConvTranspose2d(128,out_channels=64,kernel_size=(3,3),stride=1,padding=0)
        self.decod2=nn.ConvTranspose2d(64,out_channels=32,kernel_size=(3,3),stride=1,padding=0)
        self.decod3=nn.ConvTranspose2d(32,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
        self.decod4=nn.ConvTranspose2d(16,out_channels=8,kernel_size=(3,3),stride=2,padding=0)
        self.decod5=nn.ConvTranspose2d(8,out_channels=out,kernel_size=(4,4),stride=2,padding=0)
        self.act=nn.ReLU()
    def forward(self,x):
      x=self.act(self.decod1(self.act(x)))
      x=self.act(self.decod2(x))
      x=self.act(self.decod3(x))
      x=self.act(self.decod4(x))  
      x=self.decod5(x)
      return self.act(x)


class Encoder_Batch(nn.Module):
    def __init__(self,inp):
        super().__init__()
        self.encod1 = nn.Conv2d(inp,8,(5,5),stride=2,padding=1)
        self.BN1=nn.BatchNorm2d(8)
        self.encod2 = nn.Conv2d(8,16,(3,3),stride=2,padding=0)
        self.BN2=nn.BatchNorm2d(16)
        self.encod3 = nn.Conv2d(16,32,(3,3),stride=1,padding=0)
        self.BN3=nn.BatchNorm2d(32)
        self.encod4 = nn.Conv2d(32,64,(3,3),stride=1,padding=0)
        self.BN4=nn.BatchNorm2d(64)
        self.encod5 = nn.Conv2d(64,128,(3,3),stride=1,padding=0)
        self.BN5=nn.BatchNorm2d(128)
        self.act=nn.ReLU()
    def forward(self,x):
      x=self.BN1(self.encod1(x))
      x=self.act(x)
      x=self.BN2(self.encod2(x))
      x=self.act(x)
      x=self.BN3(self.encod3(x))
      x=self.act(x)
      x=self.BN4(self.encod4(x))
      x=self.act(x)
      x=self.BN5(self.encod5(x))
      x=self.act(x)
      return x

        
class Decoder_Batch(nn.Module):
    def __init__(self,out):
        super().__init__()
        self.decod1 = nn.ConvTranspose2d(128,out_channels=64,kernel_size=(3,3),stride=1,padding=0)
        self.BN1=nn.BatchNorm2d(64)
        self.decod2=nn.ConvTranspose2d(64,out_channels=32,kernel_size=(3,3),stride=1,padding=0)
        self.BN2=nn.BatchNorm2d(32)
        self.decod3=nn.ConvTranspose2d(32,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
        self.BN3=nn.BatchNorm2d(16)
        self.decod4=nn.ConvTranspose2d(16,out_channels=8,kernel_size=(3,3),stride=2,padding=0)
        self.BN4=nn.BatchNorm2d(8)
        self.decod5=nn.ConvTranspose2d(8,out_channels=out,kernel_size=(4,4),stride=2,padding=0)
        self.BN5=nn.BatchNorm2d(3)
        self.act=nn.ReLU()
    def forward(self,x):
      x=self.BN1(self.decod1(self.act(x)))
      x=self.act(x)
      x=self.BN2(self.decod2(x))
      x=self.act(x)
      x=self.BN3(self.decod3(x))
      x=self.act(x)
      x=self.BN4(self.decod4(x)) 
      x=self.act(x) 
      x=self.decod5(x)
      x=self.BN5(x)
      return self.act(x)

def train_epoch_ed(encoder, decoder, dataloader, loss_fn, optimizer,device='cuda'):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        decoded_data=decoded_data
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def end_dist(encoder,dataloader,device='cuda'): # get the gaussian distribution for the data after encoding

    k=[]
    encoder.train(False)
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        encoded_data=encoded_data[:,:,0,0]
        for i in encoded_data:
          k.append(i.tolist())

    k=torch.tensor(k)
    return np.array(k)


def train_encoder_decoder(epochs,loaders,Device):
  loss_fn = torch.nn.MSELoss()

  ### Define an optimizer (both for the encoder and the decoder!)
  lr= 0.01

  ### Set the random seed for reproducible results
  torch.manual_seed(0)

  #model = Autoencoder(encoded_space_dim=encoded_space_dim)
  encoder = Encoder_Batch(3) #or #Encoder_Batch(3)
  decoder = Decoder_Batch(3) #or #Decoder_Batch(3)
  params_to_optimize = [
      {'params': encoder.parameters()},
      {'params': decoder.parameters()}
  ]
  optim_ed = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
  scheduler=optim.lr_scheduler.StepLR(optim_ed, step_size=30, gamma=0.1)
  encoder.to(Device)
  decoder.to(Device)

  for i in range(epochs):
    k=train_epoch_ed(encoder,decoder,loaders,loss_fn,optim_ed)
    print('\t partial train loss (single batch): %f' % (k))
    scheduler.step()
    if(i%50==0):
      print(i)
      print(scheduler.get_last_lr())

  k=end_dist(encoder,loaders)
  mean=k.mean(axis=0)
  k=k-mean
  cov=0
  for i in k:
    cov+=np.dot(i.reshape((128,1)),i.reshape((1,128)))

  return decoder,mean,cov
    
