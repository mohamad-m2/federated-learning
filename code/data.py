import copy
import collections
import torch
import torchvision
from torch.utils.data import DataLoader,Dataset,Subset
import numpy as np
from torchvision import transforms
import os

from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def tiny_image_net(file,trans,threshold): # threshold how much toignore
  files=os.walk(file+'/train')
  files=list(files)[0][1]
  data=[]
  for i in files:
    j=list(os.walk(file+'/train/'+i+'/images'))[0][2]
   
    for k in j:
  
      prob=np.random.rand()
      if(prob<threshold):
        continue
      image=pil_loader(file+'/train/'+i+'/images/'+k)
      image=trans(image)
      data.append((0,image))
  return data


def unpickle(file): #function found on ciafer webpage used to read batch files
    import pickle
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1

def read_data(path,train=True):
  
  data=[]
  data_dict={}
  if not os.path.isdir(path):
    print("Dataset doesn't exist")
  else:
    if(train):
      for file in ['/data_batch_'+str(i) for i in range(1,6)]:
        dict_temp=unpickle(path+file)
        for j,k in enumerate(dict_temp[b'labels']):
          image=torch.from_numpy(dict_temp[b'data'][j].reshape((1,3,32,32)).astype('float32'))
          data.append((k,image))
    else:
       dict_temp=unpickle(path+'/test_batch')
       for j,k in enumerate(dict_temp[b'labels']):
          image=torch.from_numpy(dict_temp[b'data'][j].reshape((1,3,32,32)).astype('float32'))
          data.append((k,image))
  return data



class data_set(Dataset):
  
  def __init__(self,data,transform=None):
    x=[]
    y=[]
    for j in data:
      x.append(j[1])
      y.append(j[0])
    self.label=torch.tensor(y)
    self.images=torch.cat(x)
    print(self.images.shape)
    self.transform=transform


  def __getitem__(self,index):
    image,label= self.images[index],self.label[index]
    if(self.transform):
      image=self.transform(image)
    return image,label


  def __len__(self):
    return self.label.shape[0]



def sampling(data,alpha,nb_per_client):
  available=np.array([len(i) for i in data.values()]) #calculating nb of the current data available
  nb_samples=available.sum() # total nb
  prior=available/nb_samples # prior distribution for the availble data by category
  prior[prior==0]=0.00000000001 # without this dirchilrt distribution have an error of input 0

  multinomial=np.random.dirichlet(prior*alpha) # the multinomial distribution sampled from dirichlert

  sample=np.random.multinomial(nb_per_client,multinomial) # getting the multinomial distribution for each client
  sample=np.array(sample)
  result=available-sample

  while(np.any(result<0)): ### this loop takes the initial sample and adjust it ( if sample would take all the items in a category and more this more will be distributed on the rest category according to multinomial taken to the updated prior)
    result=result*-1
    result[result<0]=0
    sample=sample-result
    available_now=available-sample

    quantity=result.sum() #quantity to redraw from drichlit distribution
    nb_samples=available_now.sum()#new available

    prior=available_now/nb_samples
    prior[prior==0]=0.00000000001
    multinomial=np.random.dirichlet(prior*alpha)
    resample=np.random.multinomial(quantity,multinomial)

    sample=sample+resample
    result=available-sample
  return sample




def prepare_data_loader(data_m,transform,validation_fraction=0,centralized=True,centralized_testing=True,alpha=1000,nb_clients=10):
  # centralized is for centralized or client server experiments , while testing can take place on the global model or for each client 
  # for centralized experiment only centralized testing makes sense while for client_server there are 2 options divide the validation on the client side 
  # or divide data before assignment to the client for centralized testing , validation distribution in a centralized way should have iid 
  # conditions with the whole data , while for client_server it follows the ditribution of clients

  if(validation_fraction!=0 and (centralized or centralized_testing)): #in case we want to divide the data to training and validation the function will return 2 loaders first for training and second for validation
    index_t,index_v=divide_validation_training(data_m,validation_fraction)
    data_train=[data_m[i] for i in index_t]
    data_val=[data_m[i] for i in index_v]
  else:
    data_train=copy.deepcopy(data_m)
  
  if(centralized):  # keep the list or transform it to dictionary the dictionary is best used for the client distributions
    data=data_train
  else:
    data={}
    for i in data_train:
      if i[0] in data:
        data[i[0]].append(i[1])
      else:
        data[i[0]]=[i[1]]


  loaders=[]
  if(centralized):
      data_set_train=data_set(data_train,transform)
      loaders.append(DataLoader(data_set_train,batch_size=256,shuffle=True,num_workers=4))
  else:
    nb_per_client=sum([len(i) for i in data.values()])/nb_clients # clients have equal amount of data / first experiemnts 
    data=collections.OrderedDict(sorted(data.items()))
    for client in range(nb_clients-1):

      sample_quantity_per_Calss=sampling(data,alpha,nb_per_client)
      data_client_k=[]

      for j,i in enumerate(sample_quantity_per_Calss):
        if(i==0):
          continue

        if(i==len(data[j])):
          data_client_k+=[(j,f) for f in data[j]]
          data[j]=[]
        else:
          for i1 in range(i):
            index=np.random.randint(0,i-i1)
            data_client_k.append((j,data[j].pop(index)))
      data_set_train_client_k=data_set(data_client_k,transform)
      if(centralized_testing or validation_fraction==0):#if centralized testing each client dataset are the training one else they need to be split into training and validation
        loaders.append(DataLoader(data_set_train_client_k,batch_size=256,shuffle=True,num_workers=4))
      else:
        loaders.append(data_set_train_client_k)
      
    data_client_k=[]
    for i in data:
      data_client_k+=[(i,j) for j in data[i]]
    data_set_train_client_k=data_set(data_client_k,transform)

    if(centralized_testing or validation_fraction==0):#if centralized testing each client dataset are the training one else they need to be split into training and validation
      loaders.append(DataLoader(data_set_train_client_k,batch_size=256,shuffle=True,num_workers=4))
    else:
      loaders.append(data_set_train_client_k)
  

  if(validation_fraction!=0 and  (centralized or centralized_testing)):
    data_set_val=data_set(data_val,transform)
    loaders.append(DataLoader(data_set_val,batch_size=256,shuffle=True,num_workers=4))
  else:
    if (validation_fraction!=0 and not centralized_testing):
      train_sets=[]
      val_sets=[]
      for i in loaders:
        index_train,index_val=divide_validation_training(i,validation_fraction)
        train_sets.append(Subset(i,index_train))
        val_sets.append(Subset(i,index_val))
      loaders=[DataLoader(i,batch_size=256,shuffle=True,num_workers=4) for i in train_sets]+[DataLoader(j,batch_size=256,shuffle=True,num_workers=4) for j in val_sets]


  return loaders
  
  
def divide_validation_training(data,fraction=0.2):
  all_indexes= list(range(len(data)))
  np.random.shuffle(all_indexes)
  n=round(len(data)*(fraction))
  val_index=all_indexes[:n]
  train_index=all_indexes[n:]
  
  return train_index,val_index

def EMD(loader): # given a set of loaders calculate the heteroginity index
  pi=[]
  ni=[]
  for i in loader:
    nj=0
    pj=[0]*10
    for image,label in i:
      for z in label:
        pj[z]+=1
        nj+=1
    ni.append(nj)
    pi.append(pj)
  pi=np.array(pi)
  ni=np.array(ni)
  n=ni.sum()
  pi=pi/pi.sum(axis=1).reshape(-1,1)
  p=((ni/n).reshape(-1,1)*pi).sum(axis=0)
  d= ((ni/n)*(np.abs(pi-p).sum(axis=1))).sum()

  return d

