import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self,input_channel,output_size):
        super(LeNet,self).__init__()
              
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=6,kernel_size=(5,5),padding=0,stride=1)
        
        #Start implementation of Layer 3 (C3) which has 16 kernels of size 5x5 with padding 0 and stride 1
        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16,kernel_size = (5,5),padding=0,stride=1)
        
        #Start implementation of Layer 5 (C5) which is basically flattening the data 
            
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120,kernel_size = (5,5),padding=0,stride=1)
        
        #Start implementation of Layer 6 (F6) which has 85 Linear Neurons and input of 120
        
        self.L1 = nn.Linear(120,84)
        
        #Start implementation of Layer 7 (F7) which has 10 Linear Neurons and input of 84
        
        self.L2 = nn.Linear(84,output_size)
        
        #We have used pooling of size 2 and stride 2 in this architecture 
        
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        #We have used tanh as an activation function in this architecture so we will use tanh at all layers excluding F7.
        self.act = nn.Tanh()
        
    #Now we will implement forward function to produce entire flow of the architecture.
    
    def forward(self,x):
        x = self.conv1(x)
        #We have used tanh as an activation function in this architecture so we will use tanh at all layers excluding F7.
        x = self.act(x)
        #Now this will be passed from pooling 
        x = self.pool(x)
        #Next stage is convolution
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        #next we will pass from conv3, here we will not pass data from pooling as per Architecture 
        x = self.conv3(x)
        x = self.act(x)
        
        #Now the data should be flaten and it would be passed from FC layers. 
        x = x.view(x.size()[0], -1)
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        
        
        return x
        

class LeNet_BatchNorm(nn.Module):
    def __init__(self,input_channel,output_size):
        super(LeNet_BatchNorm,self).__init__()
              
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=6,kernel_size=(5,5),padding=0,stride=1)
        self.BN1=nn.BatchNorm2d(6)
        #Start implementation of Layer 3 (C3) which has 16 kernels of size 5x5 with padding 0 and stride 1
        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16,kernel_size = (5,5),padding=0,stride=1)
        self.BN2=nn.BatchNorm2d(16)
        #Start implementation of Layer 5 (C5) which is basically flattening the data 
            
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120,kernel_size = (5,5),padding=0,stride=1)
        self.BN3=nn.BatchNorm2d(120)
        #Start implementation of Layer 6 (F6) which has 85 Linear Neurons and input of 120
        
        self.L1 = nn.Linear(120,84)
        
        #Start implementation of Layer 7 (F7) which has 10 Linear Neurons and input of 84
        
        self.L2 = nn.Linear(84,output_size)
        
        #We have used pooling of size 2 and stride 2 in this architecture 
        
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        #We have used tanh as an activation function in this architecture so we will use tanh at all layers excluding F7.
        self.act = nn.Tanh()
        
    #Now we will implement forward function to produce entire flow of the architecture.
    
    def forward(self,x):
        x = self.BN1(self.conv1(x))
        #We have used tanh as an activation function in this architecture so we will use tanh at all layers excluding F7.
        x = self.act(x)
        #Now this will be passed from pooling 
        x = self.pool(x)
        #Next stage is convolution
        x = self.BN2(self.conv2(x))
        x = self.act(x)
        x = self.pool(x)
        #next we will pass from conv3, here we will not pass data from pooling as per Architecture 
        x = self.BN3(self.conv3(x))
        x = self.act(x)
        
        #Now the data should be flaten and it would be passed from FC layers. 
        x = x.view(x.size()[0], -1)
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        
        
        return x