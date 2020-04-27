## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 5) # 1,224,224 --> 32,220,220
        I.xavier_uniform_(self.conv1.weight)  #Xavier initialization
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        I.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(64)  #Batch Normalization
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        I.xavier_uniform_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        I.xavier_uniform_(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, 3)
        I.xavier_uniform_(self.conv5.weight)
        self.conv5_bn = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 1024, 3)
        I.xavier_uniform_(self.conv6.weight)
        self.conv6_bn = nn.BatchNorm2d(1024)
        
        self.drop1 = nn.Dropout(p=0.2) 
        self.drop2 = nn.Dropout(p=0.3)
        self.drop3 = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(1024, 500)
        I.xavier_uniform_(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm1d(500)
        
        self.fc2 = nn.Linear(500, 136)
        I.xavier_uniform_(self.fc2.weight)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x))) #1,224,224 --> 32,220,220 --> 32,110,110
        x = self.drop1(self.pool(F.relu(self.conv2_bn(self.conv2(x))))) #32,110,110 --> 64,108,108 --> 64,54,54
        x = self.drop1(self.pool(F.relu(self.conv3_bn(self.conv3(x))))) #64,54,54 --> 128,52,52 --> 128, 26, 26
        x = self.drop2(self.pool(F.relu(self.conv4_bn(self.conv4(x))))) #128,26,26 -->256, 24, 24, --> 256,12,12
        x = self.drop2(self.pool(F.relu(self.conv5_bn(self.conv5(x))))) #256,12,12 --> 512,10,10 --> 512,5,5
        x = self.drop2(self.pool(F.relu(self.conv6_bn(self.conv6(x))))) #512,5,5 --> 1024,3,3 --> 1024,1,1
        x = x.view(x.size(0), -1)  
        x = self.drop3(F.relu(self.fc1_bn(self.fc1(x)))) #1024 --> 500
        x = self.fc2(x) #500 --> 136
        # a modified x, having gone through all the layers of your model, should be returned
        return x
