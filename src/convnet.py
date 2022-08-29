import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F

class ConvNet(nn.Module):
    #This code is heavily inspired by/copied from this tutorial; https://pythonprogramming.net/introduction-deep-learning-neural-network-pytorch/
    def __init__(self, resolution, image_channels, num_classes, stride=1):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(3, 32, stride) # input is 3 channels, 32 output channels, 3x3 kernel / window
        self.conv2 = nn.Conv2d(32, 64, stride) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, stride)

        x = torch.randn(resolution, resolution, image_channels).view(-1, image_channels, resolution, resolution) #Dummy image used in to_linear func
        print(x.shape)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, num_classes) # 512 in, CLASSES out -> number of classes

    def convs(self, x:Tensor):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] #calculates the output that needs to be flattened
        return x

    def forward(self, x:Tensor):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)