import torch
import torch.nn as nn
from torch.nn import functional as F


class encoderMNIST(nn.Module):
    def __init__(self, args, is_training):
        super(encoderMNIST, self).__init__()
        self.dim = args.dim
        self.in_c = 1
        self.n_filters = 1024
        self.is_training = is_training

        self.linear = nn.Linear(3*3*1024, self.dim)
        encoders = []
        for i in range(4):
            scale = 2**(4 - i - 1)
            conv = nn.Sequential(
                nn.Conv2d(self.in_c, 1024//scale, 4, 2, 2),
                nn.BatchNorm2d(1024//scale),
                nn.ReLU(inplace=True))
            encoders.append(conv)
            self.in_c = 1024//scale
        self.encoders = nn.Sequential(*encoders)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # print ('e in: ', x.size())
        if self.is_training:
            z = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += z
        x = x.view(-1, 1, 28, 28)
        x = self.encoders(x)
        x = x.view(-1, 3*3*1024)
        x = self.linear(x)
        # print ('e out: ', x.size())
        return x

"""
class encoderMNIST(nn.Module):
    def __init__(self, args):
        super(encoderMNIST, self).__init__()
        self.dim = args.dim
        self.dropout = args.e_dropout

        self.conv1 = nn.Conv2d(1, 100, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 200, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200, 400, 5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(400)
        self.lin1 = nn.Linear(4*4*4*100, args.dim)

    def forward(self, x):
        # print ('q in', x.size())
        x = x.view(-1, 1, 28, 28)
        if self.dropout:
            x = F.dropout(self.conv1(x), p=0.25, training=self.training)
            x = F.leaky_relu(x)
            x = F.dropout(self.conv2(x), p=0.25, training=self.training)
            x = F.leaky_relu(x)
            x = F.dropout(self.conv3(x), p=0.25, training=self.training)
            x = F.leaky_relu(x)
            x = x.view(-1, 4*4*4*100)
            xgauss = self.lin1(x)
            xgauss = xgauss.view(-1, self.dim)
            # print ('q out', xgauss.size())
        else: 
            x = self.conv1(x)
            x = F.leaky_relu(self.bn1(x))
            x = self.conv2(x)
            x = F.leaky_relu(self.bn2(x))
            x = self.conv3(x)
            x = F.leaky_relu(self.bn3(x))
            x = x.view(-1, 4*4*4*100)
            x = self.lin1(x)
            xgauss = x.view(-1, self.dim)
            # print ('q out', xgauss.size())
        return xgauss
"""
