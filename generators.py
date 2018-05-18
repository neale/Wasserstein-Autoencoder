import torch
import torch.nn as nn
from torch.nn import functional as F


class generatorMNIST(nn.Module):
    def __init__(self, args):
        super(generatorMNIST, self).__init__()
        self.in_c = 1024
        self.linear = nn.Sequential(
                nn.Linear(args.dim, 8*8*1024),
                nn.ReLU(inplace=True))
        generators = []
        for i in range(2):
            scale = 2**(i + 1)
            deconv = nn.Sequential(
                    nn.ConvTranspose2d(self.in_c, 1024//scale, 4, 2, 2, 1),
                    nn.BatchNorm2d(1024//scale),
                    nn.ReLU(inplace=True))
            generators.append(deconv)
            self.in_c = 1024//scale
        self.generators = nn.Sequential(*generators)
        self.deconv = nn.ConvTranspose2d(self.in_c, 1, 4, stride=1, padding=2)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        # print ('p in: ', x.size())
        x = self.linear(x)
        x = x.view(-1, 1024, 8, 8)
        x = self.generators(x)
        x = self.deconv(x)
        x_out = F.sigmoid(x)
        # print ('P out: ', x.size())
        return x_out, x

