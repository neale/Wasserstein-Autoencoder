import torch
import torch.nn as nn
from torch.nn import functional as F


class discriminatorMNIST(nn.Module):
    def __init__(self, args):
        super(discriminatorMNIST, self).__init__()
        self.lin1 = nn.Linear(args.dim, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 512)
        self.lin4 = nn.Linear(512, 512)
        self.lin5 = nn.Linear(512, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()


    def forward(self, x):
        # print ('d in: ', x.size())
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        x = F.relu(x)
        x = self.lin5(x)
        # x = F.sigmoid(x)
        # print ('d out:', x.size())
        return x


