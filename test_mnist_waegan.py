import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
from torchvision.datasets import MNIST
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import StepLR
import utils

cuda = True
cnt = 0
lr = 0.001
out_dir = "out_aae3"
batch_size = 100

nc = 1 # number of channels
nz = 8 # size of latent vector
ngf = 128 # decoder (generator) filter factor
ndf = 128 # encoder filter factor
h_dim = 512 # discriminator hidden size
lam = 10 # regulization coefficient

trainset = MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)


def load_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')

    parser.add_argument('--batch-size', type=int, default=100, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--dim', type=int, default=8, help='')
    parser.add_argument('--epsilon', type=float, default=1e-15, help='')
    parser.add_argument('--l', type=int, default=10, help='')
    parser.add_argument('--lr', type=float, default=.001, help='')
    parser.add_argument('--e_dropout', type=bool, default=False, help='')
    parser.add_argument('--resume', type=bool, default=False, help='')
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='')
    parser.add_argument('--loss', type=str, default='l2sq', help='')
    parser.add_argument('--dataset', type=str, default='mnist', help='')


    args = parser.parse_args()
    return args


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # input is Z, going into a convolution
        self.linear = nn.Linear(nz, ngf*8*7*7)

        self.deconv1 = nn.ConvTranspose2d(ngf*8 , ngf*4, 4, 1, 0, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*4)
        self.deconv2 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 1, 0, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.deconv_out = nn.ConvTranspose2d(ngf*2, nc, 4, 2, 0, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('G in: ', x.shape)
        x = x.view(-1, nz) 
        x = F.relu(self.linear(x))
        x = x.view(-1, ngf*8, 7, 7)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv_out(x)
        # print ('G out: ', x.shape)
        x_act = self.sigmoid(x)

        return (x_act, x)


class Encoder(nn.Module):
    def __init__(self, is_training=True):
        super(Encoder, self).__init__()
        self.is_training = is_training
        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf*8) x 4 x 4
        self.linear = nn.Linear(ndf*8, nz)

    def forward(self, x):
        #print ('E in: ', x.shape)
        if self.is_training:
            z = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += z
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        # x = self.conv5(x)
        x = x.view(-1, ndf*8)
        x = self.linear(x)
        #print ('E out: ', x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(nz, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.linear3 = nn.Linear(h_dim, h_dim)
        self.linear4 = nn.Linear(h_dim, h_dim)
        self.logits = nn.Linear(h_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.a = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.logits(x)
        x = self.sigmoid(x)
        return x


def load_networks():
    netE = Encoder().cuda()
    netG = Decoder().cuda()
    netD = Discriminator().cuda()
    print (netE, netG, netD)
    
    return netE, netG, netD


def reset_grad():
    netE.zero_grad()
    netG.zero_grad()
    netD.zero_grad()


def param_switch(module: nn.Module, state):
    if state == "freeze":
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            p.requires_grad = True


args = load_args()

netE, netG, netD = load_networks()

# optimG = optim.Adam(netG.parameters(), lr=lr)
optimE = optim.Adam(netE.parameters(), lr=lr)
optimD = optim.Adam(netD.parameters(), lr=lr*0.5)
optimAE = optim.Adam(list(netE.parameters())+list(netG.parameters()), lr=lr)

# LRgen = StepLR(optimG, step_size=30, gamma=0.5)
LRenc = StepLR(optimE, step_size=30, gamma=0.5)
LRdis = StepLR(optimD, step_size=30, gamma=0.5)
LRae = StepLR(optimAE, step_size=30, gamma=0.5)

print (args.resume)
if args.resume is True:
    print ("\n==> Loading old weights if possible")
    netE, optimE, _ = utils.load_model(netE, optimE, "E_latest.pth")
    netG, optimG, _ = utils.load_model(netG, optimG, "G_latest.pth")
    netD, optimD, _ = utils.load_model(netD, optimD, "D_latest.pth")
    optimAE = utils.load_model(optimAE, "AE_latest.pth")


def pretrain_e():
    print ("==> Pretraining Encoder")
    for epoch in range(args.pretrain_epochs):
        for i, images in enumerate(data_loader):
            x = Variable(images[0].cuda())
            noise = Variable(torch.rand(batch_size, nz)).cuda()
            latent = netE(x).view(batch_size, nz)
            #noise = noise.view(*noise.size(), 1, 1)
            loss = utils.pretrain_loss(latent, noise)
            loss.backward()
            optimE.step()
            reset_grad()
            print ("Pretrain Enc iter: {}, Loss: {}".format(i, loss.data[0]))
            if loss.data[0] < 0.1:
                print ("Finished Pretraining Encoder")
                return

if not args.resume:
    pretrain_e()

one = torch.Tensor([1]).cuda()
mone = (one * -1).cuda()

for epoch in range(args.epochs):
    for batch_idx, batch_images in enumerate(data_loader):
        """ var creation """
        X = Variable(batch_images[0]).cuda()
        z_sample = Variable(torch.randn(batch_size, nz)).cuda()
        
        z_enc = netE(X)
        X_sample, X_logits = netG(z_enc)
        reset_grad()
	
        """ Regularization phase """
        param_switch(netG, "freeze")
        param_switch(netE, "freeze")
        param_switch(netD, "free")
        # z adversary
        D_fake = netD(z_sample)
        z_real = netE(X)
        D_real = netD(z_real)
        D_loss_fake = torch.log(D_fake).mean()
        D_loss_real = torch.log(1-(D_real+args.epsilon)).mean()
        D_loss = D_loss_fake + D_loss_real
        D_loss_fake.backward(mone)
        D_loss_real.backward(mone)
        optimD.step()
       
        """ WAE update """
        param_switch(netG, "free")
        param_switch(netE, "free")
        param_switch(netD, "freeze")

        z_enc = netE(X)

        # recon_loss = utils.ae_loss(args, X+args.epsilon, X_sample+args.epsilon)
        recon_loss = F.mse_loss(X_sample, X)
        penalty = utils.gan_loss2(args, z_sample, z_enc, netD)
        recon_loss.backward(one, retain_graph=True)
        penalty.backward(mone)
        optimAE.step()

        """
        z_fake = netE(X).view(batch_size,-1)
        D_fake = netD(z_fake)

        #G_loss = -torch.mean(torch.log(D_fake))
        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        optimE.step()
        reset_grad()
	"""
        if batch_idx % 50 == 0:
            losses = [D_loss.data[0], penalty.data[0], recon_loss.data[0]]
            print('Epoch {}; iter {}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
                  .format(epoch, batch_idx, losses[0], losses[1], losses[2]))
            utils.save_model(netE, optimE, epoch, "E_latest.pth")
            utils.save_model(netG, optimAE, epoch, "G_latest.pth")
            utils.save_model(netD, optimD, epoch, "D_latest.pth")

        # Print and plot every now and then
        if batch_idx % 100 == 0:
            utils.save_image(netG, epoch, batch_idx, orthogonal=False)
       
