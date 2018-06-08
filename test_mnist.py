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
import utils

cuda = True
cnt = 0
lr = 1e-4
out_dir = "out_aae3"
batch_size = 256

nc = 1 # number of channels
nz = 8 # size of latent vector
ngf = 64 # decoder (generator) filter factor
ndf = 64 # encoder filter factor
h_dim = 256 # discriminator hidden size
lam = 10 # regulization coefficient

trainset = MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)


def load_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')

    parser.add_argument('--batch-size', type=int, default=100, help='')
    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--dim', type=int, default=64, metavar='N', help='')
    parser.add_argument('--epsilon', type=float, default=1e-15, metavar='N', help='')
    parser.add_argument('--l', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=float, default=.0001, metavar='N', help='')
    parser.add_argument('--e_dropout', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--resume', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--pretrain_epochs', type=int, default=200, metavar='N', help='')
    parser.add_argument('--loss', type=str, default='l2sq', metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='celeba', metavar='N', help='')


    args = parser.parse_args()
    return args


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # input is Z, going into a convolution
        self.deconv1 = nn.ConvTranspose2d(nz, ngf*8, 5, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.a1 = nn.ReLU(True)
        self.deconv2 = nn.ConvTranspose2d(ngf*8, ngf*4, 5, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.a2 = nn.ReLU(True)
        self.deconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.a3 = nn.ReLU(True)
        self.deconv4 = nn.ConvTranspose2d(ngf * 2, ngf, 5, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.a4 = nn.ReLU(True)
        self.deconv5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print ('G in: ', x.shape)
        x = self.a1(self.bn1(self.deconv1(x)))
        x = self.a2(self.bn2(self.deconv2(x)))
        x = self.a3(self.bn3(self.deconv3(x)))
        x = self.a4(self.bn4(self.deconv4(x)))
        x = self.deconv5(x)
        #print ('G out: ', x.shape)
        x = self.sigmoid(x)

        return x

class Encoder(nn.Module):
    def __init__(self, is_training=True):
        super(Encoder, self).__init__()
        self.is_training = is_training
        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(nc, ndf, 3, 2, 2, bias=False)
        self.a1 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 3, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.a2 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.a3 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.a4 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(ndf * 8, nz, 3, 1, 0, bias=False)

    def forward(self, x):
        #print ('E in: ', x.shape)
        if self.is_training:
            z = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += z
        x = self.conv1(x)
        x = self.a1(x)
        x = self.a2(self.bn2(self.conv2(x)))
        x = self.a3(self.bn3(self.conv3(x)))
        x = self.a4(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        #print ('E out: ', x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(nz, h_dim)
        self.a1 = nn.ReLU()
        self.linear2 = nn.Linear(h_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.a1(self.linear1(x))
        x = self.linear2(x)
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


args = load_args()

netE, netG, netD = load_networks()

optimE = optim.Adam(netE.parameters(), lr=lr)
optimG = optim.Adam(netG.parameters(), lr=lr)
optimD = optim.Adam(netD.parameters(), lr=lr*0.1)

if args.resume:
    print ("\n==> Loading old weights if possible")
    netE, optimE, _ = utils.load_model(netE, optimE, "E_latest.pth")
    netG, optimG, _ = utils.load_model(netG, optimG, "G_latest.pth")
    netD, optimD, _ = utils.load_model(netD, optimD, "D_latest.pth")


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

pretrain_e()
for it in range(100000):

    for batch_idx, batch_item in enumerate(data_loader):
        #X = sample_X(mb_size)
        """ Reconstruction phase """
        X = Variable(batch_item[0]).cuda()
        z_sample = netE(X)
        X_sample = netG(z_sample)
        recon_loss = F.mse_loss(X_sample, X)

        recon_loss.backward()
        optimG.step()
        optimE.step()
        reset_grad()

        """ Regularization phase """
        # Discriminator
        for _ in range(5):
            z_real = Variable(torch.randn(batch_size, nz)).cuda()
            z_fake = netE(X).view(batch_size,-1)

            D_real = netD(z_real)
            D_fake = netD(z_fake)

            #D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward()
            optimD.step()

            # Weight clipping
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            reset_grad()

        # Generator
        z_fake = netE(X).view(batch_size,-1)
        D_fake = netD(z_fake)

        #G_loss = -torch.mean(torch.log(D_fake))
        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        optimE.step()
        reset_grad()

        if batch_idx % 50 == 0:
            print('Iter {}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
                  .format(batch_idx, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))
            utils.save_model(netE, optimE, it, "E_latest.pth")
            utils.save_model(netG, optimG, it, "G_latest.pth")
            utils.save_model(netD, optimD, it, "D_latest.pth")

        # Print and plot every now and then
        if batch_idx % 100 == 0:
            utils.save_image(netG, it, orthogonal=False)
       
