import argparse
import torch
import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image

import utils
import encoders
import generators
import discriminators

torch.manual_seed(123)

def load_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')

    parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N', help='')
    parser.add_argument('--dim', type=int, default=2, metavar='N', help='')
    parser.add_argument('--epsilon', type=float, default=1e-15, metavar='N', help='')
    parser.add_argument('--l', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=int, default=0.001, metavar='N', help='')

    args = parser.parse_args()
    return args


def load_networks(args):
    netE = encoders.encoderMNIST(args).cuda()
    netG = generators.generatorMNIST(args).cuda()
    netD = discriminators.discriminatorMNIST(args).cuda()
    return netE, netG, netD
   

def train(args):

    netE, netG, netD = load_networks(args)

    optimizerEnc = optim.Adam(netE.parameters(), lr=args.lr, betas=(.5, .9))
    optimizerEgen = optim.Adam(netE.parameters(), lr=args.lr, betas=(.5, .9))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(.5, .9))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(.5, .9))
    
    schedulerEnc = optim.lr_scheduler.ExponentialLR(optimizerEnc, gamma=0.99)
    schedulerEgen = optim.lr_scheduler.ExponentialLR(optimizerEgen, gamma=0.99)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    
    (trainset, testset), (train_loader, test_loader) = utils.load_data(args)
    
    ae_criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        step = 0
        for i, (images, _) in enumerate(train_loader):
            netG.zero_grad()
            netE.zero_grad()
            netD.zero_grad()
            
            """ Update AutoEncoder """
            images = Variable(images).cuda()
            batch_size = images.size()[0]
            images = images.view(batch_size, -1)
            
            z_sample = netE(images)
            x_sample = netG(z_sample)
            ae_loss = ae_criterion(x_sample+args.epsilon, images+args.epsilon)
            ae_loss.backward()

            optimizerG.step()
            optimizerEnc.step()
            
            # Update Discriminator
            netE.eval()
            z_real_gauss = Variable(torch.randn(images.size()[0], args.dim) * 5.).cuda()
            D_real_gauss = netD(z_real_gauss)

            z_fake_gauss = netE(images)
            D_fake_gauss = netD(z_fake_gauss)

            log_real = torch.log(D_real_gauss + args.epsilon)
            log_fake = torch.log(1 - D_fake_gauss + args.epsilon)
            D_loss = -args.l * torch.mean(log_real + log_fake)
            D_loss.backward()
            optimizerD.step()
            
            # Update Generator
            netE.train()
            z_fake_gauss = netE(images)
            D_fake_gauss = netD(z_fake_gauss)

            G_loss = -args.l * torch.mean(torch.log(D_fake_gauss + args.epsilon))
            G_loss.backward()
            optimizerEgen.step()

            step += 1
            schedulerG.step()
            schedulerD.step()
            schedulerEnc.step()
            schedulerEgen.step()

            if (step + 1) % 100 == 0:
                print("Epoch: %d, Step: [%d/%d], Reconstruction Loss: %.4f, Discriminator Loss: %.4f, Generator Loss: %.4f" %
                      (epoch + 1, step + 1, len(train_loader), ae_loss.data[0], D_loss.data[0], G_loss.data[0]))
        z1 = np.arange(-10, 10, 1.).astype('float32')
        z2 = np.arange(-10, 10, 1.).astype('float32')
        nx, ny = len(z1), len(z2)
        recons_image = []

        for z1_ in z1:
            for z2_ in z2:
                v = Variable(torch.from_numpy(np.asarray([z1_, z2_]))).view(-1, args.dim)
                x = netG(v.cuda()).view(1, 1, 28, 28).cpu()
                recons_image.append(x)
            # x = P(Variable(torch.from_numpy(np.asarray([z1_]))).view(-1, z_dim)).view(1, 1, 28, 28)
        recons_image = torch.cat(recons_image, dim=0)
        
        if not os.path.isdir('./data/reconst_images'):
            os.makedirs('data/reconst_images')
        save_image(recons_image.data, './data/reconst_images/wae_gan_images_%d.png' % (epoch+1), nrow=nx)


if __name__ == '__main__':
    args = load_args()
    train(args)
