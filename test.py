import argparse
import torch
import os
import itertools
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

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
			help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
			help='number of epochs to train (default: 10)')
    parser.add_argument('--dim', type=int, default=2, metavar='N', help='')
    parser.add_argument('--epsilon', type=float, default=1e-15, metavar='N', help='')
    parser.add_argument('--l', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=int, default=.001, metavar='N', help='')
    parser.add_argument('--e_dropout', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--pretrain_epochs', type=int, default=200, metavar='N', help='')
    parser.add_argument('--loss', type=str, default='l2sq', metavar='N', help='')

    args = parser.parse_args()
    return args


def load_networks(args):
    netE = encoders.encoderMNIST(args, True).cuda()
    netG = generators.generatorMNIST(args).cuda()
    netD = discriminators.discriminatorMNIST(args).cuda()
    return netE, netG, netD


def train(args):

    netE, netG, netD = load_networks(args)
    
    ae_params = itertools.chain(netE.parameters(), netG.parameters())

    optimizerE = optim.Adam(netE.parameters(), lr=args.lr, betas=(.5, .9))
    optimizerAE = optim.Adam(netG.parameters(), lr=args.lr, betas=(.5, .9))
    optimizerD = optim.Adam(netD.parameters(), lr=5e-4, betas=(.5, .9))

    schedulerE = optim.lr_scheduler.ExponentialLR(optimizerE, gamma=0.99)
    schedulerAE = optim.lr_scheduler.ExponentialLR(optimizerAE, gamma=0.99)
    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    
    (trainset, testset), (train_loader, test_loader) = utils.load_data(args)

    netE.train()
    netG.train()
    netD.train()
    
    # Pretrain AE 
    def pretrain_e():
        for epoch in range(args.pretrain_epochs):
            for i, (images, _) in enumerate(train_loader):
                images = Variable(images.cuda())
                noise = Variable(utils.sample_noise(args)).cuda()
                latent = netE(images)
                optimizerE.zero_grad()
                loss = utils.pretrain_loss(latent, noise)
                loss.backward()
                optimizerE.step()
            print ("Pretrain Enc epoch: {}, Loss: {}".format(epoch, loss.data[0]))
            if loss.data[0] < 0.1:
                break
        utils.save_model(netE, optimizerE, epoch, './netE_{}.pt'.format(epoch))
       
    #netE, optimizerE, ae_epoch = utils.load_model(netE, optimizerE, './netE_19.pt')
    pretrain_e()

    for epoch in range(args.epochs):
        step = 0
        for i, (images, _) in enumerate(train_loader):
            
            images = Variable(images).cuda()
            e_sample = netE(images)

            gen_reconst, gen_logits = netG(e_sample)
            recon_loss = utils.ae_loss(args, images+1e-15, gen_reconst+1e-15)
            
            z_real = Variable(utils.sample_noise(args)).cuda()
            decoded, decoded_logits = netG(z_real)

            loss_gan, loss_penalty = utils.gan_loss(args, e_sample, z_real, netD)
            
            loss_wae = recon_loss + args.l * loss_penalty

            loss_adv = loss_gan[0]
            netG.zero_grad()
            netE.zero_grad()
            loss_wae.backward(retain_graph=True)
            optimizerAE.step()
            
            netD.zero_grad()
            loss_adv.backward()
            optimizerD.step()
            
            schedulerE.step()
            schedulerAE.step()
            schedulerD.step()

            step += 1
            if (step + 1) % 100 == 0:
                print("Epoch: %d, Step: [%d/%d], AE Loss: %.4f, GAN Loss: %.4f, WAE Loss: %.4f" %
                      (epoch + 1, step + 1, len(train_loader),
                       recon_loss.data[0], loss_adv.data[0],
                       loss_wae.data[0]))
        
        netE.eval()
        if args.dim == 2:
            z1 = np.arange(-10, 10, 1.).astype('float32')
            z2 = np.arange(-10, 10, 1.).astype('float32')
            nx, ny = len(z1), len(z2)
            recons_image = []
            for z1_ in z1:
                for z2_ in z2:
                    v = Variable(torch.from_numpy(np.asarray([z1_, z2_]))).cuda()
                    x = netG(v.view(-1, args.dim))[0].view(1, 1, 28, 28)
                    recons_image.append(x)
            recons_image = torch.cat(recons_image, dim=0)
        else:
            utils.save_image(args, netG, epoch)
    
if __name__ == '__main__':
    args = load_args()
    train(args)
