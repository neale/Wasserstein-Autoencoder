import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image as save_img


def save_model(net, optim, epoch, path):
    state_dict = net.state_dict()
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer': optim.state_dict(),
        }, path)


def load_model(net, optim, path):
    print ("==> restoring checkpoint")
    ckpt = torch.load(path)
    epoch = ckpt['epoch']
    net.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    print ("==> loaded checkpoint '{}' (epoch {})".format(path, epoch))
    return net, optim, epoch


def load_data(args):
    trainset = MNIST(root='./data/',
            train=True,
            transform=transforms.ToTensor(),
            download=True)

    testset = MNIST(root='./data/',
            train=False,
            transform=transforms.ToTensor(),
            download=True)

    train_loader = DataLoader(dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True)

    test_loader = DataLoader(dataset=testset,
            batch_size=args.batch_size,
            shuffle=False)
    return (trainset, testset), (train_loader, test_loader)


def sample_noise(args):
    n = args.batch_size
    dist = 'normal'
    scale = 1.
    if dist == 'unifrom':
        noise = np.random.uniform(-1, 1, [n, args.dim]).astype(np.float32)
    elif dist in ('normal', 'sphere'):
        mean = np.zeros(args.dim)
        cov = np.identity(args.dim)
        noise = np.random.multivariate_normal(mean, cov, n).astype(np.float32)
        if dist == 'sphere':
            noise = noise/np.sqrt(np.sum(noise**2, axis=1))[:, np.newaxis]
    noise = torch.from_numpy(noise) * scale

    return noise


def pretrain_loss(encoded, noise):
    mean_g = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_e, mean_g)

    cov_g = torch.matmul((noise-mean_g).transpose(0, 1), noise-mean_g)
    cov_g /= 1000 - 1
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 1000 - 1
    cov_loss = F.mse_loss(cov_e, cov_g)
    return mean_loss + cov_loss


def ae_loss(args, real, reconst):
    if args.loss == 'l2':
        sqr = (real - reconst).pow(2)
        loss = sqr.view(-1, sqr.size(-1)).sum(0)
        loss = 0.2 * torch.sqrt(1e-08 + loss).mean()
    if args.loss == 'l2sq':
        sqr = (real - reconst).pow(2)
        loss = sqr.view(-1, sqr.size(-1)).sum(0)
        loss = 0.05 * loss.mean()
    if args.loss == 'l1':
        abs = (real - reconst).abs()
        loss = sqr.view(-1, abs.size(-1)).sum(0)
        loss = 0.02 * loss.mean()
    return loss


def gan_loss(args, latent, gen, netD):
    loss = nn.BCEWithLogitsLoss()
    logits_e = netD(latent)
    logits_g = netD(gen)

    loss_e = loss(logits_e, torch.zeros_like(logits_e))
    loss_g = loss(logits_g, torch.ones_like(logits_g))
    loss_e_trick = loss(logits_e, torch.ones_like(logits_e))
    loss_adv = args.l * (loss_e + loss_g)
    loss_match = loss_e_trick
    return (loss_adv, logits_e, logits_g), loss_match


def save_image(args, netG, epoch):
    imgs = []
    fixed_noise = torch.randn(100, args.dim)
    noisev = Variable(fixed_noise).cuda()
    samples = netG(noisev)[0]
    samples = samples.view(args.batch_size, 28, 28)
    for sample in samples:
        imgs.append(sample.view(1, 1, 28, 28))
    recons_image = torch.cat(imgs, dim=0)

    if not os.path.isdir('./data/reconst_images'):
        os.makedirs('data/reconst_images')
    save_img(recons_image.data,
        './data/reconst_images/wae_gan_images_%d.png' % (epoch+1), nrow=10)


