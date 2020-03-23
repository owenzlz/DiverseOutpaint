# Import Packages
import warnings
warnings.filterwarnings("ignore")
import os
import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from model_utils import Generator, VEncoder, Discriminator
from data_utils import SyntheticDataset
import numpy as np
import pdb

# Configurations and Hyperparameters
gpu_id = 3
lr_rate = 0.002
num_epochs = 501
num_sample = 6
noise_dim = 2

# Random Initialization
torch.manual_seed(1)
np.random.seed(1)

# Dataloader
action_dir = '../data/train_input.npy'
state_dir = '../data/train_output.npy'
dataset = SyntheticDataset(action_dir, state_dir)
loader = data.DataLoader(dataset, batch_size=10)

# Models
encoder = VEncoder(); encoder.weight_init(mean=0.0, std=0.02)
generator = Generator(); generator.weight_init(mean=0.0, std=0.02)
D_VAE = Discriminator(); D_VAE.weight_init(mean=0.0, std=0.02)
D_LR = Discriminator(); D_LR.weight_init(mean=0.0, std=0.02)

# Initialize Loss
l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()

# Initialize Optimizer
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(0.5, 0.999))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate, betas=(0.5, 0.999))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=lr_rate, betas=(0.5, 0.999))


valid = 1
fake = 0

# Train Networks
step = 0
for epoch in range(num_epochs):
    for i, inputs in enumerate(loader):
        real_A, real_B = inputs

        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------
        
        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        # ----------
        # cVAE-GAN
        # ----------
        
        # encode 
        mu, logvar = encoder(real_B)
        # reparameterization
        dyna_noise = torch.randn(mu.size())
        encoded_z = encoder.sample(mu, logvar, dyna_noise) 
        # generate B
        fake_B = generator(torch.cat([real_A,encoded_z], dim=1))        
        
        # Pixelwise loss of translated image by VAE
        loss_pixel = l1(fake_B, real_B)
        # Kullback-Leibler divergence of encoded B
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        # Adversarial loss
        loss_VAE_GAN = nn.BCEWithLogitsLoss()(torch.squeeze(D_VAE(fake_B)), torch.ones(fake_B.size(0)))
        
        # ---------
        # cLR-GAN
        # ---------        
        # Produce output using sampled z (cLR-GAN)
        sampled_z = torch.randn(mu.size())
        _fake_B = generator(torch.cat([real_A,sampled_z], dim=1))
        # cLR Loss: Adversarial loss
        loss_LR_GAN = nn.BCEWithLogitsLoss()(torch.squeeze(D_LR(_fake_B)), torch.ones(_fake_B.size(0)))
        
        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------
        loss_GE = loss_VAE_GAN + loss_LR_GAN + 10 * loss_pixel + 0.01 * loss_kl
        loss_GE.backward(retain_graph=True)
        optimizer_E.step()
        
        # ---------------------
        # Generator Only Loss
        # ---------------------
        
        # Latent L1 loss
        _mu, _ = encoder(_fake_B)
        loss_latent = 0.01 * l1(_mu, sampled_z)
        loss_latent.backward()
        optimizer_G.step()
        
        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------
        optimizer_D_VAE.zero_grad()
        loss_D_VAE = nn.BCEWithLogitsLoss()(torch.squeeze(D_VAE(real_B)), torch.ones(real_B.size(0))) + nn.BCEWithLogitsLoss()(torch.squeeze(D_VAE(fake_B.detach())), torch.zeros(fake_B.size(0)))
        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------
        optimizer_D_LR.zero_grad()
        loss_D_LR = nn.BCEWithLogitsLoss()(torch.squeeze(D_LR(real_B)), torch.ones(real_B.size(0))) + nn.BCEWithLogitsLoss()(torch.squeeze(D_LR(_fake_B.detach())), torch.zeros(_fake_B.size(0)))
        loss_D_LR.backward()
        optimizer_D_LR.step()

    if epoch%20 == 0:
        if not os.path.exists('models'):os.makedirs('models')  
        torch.save(generator, 'models/generator.pt')
        
    D_VAE_np = loss_D_VAE.data.numpy()
    D_LR_np = loss_D_LR.data.numpy()
    GE_np = loss_GE.data.numpy()
    pixel_np = loss_pixel.data.numpy()
    kl_np = loss_kl.data.numpy()
    print(epoch, 'D_VAE: ', D_VAE_np, 'D_LR: ', D_LR_np, 'GE: ', GE_np, 'pixel: ', pixel_np, 'kl: ', kl_np)




