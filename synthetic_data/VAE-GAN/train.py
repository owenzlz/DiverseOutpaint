# Import Packages
import warnings
warnings.filterwarnings("ignore")
import os
import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from model_utils import Decoder, VEncoder, Discriminator
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
vencoder = VEncoder()
decoder = Decoder()
discriminator = Discriminator()
decoder.weight_init(mean=0.0, std=0.02)
vencoder.weight_init(mean=0.0, std=0.02)
discriminator.weight_init(mean=0.0, std=0.02)

# Initialize Loss
l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()

# Initialize Optimizer
G_optimizer = optim.Adam([{'params': vencoder.parameters()}, {'params': decoder.parameters()}], lr=lr_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(0.5, 0.999))

# Train Networks
step = 0
for epoch in range(num_epochs):
    for i, inputs in enumerate(loader):
        x, y = inputs
        
        mu, logvar = vencoder(x)
        dyna_noise = torch.randn(mu.size())
        dyna_code = vencoder.sample(mu, logvar, dyna_noise)
        y_hat = decoder(dyna_code)   
        
        kl_div = vencoder.kl_div(mu, logvar)
        recon_loss = mse(y_hat, y)

        ################## Train Discriminator ##################
        D_loss = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y)), torch.ones(y.size(0))) + nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_hat)), torch.zeros(y_hat.size(0)))
        D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        D_optimizer.step()
        
        ########## G Loss ##########
        G_loss = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_hat)), torch.ones(y_hat.size(0)))
        
        ########## Div Loss ##########
        total_loss = recon_loss + G_loss + 0.01*kl_div
        G_optimizer.zero_grad()
        total_loss.backward()
        G_optimizer.step()

    if epoch%20 == 0:
        if not os.path.exists('models'):os.makedirs('models')
        torch.save(vencoder, 'models/vencoder.pt')    
        torch.save(decoder, 'models/decoder.pt')    
    
    G_loss_np = G_loss.data.numpy()
    D_loss_np = D_loss.data.numpy()
    kl_div_np = kl_div.data.numpy()
    recon_loss_np = recon_loss.data.numpy()
    print(epoch, 'recon_loss_np: ', recon_loss_np, 'G_loss_np: ', G_loss_np, 'D_loss_np: ', D_loss_np, 'kl_div_np: ', kl_div_np)




