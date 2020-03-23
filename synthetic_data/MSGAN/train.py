# Import Packages
import warnings
warnings.filterwarnings("ignore")
import os
import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from model_utils import Decoder, Encoder, Discriminator
from data_utils import SyntheticDataset
import numpy as np
import pdb

def get_z_random(batch, nz):
    z = torch.FloatTensor(batch, nz)
    z.copy_(torch.randn(batch,nz))
    return z

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
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()
decoder.weight_init(mean=0.0, std=0.02)
encoder.weight_init(mean=0.0, std=0.02)
discriminator.weight_init(mean=0.0, std=0.02)

# Initialize Loss
l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()

# Initialize Optimizer
G_optimizer = optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=lr_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(0.5, 0.999))

# Train Networks
step = 0
for epoch in range(num_epochs):
    for i, inputs in enumerate(loader):
        x, y = inputs
        batch = x.shape[0]
        
        c = encoder(x)
        z_1 = get_z_random(batch, noise_dim)
        z_2 = get_z_random(batch, noise_dim)
        
        y_hat_1 = decoder(torch.cat([c, z_1], dim=1))
        y_hat_2 = decoder(torch.cat([c, z_2], dim=1))
        
        y_x_cat = torch.cat([y, x], dim=1)
        y_hat_x_cat_1 = torch.cat([y_hat_1, x], dim=1)
        y_hat_x_cat_2 = torch.cat([y_hat_2, x], dim=1)
        
        ################## Train Discriminator ##################
        D_loss_1 = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_x_cat)), torch.ones(y_x_cat.size(0))) + nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_hat_x_cat_1)), torch.zeros(y_hat_x_cat_1.size(0)))
        D_loss_2 = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_x_cat)), torch.ones(y_x_cat.size(0))) + nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_hat_x_cat_2)), torch.zeros(y_hat_x_cat_2.size(0)))
        D_loss = (D_loss_1 + D_loss_2) / 2
        D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        D_optimizer.step()
        
        ########## G Loss ##########
        G_loss_1 = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_hat_x_cat_1)), torch.ones(y_hat_x_cat_1.size(0)))
        G_loss_2 = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_hat_x_cat_2)), torch.ones(y_hat_x_cat_2.size(0)))
        G_loss = (G_loss_1 + G_loss_2) / 2
        
        ########## Div Loss ##########
        lz = torch.mean(torch.abs(y_hat_2 - y_hat_1)) / torch.mean(torch.abs(z_2 - z_1))
        eps = 1 * 1e-5
        div_loss = 1/(1+lz+eps)

        total_loss = G_loss + div_loss
        G_optimizer.zero_grad()
        total_loss.backward()
        G_optimizer.step()
        
        recon_loss = mse(y_hat_1, y)

    if epoch%20 == 0:
        if not os.path.exists('models'):os.makedirs('models')
        torch.save(encoder, 'models/encoder.pt')    
        torch.save(decoder, 'models/decoder.pt')    
    
    G_loss_np = G_loss.data.numpy()
    D_loss_np = D_loss.data.numpy()
    div_loss_np = div_loss.data.numpy()
    recon_loss_np = recon_loss.data.numpy()
    print(epoch, 'recon_loss_np: ', recon_loss_np, 'G_loss_np: ', G_loss_np, 'D_loss_np: ', D_loss_np, 'div_loss_np: ', div_loss_np)




