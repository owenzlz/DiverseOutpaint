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
import diversity as div
import numpy as np
import pdb

# noise sampling
def noise_sampling(batch_size, num_sample, noise_dim):
	noise = torch.FloatTensor(batch_size, num_sample, noise_dim).uniform_()
	return noise

# Configurations and Hyperparameters
gpu_id = 3
lr_rate = 0.002
num_epochs = 1001
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
        
        ########## Encode #########
        x_unsqueeze = (x[:,None,:]).expand(-1,num_sample,-1)
        x_unsqueeze = x_unsqueeze.contiguous().view(-1,x_unsqueeze.shape[2])
        z = encoder(x_unsqueeze)
        
        ########## Diverse Sampling in Uniform Space ########
        batch_size = x.shape[0]
        noise = noise_sampling(batch_size, num_sample, noise_dim)
        
        ########## Decode ########
        z_c_concat = torch.cat([z, noise.view(-1, noise.size(2))], dim=1)
        y_hat = decoder(z_c_concat)        

        y_x_cat = torch.cat([y, x], dim=1)
        y_hat_x_cat = torch.cat([y_hat, x_unsqueeze], dim=1)
        
        ################## Train Discriminator ##################       
        D_loss = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_x_cat)), torch.ones(y_x_cat.size(0))) + nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_hat_x_cat)), torch.zeros(y_hat_x_cat.size(0)))
        D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        D_optimizer.step()
        
        ########## Recon Loss ##########
        y_unsqueeze = (y[:,None,:]).expand(-1,num_sample,-1)
        y_unsqueeze = y_unsqueeze.contiguous().view(-1,y_unsqueeze.shape[2])        
        recon_loss = mse(y_hat, y_unsqueeze)
        
        ########## G Loss ##########
        G_loss = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(y_hat_x_cat)), torch.ones(y_hat_x_cat.size(0)))
        
        ########## Div Loss ##########
        pair_div_loss = div.compute_pairwise_divergence(y_hat.view(batch_size,num_sample,-1), noise)
        total_loss = G_loss + 0.1*pair_div_loss
        G_optimizer.zero_grad()
        total_loss.backward()
        G_optimizer.step()

    if epoch%20 == 0:
        if not os.path.exists('models'):os.makedirs('models')
        torch.save(encoder, 'models/encoder.pt')
        torch.save(decoder, 'models/decoder.pt')    
    
    G_loss_np = G_loss.data.numpy()
    D_loss_np = D_loss.data.numpy()
    pair_div_loss_np = pair_div_loss.data.numpy()
    recon_loss_np = recon_loss.data.numpy()
    print(epoch, 'recon_loss_np: ', recon_loss_np, 'G_loss_np: ', G_loss_np, 'D_loss_np: ', D_loss_np, 'pair_div_loss_np: ', pair_div_loss_np)




