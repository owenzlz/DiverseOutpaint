# Import Packages
import warnings
warnings.filterwarnings("ignore")
import os
import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from model_utils import Decoder, Encoder
from data_utils import SyntheticDataset
import matplotlib.pyplot as plt
from skimage.io import imsave
import numpy as np
import pdb

def get_z_random(batch, nz):
    z = torch.FloatTensor(batch, nz)
    z.copy_(torch.randn(batch,nz))
    return z

# Configurations and Hyperparameters
gpu_id = 3
lr_rate = 2e-4
num_epochs = 101
num_sample = 6
noise_dim = 2

# Random Initialization
torch.manual_seed(1)
np.random.seed(1)

# Dataloader
action_dir = '../data/test_input.npy'
state_dir = '../data/test_output.npy'
dataset = SyntheticDataset(action_dir, state_dir)
loader = data.DataLoader(dataset, batch_size=1)

# Models
encoder = torch.load('models/encoder.pt')
decoder = torch.load('models/decoder.pt')

# Inference and draw the datapoints
clen = 150
n_samples = len(loader)
x_arr = np.zeros((n_samples, 2))
y_arr = np.zeros((n_samples, 2))
x_canvas = np.ones((clen,clen,3))*255
y_canvas = np.ones((clen,clen,3))*255
y_hat_canvas = np.ones((clen,clen,3))*255
mode_map = np.zeros((y_hat_canvas.shape[0], y_hat_canvas.shape[1]))
for i, inputs in enumerate(loader):
    x, y = inputs
    batch = x.shape[0]

    x_np = x.data.numpy()[0]
    x_vec = (int(x_np[0]+int(clen/2)), int(x_np[1]+int(clen/2)))
    x_canvas[x_vec[0], x_vec[1]] = (73,153,223)       
    
    y_np = y.data.numpy()[0]
    y_vec = (int(y_np[0]+int(clen/2)), int(y_np[1]+int(clen/2)))
    y_canvas[y_vec[0], y_vec[1]] = (27,192,106)   

    c = encoder(x)
    
    for _ in range(10):
        z = get_z_random(batch, noise_dim)
        y_hat = decoder(torch.cat([c, z], dim=1)) 
        
        y_hat_np = y_hat.data.numpy()[0]
        y_hat_vec = (int(y_hat_np[0]+int(clen/2)), int(y_hat_np[1]+int(clen/2)))
        y_hat_canvas[y_hat_vec[0], y_hat_vec[1]] = (27,192,106)
        mode_map[y_hat_vec[0], y_hat_vec[1]] = 1
        
    print(i)

plt.subplot(1,3,1)
plt.imshow(x_canvas/255)
plt.subplot(1,3,2)
plt.imshow(y_canvas/255)        
plt.subplot(1,3,3)
plt.imshow(y_hat_canvas/255)

print('The number of modes: ', np.sum(mode_map))

#imsave('input.png', x_canvas/255)
#imsave('output.png',y_canvas/255)
#imsave('msgan.png', y_hat_canvas/255)



