import warnings
warnings.filterwarnings("ignore")
from torch import nn, optim
from torch.utils import data
from data_utils import CelebA
import diversity as div
from time import time
from tqdm import tqdm
import argparse, os
import torch
import pdb
from data_utils import *
from vis_tools import *
from models import *
from utils import *

# Configurations and Hyperparameters
mode = 'test'
gpu_id = 1
num_sample = 1
noise_dim = 16

# Initialize Dataset
img_dir = '/home/zlz/data/images/'
mask_dir = '/home/zlz/data/Face_data/masks/eye_nose/'
transforms = Compose([CenterCrop(135),Resize(128)])
dataset = CelebA(mode, img_dir, mask_dir, transforms = transforms)
loader = data.DataLoader(dataset, batch_size=batch_size)

# Load Model
encoder = torch.load('models/encoder_100000.pt')
decoder = torch.load('models/decoder_100000.pt')
encoder, decoder = encoder.to(gpu_id), decoder.to(gpu_id)

# Training
for idx, images in enumerate(loader):
	high, mask = images
	N, C, H, W = high.shape
	high, mask = high.to(gpu_id), mask.to(gpu_id)
	mask[mask>0] = 1
	low = high*mask
	high, low = norm(high), norm(low)
	
	########## Encode LR ##########
	codes, feats = encoder(low)
	
	########## Diverse Sampling on Paired Images ########
	diverse_codes, noises = diverse_sampling(codes)
	diverse_codes, noises = diverse_codes[...,None,None], noises[...,None,None]
	
	########## Decode HR for diversification objective ##########
	high_hat = decoder(diverse_codes.view(-1, diverse_codes.size(2), 1, 1), [collapse_batch((feat[:,None,...]).expand(-1, num_sample, -1, -1, -1).contiguous()) for feat in feats])
	mask = (mask[:,None,:]).expand(-1,num_sample,-1,-1,-1)
	mask = mask.contiguous().view(-1,mask.shape[2],mask.shape[3],mask.shape[4])
	
	high_unsqueeze = (high[:,None,:]).expand(-1,num_sample,-1,-1,-1)
	high_unsqueeze = high_unsqueeze.contiguous().view(-1,high_unsqueeze.shape[2],high_unsqueeze.shape[3],high_unsqueeze.shape[4])

	high_hat = (1-mask)*high_hat + mask*high_unsqueeze
	
	save_image(denorm(high), '../Eval/GT/'+str(idx)+'.png', nrow=1, padding=0,normalize=True)
	save_image(denorm(high_hat), '../Eval/multiscale_reg/'+str(idx)+'.png', nrow=1, padding=0,normalize=True)

	print(idx)

	# pdb.set_trace()


