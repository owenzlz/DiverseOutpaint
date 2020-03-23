import warnings
warnings.filterwarnings("ignore")
import argparse, os
import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from vis_tools import *
from models import *
from data_utils import CelebA
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.utils import save_image
import pdb
from time import time
from diversity import VGG
import diversity as div
from tqdm import tqdm

# Configurations and Hyperparameters
port_num = 8081
data_root = '../../../data/celeba'
mode = 'train'
batch_size = 14
gpu_id = 1
lr_rate = 2e-4
num_epochs = 200
report_feq = 50
num_sample = 20
noise_dim = 16

torch.manual_seed(1)
np.random.seed(1)

# Set up data and training gadgets
display = visualizer(port=port_num)

def adjust_lr_rate(optimizer, decay_ratio = 0.1):
    group_num = len(optimizer.param_groups)
    for i in range(group_num):
        optimizer.param_groups[i]["lr"] *= decay_ratio

def denorm(tensor):
	return ((tensor+1.0)/2.0)*255.0

def norm(image):
	return (image/255.0-0.5)*2.0

def collapse_batch(batch):
	if len(batch.shape) == 3:
		_, _, C = batch.size()
		return batch.view(-1, C)
	elif len(batch.shape) == 5:
		_, _, C, H, W = batch.size()
		return batch.view(-1, C, H, W)
	else:
		print("Error: No need to collapse")
		return batch

def uncollapse_batch(batch):
	if len(batch.shape) == 2:
		N, C = batch.size()
		return batch.view(int(N/num_sample), num_sample, C)
	elif len(batch.shape) == 4:
		pdb.set_trace()
		N, C, H, W = batch.size()
		return batch.view(int(N/num_sample), num_sample, C, H, W)
	else:
		print("Error: No need to un-collapse")
		return batch

# 30
def diverse_sampling(code):
	N, C = code.size(0), code.size(1)
	noise = torch.FloatTensor(N, num_sample, noise_dim).uniform_().to(gpu_id)
	code = (code[:,None,:]).expand(-1,num_sample,-1)
	code = torch.cat([code, noise], dim=2)
	return code, noise

transforms = Compose([
				CenterCrop(135),
				Resize(128)
				])

# Initialize Dataset
img_dir = '../../../data/celeba/images/'
mask_dir = '../../../data/celeba/masks/eye_nose/'
dataset = CelebA('test', img_dir, mask_dir, transforms = transforms)
loader = data.DataLoader(dataset, batch_size=1)

# Load models
encoder = torch.load('models/encoder_160000.pt')
decoder = torch.load('models/decoder_160000.pt')
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
	
	save_image(denorm(low), '../Eval/input_v2/'+str(idx)+'.png', nrow=1, padding=0,normalize=True)
	save_image(denorm(high), '../Eval/GT_v2/'+str(idx)+'.png', nrow=1, padding=0,normalize=True)
	for i in range(high_hat.shape[0]):
		save_image(denorm(high_hat[i].unsqueeze(0)), '../Eval/multiscale_reg_div_v2/'+str(idx)+'_'+str(i)+'.png', nrow=1, padding=0, normalize=True)
	
	print(idx)

	# if idx == 100:
	# 	break





