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
mode = 'train'
port_num = 8092
batch_size = 16
gpu_id = 1
lr_rate = 2e-4
num_epochs = 200
report_feq = 50
num_sample = 4
noise_dim = 16

# Set up data and training gadgets
display = visualizer(port=port_num)

# Initialize Dataset
img_dir = '/home/zlz/data/images/'
mask_dir = '/home/zlz/data/Face_data/masks/eye_nose/'
transforms = Compose([CenterCrop(135),Resize(128)])
dataset = CelebA(mode, img_dir, mask_dir, transforms = transforms)
loader = data.DataLoader(dataset, batch_size=batch_size)

# Initialize Model
encoder, decoder, discriminator = Encoder().to(gpu_id), Decoder(noise_dim=noise_dim).to(gpu_id), PSPDiscriminator().to(gpu_id)

# Initialize Loss
l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()

# Initialize Optimizer
G_optimizer = optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=lr_rate)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr_rate)

# Adversarial loss
valid = 1
fake = 0

# Training
encoder, decoder, discriminator = encoder.train(), decoder.train(), discriminator.train()
total_steps = len(loader)*num_epochs
step = 0
high_recon_acc = 0
low_recon_acc = 0
g_loss_acc = 0
d_loss_acc = 0
div_loss_acc = 0
for e in range(num_epochs):
	for idx, images in enumerate(tqdm(loader)):
		high, mask = images
		N, C, H, W = high.shape
		high, mask = high.to(gpu_id), mask.to(gpu_id)
		mask[mask>0] = 1
		low = high*mask
		high, low = norm(high), norm(low)
		
		########## Encode LR ##########
		codes, feats = encoder(low)
		
		#######################################################
		################### Centroid Sample ###################
		#######################################################

		########## Diverse Sampling on Paired Images ########
		diverse_codes, center_noises = center_sampling(codes, num_sample, noise_dim, gpu_id)
		diverse_codes, center_noises = diverse_codes[...,None,None], center_noises[...,None,None]
		
		########## Decode HR for diversification objective ##########
		high_hat = decoder(diverse_codes.view(-1, diverse_codes.size(2), 1, 1), [collapse_batch((feat[:,None,...]).expand(-1, num_sample, -1, -1, -1).contiguous()) for feat in feats])
		mask = (mask[:,None,:]).expand(-1,num_sample,-1,-1,-1)
		mask = mask.contiguous().view(-1,mask.shape[2],mask.shape[3],mask.shape[4])
		high_unsqueeze = (high[:,None,:]).expand(-1,num_sample,-1,-1,-1)
		high_unsqueeze = high_unsqueeze.contiguous().view(-1,high_unsqueeze.shape[2],high_unsqueeze.shape[3],high_unsqueeze.shape[4])
		high_center_hat = (1-mask)*high_hat + mask*high_unsqueeze
		
		################## Train Discriminator ##################
		D_loss = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(high)), torch.ones(high.size(0),).to(gpu_id)) + nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(high_center_hat)), torch.zeros(high_center_hat.size(0),).to(gpu_id))
		D_optimizer.zero_grad()
		D_loss.backward(retain_graph=True)
		D_optimizer.step()
		
		########## Reconstruction Loss ##########
		high_recon = torch.nn.MSELoss()(high_center_hat.view(N,num_sample,3,H,W), high.view(N,1,3,H,W))

		########## G Loss ##########
		G_loss = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(high_center_hat)), torch.ones(high_center_hat.size(0),).to(gpu_id))
		
		########## Div Loss ##########
		total_loss = G_loss + high_recon
		G_optimizer.zero_grad()
		total_loss.backward(retain_graph=True)
		G_optimizer.step()

		#######################################################
		################### Random Sample #####################
		#######################################################

		########## Diverse Sampling on Paired Images ########
		diverse_codes, noises = diverse_sampling(codes, num_sample, noise_dim, gpu_id)
		diverse_codes, noises = diverse_codes[...,None,None], noises[...,None,None]
		
		########## Decode HR for diversification objective ##########
		high_hat = decoder(diverse_codes.view(-1, diverse_codes.size(2), 1, 1), [collapse_batch((feat[:,None,...]).expand(-1, num_sample, -1, -1, -1).contiguous()) for feat in feats])
		high_unsqueeze = (high[:,None,:]).expand(-1,num_sample,-1,-1,-1)
		high_unsqueeze = high_unsqueeze.contiguous().view(-1,high_unsqueeze.shape[2],high_unsqueeze.shape[3],high_unsqueeze.shape[4])
		high_hat = (1-mask)*high_hat + mask*high_unsqueeze

		################## Train Discriminator ##################
		D_loss = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(high)), torch.ones(high.size(0),).to(gpu_id)) + nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(high_hat)), torch.zeros(high_hat.size(0),).to(gpu_id))
		D_optimizer.zero_grad()
		D_loss.backward(retain_graph=True)
		D_optimizer.step()
		
		########## G Loss ##########
		G_loss = nn.BCEWithLogitsLoss()(torch.squeeze(discriminator(high_hat)), torch.ones(high_hat.size(0),).to(gpu_id))
		
		########## Div Loss ##########
		pair_div_loss = div.compute_pairwise_divergence(high_hat.view(N,num_sample,3,H,W), noises)
		total_loss = G_loss + 0.5*pair_div_loss
		G_optimizer.zero_grad()
		total_loss.backward()
		G_optimizer.step()
		
		# high_recon_acc += high_recon.item()
		g_loss_acc += G_loss.item()
		div_loss_acc += pair_div_loss.item()
		d_loss_acc  += D_loss.item()
		
		step += 1
		
		if step % report_feq == 0:
			lr = [denorm(low[0].detach()).cpu().numpy().astype(np.uint8)]
			hr = [denorm(high[0].detach()).cpu().numpy().astype(np.uint8)]
			sr = [denorm(high_hat[i].detach()).cpu().numpy().astype(np.uint8) for i in range(4)]
			display.img_result(lr, win=1, caption="low")
			display.img_result(hr, win=2, caption="high")
			display.img_result(sr, win=3, caption="sr")
			print('epoch:{}, iter:{}, G_loss:{:.4f}, D_loss:{:.4f}, div_loss:{:.4f}'.format(e, idx, g_loss_acc / report_feq, d_loss_acc / report_feq, div_loss_acc / report_feq))
			# high_recon_acc = 0
			g_loss_acc = 0
			d_loss_acc = 0
			div_loss_acc = 0

	########## Save Models At Each Epoch ##########
	if not os.path.exists('models'): os.mkdir('models')
	torch.save(encoder, 'models/encoder_'+str(epoch)+'.pt')
	torch.save(decoder, 'models/decoder_'+str(epoch)+'.pt')
	torch.save(discriminator, 'models/discriminator_'+str(step)+'.pt')
	torch.save({
			'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
            }, 'models/checkpoint.tar')

	if e % 2 == 0 and e != 0:
		adjust_lr_rate(G_optimizer)
		adjust_lr_rate(D_optimizer)






