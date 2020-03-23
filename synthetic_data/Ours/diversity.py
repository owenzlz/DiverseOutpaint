import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torchvision import models

class MeanShift(nn.Conv2d):
	def __init__(
		self, rgb_range,
		rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

		super(MeanShift, self).__init__(3, 3, kernel_size=1)
		std = torch.Tensor(rgb_std)
		self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
		self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
		for p in self.parameters():
			p.requires_grad = False

class VGG(nn.Module):
	def __init__(self, conv_index, rgb_range=1):
		super(VGG, self).__init__()
		vgg_features = models.vgg19(pretrained=True).features
		modules = [m for m in vgg_features]
		if conv_index.find('22') >= 0:
			self.vgg = nn.Sequential(*modules[:8])
		elif conv_index.find('54') >= 0:
			self.vgg = nn.Sequential(*modules[:35])
		# pdb.set_trace()
		vgg_mean = (0.485, 0.456, 0.406)
		vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
		self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
		for p in self.parameters():
			p.requires_grad = False

	def forward(self, img):
		x = self.sub_mean(img)
		x = F.interpolate(img, (224, 224))
		# def _forward(x):
		#     x = self.sub_mean(x)
		#     x = self.vgg(x)
		#     return x
			
		# vgg_sr = _forward(sr)
		# with torch.no_grad():
		#     vgg_hr = _forward(hr.detach())

		# loss = F.mse_loss(vgg_sr, vgg_hr)
		return self.vgg(x)

def compute_pairwise(z):
	return torch.norm(z[:,:,None,:] - z[:,None,:,:], p = 2, dim = 3)

def compute_pair_distance(z, weight = None):
	if weight is not None:
		z_pair_dist = compute_pairwise(weight) * compute_pairwise(z)
	else:
		z_pair_dist = compute_pairwise(z)
	norm_vec_z = torch.sum(z_pair_dist, dim = 2)
	z_pair_dist = z_pair_dist / norm_vec_z[...,None].detach()
	return z_pair_dist

def compute_pair_unnormal_distance(z, weight = None):
	if weight is not None:
		z_pair_dist = compute_pairwise(weight) * compute_pairwise(z)
	else:
		z_pair_dist = compute_pairwise(z)
	norm_vec_z = torch.sum(z_pair_dist, dim = 2)
	z_pair_dist = z_pair_dist
	return z_pair_dist

# z: N x S x C
# x: N x S x C x H x W

def compute_pairwise_divergence(recodes, codes):
	N, k, _ = codes.size()
	z_delta = compute_pair_distance(torch.squeeze(codes).view(N, k, -1))
	x_delta = compute_pair_distance(torch.squeeze(recodes).view(N, k, -1))
	div = F.relu(z_delta*0.8 - x_delta)
	return div.sum()





