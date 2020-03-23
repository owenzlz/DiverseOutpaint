import torch
from torch.utils import data
from torchvision.transforms import functional as tvf
from torchvision.transforms import Compose, Resize, CenterCrop
import numpy as np
from PIL import Image
import pdb
import time
import glob

class CelebA(data.Dataset):
	def __init__(self, mode, img_dir, mask_dir, transforms=None):
		image_list = []
		mask = []
		for img_file in glob.glob(str(img_dir)+'*'):
			image_list.append(img_file)
			name = img_file.split('/')[-1].split('.')[0]
			mask.append(str(mask_dir)+str(name)+'.png')
		
		if mode == 'train': 
			self.image_list = image_list[:200000]
			self.mask = mask[:200000]
		else:
			self.image_list = image_list[201000:202000]
			self.mask = mask[201000:202000]			
		self.transforms = transforms
		
	def __getitem__(self, index):
		image = Image.open(self.image_list[index])
		mask = Image.open(self.mask[index]).convert('RGB')
		if self.transforms is not None:
			image = self.transforms(image)
			mask = self.transforms(mask)
		
		image = np.asarray(image).transpose(2,0,1)
		image_tensor = torch.from_numpy(image).float()
		mask = np.asarray(mask).transpose(2,0,1)
		mask_tensor = torch.from_numpy(mask).float()		

		return image_tensor, mask_tensor

	def __len__(self):
		return len(self.image_list)



if __name__ == '__main__':
	transforms = Compose([
				CenterCrop(135),
				Resize(128)
				])

	img_dir = '../../../data/celeba/images/'
	mask_dir = '../../../data/celeba/masks/eye_mouth_nose/'
	dataset = CelebA('test', img_dir, mask_dir, transforms = transforms)
	loader = data.DataLoader(dataset, batch_size=100)
	idx = 0
	for image, mask in loader:
		print(idx, image.shape, mask.shape)
		idx += 1


