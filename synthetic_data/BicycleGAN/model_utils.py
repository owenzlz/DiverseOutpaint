import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from spectral_normalization import SpectralNorm
import pdb

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class VEncoder(nn.Module):
    def __init__(self):
        super(VEncoder, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)     
        self.fc4 = nn.Linear(128, 256)
        self.mu, self.logvar = nn.Linear(256,2), nn.Linear(256,2)
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        return self.mu(h), self.logvar(h)
    
    def sample(self, mu, logvar, noise):
        std = torch.exp(0.5*logvar)
        return noise.mul(std).add_(mu)

    def kl_div(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2+2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)     
        self.fc4 = nn.Linear(128, 256)
        
        self.fc_action1 = nn.Linear(256,128)
        self.fc_action2 = nn.Linear(128,64)
        self.fc_action3 = nn.Linear(64,32)
        self.fc_action4 = nn.Linear(32,2)
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        
        action = F.relu(self.fc_action1(h))
        action = F.relu(self.fc_action2(action))
        action = F.relu(self.fc_action3(action))
        action = self.fc_action4(action)
        
        return action

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = SpectralNorm(nn.Linear(2, 32))
        self.fc2 = SpectralNorm(nn.Linear(32, 64))
        self.fc3 = SpectralNorm(nn.Linear(64, 128))
        self.fc4 = SpectralNorm(nn.Linear(128, 1))
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self, z):
        h = F.leaky_relu(self.fc1(z))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        out = self.fc4(h)        
        return out

if __name__ == '__main__':
    vencoder, decoder = VEncoder(), Decoder()
    
    x = torch.ones((1,2))
    
    mu, logvar = vencoder(x)
    dyna_noise = torch.randn(mu.size())
    dyna_code = vencoder.sample(mu, logvar, dyna_noise)

    y_hat = decoder(dyna_code)
    






