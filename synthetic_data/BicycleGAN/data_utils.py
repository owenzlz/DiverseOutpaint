# Import Packages
import torch
from torch.utils import data
from torchvision.transforms import functional as tvf
from torchvision.transforms import Compose, Resize, CenterCrop
import numpy as np
from PIL import Image
import pdb


class SyntheticDataset(data.Dataset):
    def __init__(self, action_dir, state_dir):
        self.actions = np.load(action_dir)
        self.states = np.load(state_dir)

    def __getitem__(self, index):
        action = torch.from_numpy(self.actions[index]).float()
        state = torch.from_numpy(self.states[index]).float()
        return action, state
    
    def __len__(self):
        return self.actions.shape[0]


if __name__ == '__main__':
    action_dir = '../data/input.npy'
    state_dir = '../data/output.npy'
    dataset = SyntheticDataset(action_dir, state_dir)
    loader = data.DataLoader(dataset, batch_size=10)
    for i, inputs in enumerate(loader):
        action, state = inputs
        print(i, action.shape, state.shape)



