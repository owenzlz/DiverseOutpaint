import warnings
warnings.filterwarnings("ignore")
import argparse, os
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.metrics import mean_squared_error
from skimage.io import imsave
import pdb

# compute net action given push action
def compute_six_star_action(a_push):
    a_push_x = a_push[0]; a_push_y = a_push[1]
    quadrant = 0
    if a_push_x>0 and a_push_y>0: # first
        theta = np.arctan(a_push_y/a_push_x)
        quadrant = 1
    elif a_push_x<0 and a_push_y>0: # second
        theta = np.arctan(a_push_y/a_push_x) + np.pi
        quadrant = 2
    elif a_push_x<0 and a_push_y<0: # third
        theta = np.arctan(a_push_y/a_push_x) + np.pi
        quadrant = 3
    elif a_push_x>0 and a_push_y<0: # fourth
        theta = np.arctan(a_push_y/a_push_x) + 2*np.pi
        quadrant = 4
    mag = np.sqrt(a_push_x**2 + a_push_y**2)
    a_fric = abs(np.cos(theta*3)*mag*0.6)
    if quadrant == 1:
        a_fric_x = -a_fric*np.cos(theta)
        a_fric_y = -a_fric*np.sin(theta)
    elif quadrant == 2:
        a_fric_x = a_fric*np.cos(np.pi - theta)
        a_fric_y = -a_fric*np.sin(np.pi - theta)        
    elif quadrant == 3:
        a_fric_x = a_fric*np.cos(theta - np.pi)
        a_fric_y = a_fric*np.sin(theta - np.pi)    
    elif quadrant == 4:
        a_fric_x = -a_fric*np.cos(2*np.pi - theta)
        a_fric_y = a_fric*np.sin(2*np.pi - theta)            
    a_total = np.zeros((2))
    a_total[0] = a_push_x + a_fric_x if abs(a_push_x) > abs(a_fric_x) else 0
    a_total[1] = a_push_y + a_fric_y if abs(a_push_y) > abs(a_fric_y) else 0
    return a_total, [a_fric_x, a_fric_y]

# noise sampling 
def noise_sampling(batch_size, num_sample, noise_dim):
	noise = torch.FloatTensor(batch_size, num_sample, noise_dim).uniform_()
	return noise

# evaluate and visualize the unfold action and state space
def eval_model(actions_dir, vencoder, decoder, step):
    n = 50
    clen = 150
    actions = np.load(actions_dir)
    # draw boundary of training space
    canvas = np.ones((clen,clen,3))*255
    for i in range(actions.shape[0]):
        # draw action 
        action_vec = (int(actions[i,0]+int(clen/2)), int(actions[i,1]+int(clen/2)))
        canvas[action_vec[0], action_vec[1]] = (0,255,255)
    
    # draw interpolated spaces over train space
    n_list = [50, 300, 1000, 10000]
    demo_canvas_with_interpolate = torch.zeros((5,3,clen,clen))
    demo_canvas_with_interpolate[0] = torch.from_numpy(canvas).transpose(0,2).transpose(1,2)
    for k in range(len(n_list)):
        n = n_list[k]
        canvas_with_interpolate = canvas
        for i in range(n):
            # compute sampled action, state, and groud truth statet
            
            dyna_noise = torch.randn((1,2))
            action_hat = decoder(dyna_noise)  
            action_hat = action_hat.data.numpy()
            
            # draw action
            action_vec = (int(action_hat[0,0]+int(clen/2)), int(action_hat[0,1]+int(clen/2)))
            canvas[action_vec[0], action_vec[1]] = (0,255,0)
            
        demo_canvas_with_interpolate[k+1] = torch.from_numpy(canvas_with_interpolate).transpose(0,2).transpose(1,2)
    save_image(demo_canvas_with_interpolate, 'results/input_space_with_interpolate_'+str(step)+'.png', nrow=5, padding=2, pad_value=0)
    
    

if __name__ == '__main__':
    decoder = torch.load('models/decoder_9000.pt')





