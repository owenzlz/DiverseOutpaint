import torch

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

def uncollapse_batch(batch, num_sample):
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

def diverse_sampling(code, num_sample, noise_dim, gpu_id):
	N, C = code.size(0), code.size(1)
	noise = torch.FloatTensor(N, num_sample, noise_dim).uniform_().to(gpu_id)
	code = (code[:,None,:]).expand(-1,num_sample,-1)
	code = torch.cat([code, noise], dim=2)
	return code, noise

def center_sampling(code, num_sample, noise_dim, gpu_id):
	N, C = code.size(0), code.size(1)
	noise = (torch.ones(N, num_sample, noise_dim)*0.5).to(gpu_id)
	code = (code[:,None,:]).expand(-1,num_sample,-1)
	code = torch.cat([code, noise], dim=2)
	return code, noise





