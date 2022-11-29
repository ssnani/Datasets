from LibriSpeech import LibriSpeechDataset
from RandomTrajectory import RandomTrajectoryDataset, Parameter, dicit_array_setup
import os
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


def init_process(fun): #rank, world_size, fun

	""" Initialize the distributed environment. """
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'
	torch.distributed.init_process_group('nccl') #, rank=rank, world_size=world_size)

    world_size = dist.get_world_size()
	rank = dist.get_rank()
	print(f"# world_size's: {world_size}, rank: {rank}")

	
	fun(rank, world_size)


def get_dataset():
    path = '/scratch/bbje/battula12/Databases/LibriSpeech/LibriSpeech/train-clean-100'
	T = 20
	array_setup = dicit_array_setup
	nb_points = 64
	room_sz = Parameter([3,3,2.5], [10,8,6]) 	# Random room sizes from 3x3x2.5 to 10x8x6 meters
	T60 = Parameter(0.2, 1.3)					# Random reverberation times from 0.2 to 1.3 seconds
	abs_weights = Parameter([0.5]*6, [1.0]*6)  # Random absorption weights ratios between walls
	SNR = Parameter(5, 30)
	array_pos = Parameter([0.4, 0.1, 0.1],[0.6, 0.9, 0.5]) # 
	sourceDataset = LibriSpeechDataset(path, T, return_vad=True)
	train_dataset = RandomTrajectoryDataset(sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points)

    return train_dataset



def dist_data_loader(rank, world_size):

    train_dataset = get_dataset()
	dist_sampler = DistributedSampler(train_dataset, world_size, rank)
	print(f"Distributed Dataset Size: {len(dist_sampler)}")
	train_loader = DataLoader(train_dataset, batch_size = 6, num_workers=0, sampler=dist_sampler)

	for _batch_idx, val in enumerate(train_loader):
		print(f"{os.uname()[1]}, sig: {val[0].shape}, {val[0].dtype}, {val[0].device}")
		if _batch_idx==5:
			break


if __name__ == "__main__":



