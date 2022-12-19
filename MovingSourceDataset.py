import numpy as np
import os
import webrtcvad
import soundfile

import torch
from torch.utils.data import Dataset, DataLoader

#import gpuRIR
import random
from collections import namedtuple
from utils import Parameter

_DEBUG = False
def dbg_print(msg):
	if _DEBUG:
		print(msg)
	return 

class MovingSourceConfigDataset(Dataset):

	def _exploreCorpus(self, path, file_extension, file_list):
		for item in os.listdir(path):
			if os.path.isdir( os.path.join(path, item) ):
				self._exploreCorpus( os.path.join(path, item), file_extension, file_list )
			elif item.split(".")[-1] == file_extension:
				file_list.append(os.path.join(path, item))
		return 

	def _exploreNoiseCorpus(self, path, _is_train):
		# Noise Folder structure
		tr_noises = ['DKITCHEN', 'DWASHING', 'NPARK', 'OHALLWAY', 'OOFFICE', 'PRESTO', 'TCAR', 'DLIVING', 'NFIELD', 'NRIVER', 'OMEETING', 'PCAFETER', 'PSTATION', 'TBUS', 'TMETRO']
		tt_noises = ['STRAFFIC', 'SPSQUARE', 'SCAFE']

		noise_folder_list = tr_noises if _is_train else tt_noises

		noise_file_list = []
		for folder in noise_folder_list:
			_file = os.path.join(path, folder,'ch01.wav')
			noise_file_list.append(_file)

		return noise_file_list

	def __len__(self):
		return len(self.sph_file_list)

	def _create_random_acoustic_scene(self):

		# Room
		room_sz = self.room_size.getValue()
		T60 = self.t60.getValue()
		abs_weights = self.abs_weights.getValue()
		# Microphones
		array_pos = self.array_pos.getValue() * room_sz
		mic_pos = array_pos + self.array_setup.mic_pos

		#source_trajectory

		# Trajectory points
		src_pos_min = np.array([0.0, 0.0, 0.0])
		src_pos_max = np.array(room_sz)

		if self.array_setup.arrayType == 'planar':
			if np.sum(self.array_setup.orV) > 0:
				src_pos_min[np.nonzero(self.array_setup.orV)] = array_pos[np.nonzero(self.array_setup.orV)]
			else:
				src_pos_max[np.nonzero(self.array_setup.orV)] = array_pos[np.nonzero(self.array_setup.orV)]


		if self._same_plane:
			#mic : XZ plane
			#src moven : XY plane
			src_pos_min[2] = array_pos[2]
			src_pos_max[2] = array_pos[2]

		# Linear Motion
		src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
		src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

		Amax = np.min(np.stack((src_pos_ini - src_pos_min,
									  src_pos_max - src_pos_ini,
									  src_pos_end - src_pos_min,
									  src_pos_max - src_pos_end)),
								axis=0)

		A = np.random.random(3) * np.minimum(Amax, 1) 			# Oscilations with 1m as maximum in each axis
		w = 2*np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis

		if self._same_plane:
			A[2] = 0


		if random.random() <= self.static_prob:                              # Static
			traj_pts = np.ones((self.nb_points,1)) * src_pos_ini
			dbg_print("Static Source\n")
		else:
			traj_pts = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
			if random.random() <= self.non_linear_motion_prob:                          # Non-Linear Motion 
				traj_pts += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
				dbg_print("Moving Source Non-Linear\n")
			else:
				dbg_print("Moving Source Linear\n")

		noise_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

		snr = self.snr.getValue()

		scene_dict = {}

		scene_dict['snr'] = snr
		scene_dict['t60'] = T60
		scene_dict['room_sz'] = room_sz
		scene_dict['abs_weights'] = abs_weights
		scene_dict['mic_pos'] = mic_pos
		scene_dict['src_traj_pts'] = traj_pts
		scene_dict['noise_pos'] = noise_pos

		return scene_dict
		


	def __init__(self, sph_dataset_path, sph_file_ext, noise_dataset_path, _is_train, array_setup):
		
		self.sph_file_list = []
		self._exploreCorpus(sph_dataset_path, sph_file_ext, self.sph_file_list)

		self.noise_file_list = self._exploreNoiseCorpus(noise_dataset_path, _is_train)

		#Creating Static Dataset 
		self.snr = Parameter(0,30)
		self.t60 = Parameter(0.2, 1.3)
		self.room_size = Parameter([3, 3, 2.5],[10, 8, 6])
		self.abs_weights = Parameter([0.5]*6, [1.0]*6)  # Random absorption weights ratios between walls
		self.array_setup = array_setup
		self.array_pos = Parameter([0.4, 0.1, 0.1],[0.6, 0.9, 0.5])
		self.nb_points = 16

		self.static_prob = 0.25
		self.non_linear_motion_prob = 0.50

		self._same_plane = True

		# Generating Config
		#config_dict = self._create_random_acoustic_scene()



if __name__=="__main__":
	libri_path = '/scratch/bbje/battula12/Databases/LibriSpeech/LibriSpeech/test-clean' #train-clean-100' dev-clean
	noise_path = "/scratch/bbje/battula12/Databases/DEMAND"
	ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern')
	array_setup_10cm_2mic = ArraySetup(arrayType='planar', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array((( 0.05,  0.000, 0.000),
						(-0.05,  0.000, 0.000))), 
	mic_orV = np.array(((0.0, 1.0, 0.0),
						(0.0, 1.0, 0.0))), 
	mic_pattern = 'omni'
)

	dataset = MovingSourceConfigDataset(libri_path, 'flac', noise_path, False, array_setup_10cm_2mic)
	"""
	for idx in range(len(dataset)):
		print(f"processing {idx}")
		config_dict = dataset._create_random_acoustic_scene()
		
		room_sz = config_dict['room_sz']
		pos_src = config_dict['src_traj_pts']
		mic_pos = config_dict['mic_pos']
		try:
			
			assert not ((pos_src >= room_sz).any() or (pos_src <= 0).any())
		except:
			print(mic_pos)
			print(room_sz)
			print(pos_src)
			break

		torch.save(config_dict, f'/scratch/bbje/battula12/Databases/SourceTrajectories/config_dict_{idx}.pt')
	#breakpoint()
	"""
	
	with open('test_dataset_file.txt', 'w') as f:
		for idx in range(len(dataset)):
			print(f"processing {idx}")
			#config_dict = dataset._create_random_acoustic_scene()
			#torch.save(config_dict, f'/scratch/bbje/battula12/Databases/SourceTrajectories/config_dict_{idx}.pt')
			noise_idx = random.randint(0, len(dataset.noise_file_list)-1)
			msg = f"{dataset.sph_file_list[idx]}, {dataset.noise_file_list[noise_idx]}, {f'/scratch/bbje/battula12/Databases/SourceTrajectories/config_circular_motion_{idx}.pt'}\n"
			f.write(msg)
			#break

	
		



	

		




		


