import numpy as np
import pandas
import os
import webrtcvad
import soundfile

import scipy
from scipy import signal
from scipy.io import wavfile
from collections import namedtuple

import warnings
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

import gpuRIR
import math
from locata_utils import *
from AcousticScene import AcousticScene

import torch
import torchaudio
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import random
from debug import dbg_print

#  Acoustic Scene Datasets
class Parameter:
	""" Random parammeter class.
	You can indicate a constant value or a random range in its constructor and then
	get a value acording to that with getValue(). It works with both scalars and vectors.
	"""
	def __init__(self, *args):
		if len(args) == 1:
			self.random = False
			self.value = np.array(args[0])
			self.min_value = None
			self.max_value = None
		elif len(args) == 2:
			self.random = True
			self.min_value = np.array(args[0])
			self.max_value = np.array(args[1])
			self.value = None
		else: 
			raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
	
	def getValue(self):
		if self.random:
			return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
		else:
			return self.value
# Named tuple with the characteristics of a microphone array and definitions of the LOCATA arrays:
ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern')

dicit_array_setup = ArraySetup(arrayType='planar', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array((( 0.96, 0.00, 0.00),
						( 0.64, 0.00, 0.00),
						( 0.32, 0.00, 0.00),
						( 0.16, 0.00, 0.00),
						( 0.08, 0.00, 0.00),
						( 0.04, 0.00, 0.00),
						( 0.00, 0.00, 0.00),
						( 0.96, 0.00, 0.32),
						(-0.04, 0.00, 0.00),
						(-0.08, 0.00, 0.00),
						(-0.16, 0.00, 0.00),
						(-0.32, 0.00, 0.00),
						(-0.64, 0.00, 0.00),
						(-0.96, 0.00, 0.00),
						(-0.96, 0.00, 0.32))), 
	mic_orV = np.tile(np.array([[0.0, 1.0, 0.0]]), (15,1)),
	mic_pattern = 'omni'
)

dummy_array_setup = ArraySetup(arrayType='planar', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array(((-0.079,  0.000, 0.000),
						(-0.079, -0.009, 0.000),
						( 0.079,  0.000, 0.000),
						( 0.079, -0.009, 0.000))), 
	mic_orV = np.array(((-1.0, 0.0, 0.0),
						(-1.0, 0.0, 0.0),
						( 1.0, 0.0, 0.0),
						( 1.0, 0.0, 0.0))), 
	mic_pattern = 'omni'
)

benchmark2_array_setup = ArraySetup(arrayType='3D', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array(((-0.028,  0.030, -0.040),
						( 0.006,  0.057,  0.000),
						( 0.022,  0.022, -0.046),
						(-0.055, -0.024, -0.025),
						(-0.031,  0.023,  0.042),
						(-0.032,  0.011,  0.046),
						(-0.025, -0.003,  0.051),
						(-0.036, -0.027,  0.038),
						(-0.035, -0.043,  0.025),
						( 0.029, -0.048, -0.012),
						( 0.034, -0.030,  0.037),
						( 0.035,  0.025,  0.039))), 
	mic_orV = np.array(((-0.028,  0.030, -0.040),
						( 0.006,  0.057,  0.000),
						( 0.022,  0.022, -0.046),
						(-0.055, -0.024, -0.025),
						(-0.031,  0.023,  0.042),
						(-0.032,  0.011,  0.046),
						(-0.025, -0.003,  0.051),
						(-0.036, -0.027,  0.038),
						(-0.035, -0.043,  0.025),
						( 0.029, -0.048, -0.012),
						( 0.034, -0.030,  0.037),
						( 0.035,  0.025,  0.039))),
	mic_pattern = 'omni'
)

eigenmike_array_setup = ArraySetup(arrayType='3D', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array((( 0.000,  0.039,  0.015),
						(-0.022,  0.036,  0.000),
						( 0.000,  0.039, -0.015),
						( 0.022,  0.036,  0.000),
						( 0.000,  0.022,  0.036),
						(-0.024,  0.024,  0.024),
						(-0.039,  0.015,  0.000),
						(-0.024,  0.024,  0.024),
						( 0.000,  0.022, -0.036),
						( 0.024,  0.024, -0.024),
						( 0.039,  0.015,  0.000),
						( 0.024,  0.024,  0.024),
						(-0.015,  0.000,  0.039),
						(-0.036,  0.000,  0.022),
						(-0.036,  0.000, -0.022),
						(-0.015,  0.000, -0.039),
						( 0.000, -0.039,  0.015),
						( 0.022, -0.036,  0.000),
						( 0.000, -0.039, -0.015),
						(-0.022, -0.036,  0.000),
						( 0.000, -0.022,  0.036),
						( 0.024, -0.024,  0.024),
						( 0.039, -0.015,  0.000),
						( 0.024, -0.024, -0.024),
						( 0.000, -0.022, -0.036),
						(-0.024, -0.024, -0.024),
						(-0.039, -0.015,  0.000),
						(-0.024, -0.024,  0.024),
						( 0.015,  0.000,  0.039),
						( 0.036,  0.000,  0.022),
						( 0.036,  0.000, -0.022),
						( 0.015,  0.000, -0.039))), 
	mic_orV = np.array((( 0.000,  0.039,  0.015),
						(-0.022,  0.036,  0.000),
						( 0.000,  0.039, -0.015),
						( 0.022,  0.036,  0.000),
						( 0.000,  0.022,  0.036),
						(-0.024,  0.024,  0.024),
						(-0.039,  0.015,  0.000),
						(-0.024,  0.024,  0.024),
						( 0.000,  0.022, -0.036),
						( 0.024,  0.024, -0.024),
						( 0.039,  0.015,  0.000),
						( 0.024,  0.024,  0.024),
						(-0.015,  0.000,  0.039),
						(-0.036,  0.000,  0.022),
						(-0.036,  0.000, -0.022),
						(-0.015,  0.000, -0.039),
						( 0.000, -0.039,  0.015),
						( 0.022, -0.036,  0.000),
						( 0.000, -0.039, -0.015),
						(-0.022, -0.036,  0.000),
						( 0.000, -0.022,  0.036),
						( 0.024, -0.024,  0.024),
						( 0.039, -0.015,  0.000),
						( 0.024, -0.024, -0.024),
						( 0.000, -0.022, -0.036),
						(-0.024, -0.024, -0.024),
						(-0.039, -0.015,  0.000),
						(-0.024, -0.024,  0.024),
						( 0.015,  0.000,  0.039),
						( 0.036,  0.000,  0.022),
						( 0.036,  0.000, -0.022),
						( 0.015,  0.000, -0.039))),
	mic_pattern = 'omni'
)

miniDSP_array_setup = ArraySetup(arrayType='planar',
	orV = np.array([0.0, 0.0, 1.0]),
	mic_pos = np.array((( 0.0000,  0.0430, 0.000),
						( 0.0372,  0.0215, 0.000),
						( 0.0372, -0.0215, 0.000),
						( 0.0000, -0.0430, 0.000),
						(-0.0372, -0.0215, 0.000),
						(-0.0372,  0.0215, 0.000))),
	mic_orV = np.array(((0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0))),
	mic_pattern = 'omni'
)



class RandomTrajectoryDataset(Dataset):
	""" Dataset Acoustic Scenes with random trajectories.
	The length of the dataset is the length of the source signals dataset.
	When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
	"""
	def __init__(self, sourceDataset, noiseDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points, static_prob=0.25, non_linear_motion_prob=0.75, transforms=None):
		"""
		sourceDataset: dataset with the source signals (such as LibriSpeechDataset)
		room_sz: Size of the rooms in meters
		T60: Reverberation time of the room in seconds
		abs_weights: Absorption coefficients rations of the walls
		array_setup: Named tuple with the characteristics of the array
		array_pos: Position of the center of the array as a fraction of the room size
		SNR: Signal to (omnidirectional) Noise Ration
		nb_points: Number of points to simulate along the trajectory
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene

		Any parameter (except from nb_points and transforms) can be Parameter object to make it random.
		"""
		self.sourceDataset = sourceDataset
		self.noiseDataset = noiseDataset

		self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
		self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
		self.abs_weights = abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)

		assert np.count_nonzero(array_setup.orV) == 1, "array_setup.orV mus be parallel to an axis"
		self.array_setup = array_setup
		self.N = array_setup.mic_pos.shape[0]
		self.num_mic_pairs = int((self.N*(self.N -1))//2)
		self.mic_pairs = [(_ch1, _ch2) for _ch1 in range(self.N) for _ch2 in range(_ch1+1, self.N)]  #List of Tuples (mic1, mic2)
		self.array_pos = array_pos if type(array_pos) is Parameter else Parameter(array_pos)

		self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)
		self.nb_points = nb_points
		self.fs = sourceDataset.fs

		self.transforms = transforms

		self.static_prob = static_prob
		self.non_linear_motion_prob = non_linear_motion_prob

	def __len__(self):
		"include each microphone pair separately"
		return len(self.sourceDataset) #*self.num_mic_pairs

	def __getitem__(self, idx):
		#if idx < 0: idx = len(self) + idx

		#speech_file_idx = idx // len(self.sourceDataset)
		#mic_pair = self.mic_pairs[ (idx % self.num_mic_pairs)]

		speech_file_idx = idx
		acoustic_scene = self.getRandomScene(speech_file_idx)
		# GPU calls inside simulate
		mic_signals, dp_signals, noise_reverb = acoustic_scene.simulate()

		mic_signals = torch.from_numpy(mic_signals.T)
		dp_signals = torch.from_numpy(dp_signals.T)
		noise_reverb = torch.from_numpy(noise_reverb.T)

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, dp_signals, doa = t(mic_signals, dp_signals, acoustic_scene)
			return mic_signals, dp_signals, doa #, noise_reverb                 # noise_reverb is time domain signal: just for listening 

		return mic_signals, dp_signals, acoustic_scene.DOA    #, noise_reverb - for debug only

	def get_batch(self, idx1, idx2):
		mic_sig_batch = []
		acoustic_scene_batch = []
		for idx in range(idx1, idx2):
			mic_sig, acoustic_scene = self[idx]
			mic_sig_batch.append(mic_sig)
			acoustic_scene_batch.append(acoustic_scene)

		return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch)

	def getRandomScene(self, idx):
		# Source signal
		source_signal, vad = self.sourceDataset[idx]

		# Room
		room_sz = self.room_sz.getValue()
		T60 = self.T60.getValue()
		abs_weights = self.abs_weights.getValue()
		beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights)

		# Microphones
		array_pos = self.array_pos.getValue() * room_sz
		mic_pos = array_pos + self.array_setup.mic_pos

		# Trajectory points
		src_pos_min = np.array([0.0, 0.0, 0.0])
		src_pos_max = room_sz
		if self.array_setup.arrayType == 'planar':
			if np.sum(self.array_setup.orV) > 0:
				src_pos_min[np.nonzero(self.array_setup.orV)] = array_pos[np.nonzero(self.array_setup.orV)]
			else:
				src_pos_max[np.nonzero(self.array_setup.orV)] = array_pos[np.nonzero(self.array_setup.orV)]
		src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
		src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

		Amax = np.min(np.stack((src_pos_ini - src_pos_min,
									  src_pos_max - src_pos_ini,
									  src_pos_end - src_pos_min,
									  src_pos_max - src_pos_end)),
								axis=0)

		A = np.random.random(3) * np.minimum(Amax, 1) 			# Oscilations with 1m as maximum in each axis
		w = 2*np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis

		if random.random() < self.static_prob:                              # Static [0,1)
			traj_pts = np.ones((self.nb_points,1)) * src_pos_ini
			dbg_print("Static Source\n")
		else:
			traj_pts = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
			if random.random() < self.non_linear_motion_prob:                          # Non-Linear Motion 
				traj_pts += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
				dbg_print("Moving Source Non-Linear\n")
			else:
				dbg_print("Moving Source Linear\n")

		# Interpolate trajectory points
		timestamps = np.arange(self.nb_points) * len(source_signal) / self.fs / self.nb_points
		t = np.arange(len(source_signal))/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		# noise trajectory
		noise_idx = random.randint(0, len(self.noiseDataset)-1)
		noise_signal = self.noiseDataset[noise_idx]

		#torchaudio.save('raw_noise_signal.wav', torch.from_numpy(noise_signal).to(torch.float32).unsqueeze(dim=0), 16000)

		noise_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
		noise_pos = np.expand_dims(noise_pos, axis=0)

		noise_traj_pts = np.ones((self.nb_points,1)) * noise_pos

		acoustic_scene = AcousticScene(
			room_sz = room_sz,
			T60 = T60,
			beta = beta,
			SNR = self.SNR.getValue(),
			array_setup = self.array_setup,
			mic_pos = mic_pos,
			source_signal = source_signal,
			noise_signal = noise_signal,
			fs = self.fs,
			t = t,
			traj_pts = traj_pts,
			timestamps = timestamps,
			trajectory = trajectory,
			DOA = cart2sph(trajectory - array_pos),#[:,1:3],  #TODO
			noise_pos = noise_pos,
			noise_traj_pts = noise_traj_pts    
		)
		acoustic_scene.source_vad = vad

		return acoustic_scene



if __name__=="__main__":
	_libri_speech_path = '/scratch/bbje/battula12/Databases/LibriSpeech/LibriSpeech/train-clean-100'
	_demand_noise_path = '/scratch/bbje/battula12/Databases/DEMAND'
	T = 20

	from LibriSpeech import LibriSpeechDataset
	from Demand import DemandNoiseDataset

	array_setup = dicit_array_setup
	nb_points = 64
	room_sz = Parameter([3,3,2.5], [10,8,6]) 	# Random room sizes from 3x3x2.5 to 10x8x6 meters
	T60 = Parameter(0.2, 1.3)					# Random reverberation times from 0.2 to 1.3 seconds
	abs_weights = Parameter([0.5]*6, [1.0]*6)  # Random absorption weights ratios between walls
	SNR = Parameter(5, 30)
	array_pos = Parameter([0.4, 0.1, 0.1],[0.6, 0.9, 0.5]) # 

	#_val1, _val2 = array_pos.getValue(), room_sz.getValue()
	#array_pos = _val1 * _val2
	#mic_pos = array_pos + array_setup.mic_pos
	#print(mic_pos)
	#assert not ((mic_pos >= _val2).any() or (mic_pos <= 0).any())


	
	sourceDataset = LibriSpeechDataset(_libri_speech_path, T, return_vad=True)
	noiseDataset = DemandNoiseDataset(_demand_noise_path, T, 16000,_is_train=True)
	train_dataset = RandomTrajectoryDataset(sourceDataset, noiseDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points)#, transforms=[SRPDNN_features()]

	print(f"LibriSpeech: {len(sourceDataset)}, train_ds: {len(train_dataset)}")
	#mic_signals, dp_signals, acoustic_scene = train_dataset.__getitem__(3)
	#breakpoint()
	#print(f"inp_shape: {mic_signals.shape}, tgt_shape: {dp_signals.shape}")
	

	train_loader = DataLoader(train_dataset, batch_size = 1, num_workers=0)

	for _batch_idx, val in enumerate(train_loader):
		print(f"_batch_idx: {_batch_idx}, inp_shape: {val[0].shape}, inp_dtype: {val[0].dtype}, tgt_shape: {val[1].shape}, tgt_dtype: {val[1].dtype}, doA: {val[2].shape}, doA_dtype: {val[2].dtype}")

		if _batch_idx==0:
			break
