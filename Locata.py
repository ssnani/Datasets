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

import math

from locata_utils import *
from AcousticScene import AcousticScene

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader



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

class  LocataDataset(Dataset):
	""" Dataset with the LOCATA dataset recordings and its corresponding Acoustic Scenes.
	When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
	"""
	def __init__(self, path, array, fs, tasks=(1,3,5), recording=None, dev=False, transforms = None):
		"""
		path: path to the root of the LOCATA dataset in your file system
		array: string with the desired array ('dummy', 'eigenmike', 'benchmark2' or 'dicit')
		fs: sampling frequency (you can use it to downsample the LOCATA recordings)
		tasks: LOCATA tasks to include in the dataset (only one-source tasks are supported)
		recording: recordings that you want to include in the dataset (only supported if you selected only one task)
		dev: True if the groundtruth source positions are available
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		"""
		assert array in ('dummy', 'eigenmike', 'benchmark2', 'dicit'), 'Invalid array.'
		assert recording is None or len(tasks) == 1, 'Specific recordings can only be selected for dataset with only one task'
		for task in tasks: assert task in (1,3,5), 'Invalid task ' + str(task) + '.'

		self.path = path
		self.dev = dev
		self.array = array
		self.tasks = tasks
		self.transforms = transforms
		self.fs = fs

		self.vad = webrtcvad.Vad()
		self.vad.set_mode(1)

		if array == 'dummy':
			self.array_setup = dummy_array_setup
		elif array == 'eigenmike':
			self.array_setup = eigenmike_array_setup
		elif array == 'benchmark2':
			self.array_setup = benchmark2_array_setup
		elif array == 'dicit':
			self.array_setup = dicit_array_setup

		self.directories = []
		for task in tasks:
			task_path = os.path.join(path, 'task' + str(task))
			for recording in os.listdir( task_path ):
				arrays = os.listdir( os.path.join(task_path, recording) )
				if array in arrays:
					self.directories.append( os.path.join(task_path, recording, array) )
		self.directories.sort()

	def __len__(self):
		return len(self.directories)

	def __getitem__(self, idx):
		directory = self.directories[idx]
		mic_signals, fs = soundfile.read( os.path.join(directory, 'audio_array_' + self.array + '.wav') )
		if fs > self.fs:
			mic_signals = scipy.signal.decimate(mic_signals, int(fs/self.fs), axis=0)
			new_fs = fs / int(fs/self.fs)
			if new_fs != self.fs: warnings.warn('The actual fs is {}Hz'.format(new_fs))
			self.fs = new_fs
		elif fs < self.fs:
			raise Exception('The sampling rate of the file ({}Hz) was lower than self.fs ({}Hz'.format(fs, self.fs))

		# Remove initial silence
		start = np.argmax(mic_signals[:,0] > mic_signals[:,0].max()*0.15)
		mic_signals = mic_signals[start:,:]
		t = (np.arange(len(mic_signals)) + start)/self.fs

		df = pandas.read_csv( os.path.join(directory, 'position_array_' + self.array + '.txt'), sep='\t' )
		array_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
		array_ref_vec = np.stack((df['ref_vec_x'].values, df['ref_vec_y'].values,df['ref_vec_z'].values), axis=-1)
		array_rotation = np.zeros((array_pos.shape[0],3,3))
		for i in range(3):
			for j in range(3):
				array_rotation[:,i,j] = df['rotation_' + str(i+1) + str(j+1)]

		df = pandas.read_csv( os.path.join(directory, 'required_time.txt'), sep='\t' )
		required_time = df['hour'].values*3600+df['minute'].values*60+df['second'].values
		timestamps = required_time - required_time[0]

		if self.dev:
			sources_pos = []
			trajectories = []
			for file in os.listdir( directory ):
				if file.startswith('audio_source') and file.endswith('.wav'):
					source_signal, fs_src = soundfile.read(os.path.join(directory, file))
					if fs > self.fs:
						source_signal = scipy.signal.decimate(source_signal, int(fs_src / self.fs), axis=0)
					source_signal = source_signal[start:start+len(t)]
				if file.startswith('position_source'):
					df = pandas.read_csv( os.path.join(directory, file), sep='\t' )
					source_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
					sources_pos.append( source_pos )
					trajectories.append( np.array([np.interp(t, timestamps, source_pos[:,i]) for i in range(3)]).transpose() )
			sources_pos = np.stack(sources_pos)
			trajectories = np.stack(trajectories)

			DOA_pts = np.zeros(sources_pos.shape[0:2] + (2,))
			DOA = np.zeros(trajectories.shape[0:2] + (2,))
			for s in range(sources_pos.shape[0]):
				source_pos_local = np.matmul( np.expand_dims(sources_pos[s,...] - array_pos, axis=1), array_rotation ).squeeze() # np.matmul( array_rotation, np.expand_dims(sources_pos[s,...] - array_pos, axis=-1) ).squeeze()
				DOA_pts[s,...] = cart2sph(source_pos_local)[:,1:3]
				DOA[s,...] = np.array([np.interp(t, timestamps, DOA_pts[s,:,i]) for i in range(2)]).transpose()
			DOA[DOA[...,1]<-np.pi, 1] += 2*np.pi
		else:
			sources_pos = None
			DOA = None
			source_signal = np.NaN * np.ones((len(mic_signals),1))

		acoustic_scene = AcousticScene(
			room_sz = np.NaN * np.ones((3,1)),
			T60 = np.NaN,
			beta = np.NaN * np.ones((6,1)),
			SNR = np.NaN,
			array_setup = self.array_setup,
			mic_pos = np.matmul( array_rotation[0,...], np.expand_dims(self.array_setup.mic_pos, axis=-1) ).squeeze() + array_pos[0,:], # self.array_setup.mic_pos + array_pos[0,:], # Not valid for moving arrays
			source_signal = source_signal,
			fs = self.fs,
			t = t - start/self.fs,
			traj_pts = sources_pos[0,...],
			timestamps = timestamps - start/self.fs,
			trajectory = trajectories[0,...],
			DOA = DOA[0,...]
		)

		vad = np.zeros_like(source_signal)
		vad_frame_len = int(10e-3 * self.fs)
		n_vad_frames = len(source_signal) // vad_frame_len
		for frame_idx in range(n_vad_frames):
			frame = source_signal[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
			frame_bytes = (frame * 32767).astype('int16').tobytes()
			vad[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, int(self.fs))
		acoustic_scene.vad = vad

		#mic_signals.transpose()
		#print(f"{mic_signals.shape}")
		mic_signals = torch.from_numpy(mic_signals.T.copy())

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, direct_path_phdiff, acoustic_scene = t(mic_signals, acoustic_scene)

			return mic_signals, direct_path_phdiff, acoustic_scene.doA_reduced_frms

		return mic_signals, acoustic_scene.DOA

	def get_batch(self, idx1, idx2):
		mic_sig_batch = []
		acoustic_scene_batch = []
		for idx in range(idx1, idx2):
			mic_sig, acoustic_scene = self[idx]
			mic_sig_batch.append(mic_sig)
			acoustic_scene_batch.append(acoustic_scene)

		return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch)


if __name__=="__main__":
    _locata_speech_path = '/scratch/bbje/battula12/Databases/Locata/eval/'
    _locata_speech_dev_path = '/scratch/bbje/battula12/Databases/Locata/dev/'

    array_type = 'dicit'
    tasks = [3] #Task: 1: Static Single Source, 3: Moving Single Source

    train_eval_dataset = LocataDataset(_locata_speech_path, array_type, 16000, tasks, dev=True) #, transforms=]SRPDNN_Test_features(1,2)])
    train_dev_dataset = LocataDataset(_locata_speech_dev_path, array_type, 16000, tasks, dev=True)

    train_dataset = ConcatDataset([train_eval_dataset, train_dev_dataset])

    train_loader = DataLoader(train_dataset, batch_size = 1)
    for _batch_idx, val in enumerate(train_loader):
        print(f"inp_shape: {val[0].shape}, inp_dtype: {val[0].dtype}, tgt_shape: {val[1].shape}, tgt_dtype: {val[1].dtype}") #, doA: {val[2].shape}, doA_dtype: {val[2].dtype}")
        break