import torch
import torchaudio
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from LibriSpeech_utt import LibriSpeechUttDataset
from Demand import DemandNoiseDataset
from TestDataset import TestConfig
import random

class TestTrajectoryDataset(Dataset):
	""" Dataset Acoustic Scenes with random trajectories.
	The length of the dataset is the length of the source signals dataset.
	When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
	"""
	def __init__(self, array_setup, SNR, T60, nb_points, static_prob=0.25, src_mic_dist = 1.0, non_linear_motion_prob=0.75, same_plane=True, transforms=None, size = None ):

		self.test_path = '/scratch/bbje/battula12/Databases/LibriSpeech/LibriSpeech/test-clean'
		self.noise_path = '/scratch/bbje/battula12/Databases/DEMAND'

		self.fs = 16000

		self.speech_path = self.test_path
		self.is_train = False
		self.T = 4
		self.size = size
		self.sourceDataset = LibriSpeechUttDataset(self.speech_path, self.T, size=self.size, return_vad=True)
		self.noiseDataset =  DemandNoiseDataset(self.noise_path, self.T, self.fs, self.is_train)

		self.test_config = TestConfig(array_setup, test_snr=SNR, test_t60=T60, static_prob = static_prob, src_mic_dist=src_mic_dist, non_linear_prob=non_linear_motion_prob, nb_points=nb_points, same_plane=same_plane, noi_mic_dist=None)

		self.transforms = [transforms]

		self.static_prob = static_prob
		self.non_linear_motion_prob = non_linear_motion_prob

	def __len__(self):
		"include each microphone pair separately"
		return len(self.sourceDataset) if self.size is None else self.size 


	def __get_item__(self, idx):

		source_signal, vad = self.sourceDataset[idx]
		# noise trajectory
		noise_idx = random.randint(0, len(self.noiseDataset)-1)
		noise_signal = self.noiseDataset[noise_idx]

		scene_dict = self.test_config._create_static_acoustic_scene_config()
		
		mic_signals, dp_signals, acoustic_scene = self.test_config.simulate_acoustic_scene(scene_dict, source_signal, noise_signal)

		mic_signals = torch.from_numpy(mic_signals.T)
		dp_signals = torch.from_numpy(dp_signals.T)
		#noise_reverb = torch.from_numpy(noise_reverb.T)

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, dp_signals, doa = t(mic_signals, dp_signals, acoustic_scene)
			return mic_signals, dp_signals, doa #, noise_reverb                 # noise_reverb is time domain signal: just for listening 

		return mic_signals, dp_signals, acoustic_scene.DOA    #, noise_reverb - for debug only


