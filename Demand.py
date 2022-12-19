import numpy as np
import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class DemandNoiseDataset(Dataset):
	def __init__(self, path, fs, T, _is_train):
		self.path = path
		# Noise Folder structure
		self.tr_noises = ['DKITCHEN', 'DWASHING', 'NPARK', 'OHALLWAY', 'OOFFICE', 'PRESTO', 'TCAR', 'DLIVING', 'NFIELD', 'NRIVER', 'OMEETING', 'PCAFETER', 'PSTATION', 'TBUS', 'TMETRO']
		self.tt_noises = ['STRAFFIC', 'SPSQUARE', 'SCAFE']

		self.noise_folder_list = self.tr_noises if _is_train else self.tt_noises

		self.noise_list = []
		for folder in self.noise_folder_list:
			_file = os.path.join(path, folder,'ch01.wav')
			self.noise_list.append(_file)

		self.fs = fs
		self.T = T

		self.req_samples = self.T*self.fs
		self.transform = torchaudio.transforms.Resample(new_freq = fs, lowpass_filter_width=64,
    												rolloff=0.9475937167399596,
    												resampling_method="kaiser_window",
    												beta=14.769656459379492)

	def __len__(self):
		return len(self.noise_list)

	def __getitem__(self, idx):

		noise, fs = torchaudio.load(self.noise_list[idx])
		#Currently supported for 16k
		if 16000!=fs:
			
			self.transform.orig_freq = fs
			#noise = self.transform(noise)

			noise = torchaudio.functional.resample(noise,fs, self.fs)

		noise_len = noise.shape[1]
		if  noise_len < self.req_samples:
			req_repetitions = np.int(np.ceil(self.req_samples/noise_len))
			noise = noise.repeat(1, req_repetitions)

			noise = noise[:,:self.req_samples]
		else:
			start_idx = random.randint(0, noise_len-self.req_samples-1)
			noise = noise[:,start_idx:start_idx+self.req_samples]

		return noise.numpy()[0]


if __name__=="__main__":
	path = "/scratch/bbje/battula12/Databases/DEMAND"
	dataset = DemandNoiseDataset(path, T=20, fs=16000, _is_train=False)
	data_loader = DataLoader(dataset, batch_size = 1, num_workers=0)

	for _batch_idx, val in enumerate(data_loader):
		print(f"noise: {val[0].shape}, {val[0].dtype}, {val[0].device}")
		#torchaudio.save('noise.wav', val[0], 16000)
		#breakpoint()
		#if _batch_idx==0:
			#break
		






