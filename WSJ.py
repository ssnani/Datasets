import numpy as np
import os
import webrtcvad
import soundfile

import torch
from torch.utils.data import Dataset, DataLoader
import random 

# Source signal Datasets

class WSJSpeechUttDataset(Dataset):
	""" Dataset with random LibriSpeech utterances.
	You need to indicate the path to the root of the LibriSpeech dataset in your file system
	and the length of the utterances in seconds.
	The dataset length is equal to the number of chapters in LibriSpeech (585 for train-clean-100 subset)
	but each time you ask for dataset[idx] you get a random segment from that chapter.
	It uses webrtcvad to clean the silences from the LibriSpeech utterances.
	"""

	def _exploreCorpus(self, path, file_extension, file_list):
		directory_tree = {}
		for item in os.listdir(path):
			if os.path.isdir( os.path.join(path, item) ):
				directory_tree[item] = self._exploreCorpus( os.path.join(path, item), file_extension, file_list )
			elif item.split(".")[-1] == file_extension:
				directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
				file_list.append(os.path.join(path, item))
		return directory_tree

	def _cleanSilences(self, s, aggressiveness, return_vad=False):
		self.vad.set_mode(aggressiveness)

		vad_out = np.zeros_like(s)
		vad_frame_len = int(10e-3 * self.fs)
		n_vad_frames = len(s) // vad_frame_len
		for frame_idx in range(n_vad_frames):
			frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
			frame_bytes = (frame * 32767).astype('int16').tobytes()
			vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, self.fs)
		s_clean = s * vad_out

		return (s_clean, vad_out) if return_vad else s_clean

	def __init__(self, path, T, size=None, return_vad=False, readers_range=None):
		self.file_list = []
		self.corpus  = self._exploreCorpus(path, 'wav', self.file_list)

		self.fs = 16000
		self.T = T
		self.req_samples = self.T * self.fs
		self.return_vad = return_vad
		self.vad = webrtcvad.Vad()

		self.sz = len(self.file_list) if size is None else size

	def __len__(self):
		return self.sz

	def __getitem__(self, idx):

		sph_utt_path = self.file_list[idx]
		utterance, fs = soundfile.read(sph_utt_path) 
		if 16000==fs:
			utt_len = utterance.shape[0]
			if utt_len > self.req_samples:
				start_idx = random.randint(0, utt_len - self.req_samples)
				utterance = utterance[start_idx:start_idx+ self.req_samples]
			else:
				#zero pad
				extra_samples = self.req_samples - utt_len
				utterance = np.concatenate([utterance, np.zeros(extra_samples)])
			
			utterance -= utterance.mean()

		# Clean silences, it starts with the highest aggressiveness of webrtcvad,
		# but it reduces it if it removes more than the 66% of the samples
		s_clean, vad_out = self._cleanSilences(utterance, 1, return_vad=True)
		"""
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(utterance, 2, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(utterance, 1, return_vad=True)
		"""
		return (s_clean, vad_out) if self.return_vad else s_clean



if __name__=="__main__":
	path = '/scratch/bbje/battula12/Databases/wsj/tr_05_clean/'
	T = 4
	dataset = WSJSpeechUttDataset(path, T, return_vad=True) # size=1
	print(f"Dataset Size: {len(dataset)}")
	#breakpoint()

	train_loader = DataLoader(dataset, batch_size = 10, num_workers=0)

	for _batch_idx, val in enumerate(train_loader):
		print(f"sig: {val[0].shape}, {val[0].dtype}, {val[0].device}")
		if _batch_idx==0:
			break
	


