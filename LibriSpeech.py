import numpy as np
import os
import webrtcvad
import soundfile

import torch
from torch.utils.data import Dataset, DataLoader


# Source signal Datasets

class LibriSpeechDataset(Dataset):
	""" Dataset with random LibriSpeech utterances.
	You need to indicate the path to the root of the LibriSpeech dataset in your file system
	and the length of the utterances in seconds.
	The dataset length is equal to the number of chapters in LibriSpeech (585 for train-clean-100 subset)
	but each time you ask for dataset[idx] you get a random segment from that chapter.
	It uses webrtcvad to clean the silences from the LibriSpeech utterances.
	"""

	def _exploreCorpus(self, path, file_extension):
		directory_tree = {}
		for item in os.listdir(path):
			if os.path.isdir( os.path.join(path, item) ):
				directory_tree[item] = self._exploreCorpus( os.path.join(path, item), file_extension )
			elif item.split(".")[-1] == file_extension:
				directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
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
		self.corpus = self._exploreCorpus(path, 'flac')
		if readers_range is not None:
			for key in list(map(int, self.nChapters.keys())):
				if int(key) < readers_range[0] or int(key) > readers_range[1]:
					del self.corpus[key]

		self.nReaders = len(self.corpus)
		self.nChapters = {reader: len(self.corpus[reader]) for reader in self.corpus.keys()}
		self.nUtterances = {reader: {
				chapter: len(self.corpus[reader][chapter]) for chapter in self.corpus[reader].keys()
			} for reader in self.corpus.keys()}

		self.chapterList = []
		for chapters in list(self.corpus.values()):
			self.chapterList += list(chapters.values())

		self.fs = 16000
		self.T = T

		self.return_vad = return_vad
		self.vad = webrtcvad.Vad()

		self.sz = len(self.chapterList) if size is None else size

	def __len__(self):
		return self.sz

	def __getitem__(self, idx):
		if idx < 0: idx = len(self) + idx
		while idx >= len(self.chapterList): idx -= len(self.chapterList)
		chapter = self.chapterList[idx]

		# Get a random speech segment from the selected chapter
		s = np.array([])
		utt_paths = list(chapter.values())
		n = np.random.randint(0,len(chapter))
		while s.shape[0] < self.T * self.fs:
			utterance, fs = soundfile.read(utt_paths[n]) 
			assert fs == self.fs
			s = np.concatenate([s, utterance])
			n += 1
			if n >= len(chapter): n=0
		s = s[0: self.T * fs]
		s -= s.mean()

		# Clean silences, it starts with the highest aggressiveness of webrtcvad,
		# but it reduces it if it removes more than the 66% of the samples
		s_clean, vad_out = self._cleanSilences(s, 3, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(s, 2, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(s, 1, return_vad=True)
			
		return (s_clean, vad_out) if self.return_vad else s_clean



if __name__=="__main__":
	path = '/scratch/bbje/battula12/Databases/LibriSpeech/LibriSpeech/train-clean-100'
	T = 4
	dataset = LibriSpeechDataset(path, T, return_vad=True)
	print(f"Dataset Size: {len(dataset)}")
	breakpoint()
	distributed_dataloading =False
	 
	if distributed_dataloading:
		from torch.utils.data.distributed import DistributedSampler
		import torch.distributed as dist

		torch.distributed.init_process_group('mpi')
		world_size = dist.get_world_size()
		rank = dist.get_rank()

		dist_sampler = DistributedSampler(dataset, world_size, rank)
		train_loader = DataLoader(dataset, batch_size = 6, num_workers=0, sampler=dist_sampler)
	else:
		train_loader = DataLoader(dataset, batch_size = 1, num_workers=0)

	for _batch_idx, val in enumerate(train_loader):
		print(f"sig: {val[0].shape}, {val[0].dtype}, {val[0].device}")
		if _batch_idx==0:
			break
	


