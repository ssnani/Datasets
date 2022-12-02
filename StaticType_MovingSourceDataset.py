import numpy as np
import os
import webrtcvad
import soundfile

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

import gpuRIR
import random
from collections import namedtuple
from utils import Parameter
from locata_utils import *
from AcousticScene import AcousticScene

from debug import dbg_print

ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern')
array_setup_10cm_2mic = ArraySetup(arrayType='planar', 
								orV = np.array([0.0, 1.0, 0.0]),
								mic_pos = np.array((( 0.05,  0.000, 0.000),
													(-0.05,  0.000, 0.000))), 
								mic_orV = np.array(((0.0, 1.0, 0.0),
													(0.0, 1.0, 0.0))), 
								mic_pattern = 'omni'
)

class SegSplitter:

	def __init__(self, seg_size, seg_shift):
		self.seg_kernel = (1,seg_size)
		self.seg_stride = (1,seg_shift)

	def __call__(self, x):
		#assuming x.shape ( 2,*)
		x = x.unsqueeze(dim=1).unsqueeze(dim=1) #reshape required for unfold
		x_segs = torch.nn.functional.unfold(x, self.seg_kernel, padding=(0,0), stride=self.seg_stride)
		x_segs = torch.permute(x_segs, [2,0,1])  #( n_segs, 2 , seg_size)
		x_segs = torch.reshape(x_segs,[x_segs.shape[0]*x_segs.shape[1], x_segs.shape[2]]) #( n_segs*2 , seg_size)
		return x_segs  


class NetworkInput_Seg(object):
	def __init__(self, frame_len, frame_shift, seg_size, seg_shift):
		self.frame_len = frame_len
		self.frame_shift = frame_shift
		self.kernel = (1,frame_len)
		self.stride = (1,frame_shift)

		self.seg_split = SegSplitter(seg_size, seg_shift)

	def __call__(self, mix, tgt, acoustic_scene):
		# Expecting shape mix, tgt: (2, *)
		# acoustic_scene has numpy objects doa (samples, 2)
		# Output shape: (2*2(r,i), T, f), (T,2)

		num_ch = mix.shape[0]
		mix_seg = self.seg_split(mix)
		tgt_seg = self.seg_split(tgt)

		
		dbg_print(f"input : mix: {mix.shape} tgt: {tgt.shape},  doa:{acoustic_scene.DOA.shape}")
		ip_real_img = torch.stft(mix_seg, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=False) #(n_seg*2, num_freq, T, 2(r,i))
		tgt_real_img = torch.stft(tgt_seg, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=False)

		ip_real_img = torch.permute(ip_real_img, [0, 3, 2, 1]) #(n_seg*2, 2(r,i), T, F)
		tgt_real_img = torch.permute(tgt_real_img, [0, 3, 2, 1]) #(n_seg*2, 2(r,i), T, F)

		(num_seg, _ri, frms, freq) = ip_real_img.shape
		ip_feat = torch.reshape(ip_real_img, (num_seg//num_ch, num_ch*_ri, frms, freq))
		tgt_feat = torch.reshape(tgt_real_img, (num_seg//num_ch, num_ch*_ri, frms, freq))


		# Code for frame_level doa
		doa = torch.from_numpy(acoustic_scene.DOA.T)
		doa_seg = self.seg_split(doa)
		doa_seg = doa_seg.unsqueeze(dim=1).unsqueeze(dim=1) #reshape required for unfold
		doa_frm = torch.nn.functional.unfold(doa_seg, self.kernel, padding=(0,0), stride=self.stride)

		doa_frm = torch.mode(doa_frm, dim=1).values

		doa_frm = doa_frm.reshape(doa_frm.shape[0]//2, 2, doa_frm.shape[1])
		doa_frm = torch.permute(doa_frm, [0,2,1])

		#float32 for pytorch lightning
		ip_feat, tgt_feat, doa_frm = ip_feat.to(torch.float32), tgt_feat.to(torch.float32), doa_frm.to(torch.float32)

		#MISO 
		tgt_feat = tgt_feat[:,:2]

		dbg_print(f"Transform inp: {ip_feat.shape} tgt: {tgt_feat.shape},  doa:{doa_frm.shape}")
		return ip_feat, tgt_feat, doa_frm  
 

class MovingSourceDataset(Dataset):

	# Currently supported for > 10 sec utterances

	def __init__(self, dataset_info_file, array_setup, transforms=None, size=None):
		
		with open(dataset_info_file, 'r') as f:
			self.tr_ex_list = [line.strip() for line in f.readlines()]

		self.fs = 16000
		self.resample = torchaudio.transforms.Resample(new_freq = self.fs)
		self.array_setup = array_setup
		self.transforms = [transforms] if transforms is not None else None
		self.size = size

	def __len__(self):
		return len(self.tr_ex_list) if self.size is None else self.size

	def __getitem__(self, idx):
		line_info = self.tr_ex_list[idx].split(',')

		#sph
		sph, fs = torchaudio.load(line_info[0])
		noi, fs_noi = torchaudio.load(line_info[1])
		cfg = torch.load(line_info[2])

		dbg_print(cfg)
		if self.fs != fs_noi:
			noi = torch.resample(noi)

		if self.fs != fs:
			sph = torch.resample(sph)

		#truncating to 10 sec utterances
		

		sph_len = sph.shape[1]
		noi_len = noi.shape[1]

		dbg_print(f'sph: {sph_len}, noi: {noi_len}')
		assert sph_len > 16000*4       #TODO: tempchange to 4 sec
		sph_len = 16000*4
		sph = sph[:,:sph_len]

		if sph_len < noi_len:
			"""
			start_idx = random.randint(0, noi_len - sph_len -1)
			noi = noi[:,start_idx:start_idx+sph_len]
			"""
			noi = noi[:,:sph_len]
		else:
			sph = sph[:,:noi_len]

		dbg_print(f'sph: {sph_len}, noi: {noi_len}')

		mic_pos = cfg['mic_pos']
		array_pos = np.mean(mic_pos, axis=0)
		beta = gpuRIR.beta_SabineEstimation(cfg['room_sz'], cfg['t60'], cfg['abs_weights'])

		"""
		self.nb_points = cfg['src_traj_pts'].shape[0]
		traj_pts = cfg['src_traj_pts']
		
		# Interpolate trajectory points
		timestamps = np.arange(self.nb_points) * sph_len / self.fs / self.nb_points
		t = np.arange(sph_len)/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		noise_pos = cfg['noise_pos']
		noise_pos = np.expand_dims(noise_pos, axis=0) if noise_pos.shape[0] != 1 else noise_pos

		noise_traj_pts = np.ones((self.nb_points,1)) * noise_pos if noise_pos.shape[0] == 1 else noise_pos
		"""
		
		src_traj_pts = cfg['src_traj_pts']
		noise_pos = cfg['noise_pos']
		noise_pos = np.expand_dims(noise_pos, axis=0) if len(noise_pos.shape) == 1 else noise_pos
		noise_traj_pts = noise_pos

		src_timestamps, t, src_trajectory = self.get_timestamps(src_traj_pts, sph_len)
		noise_timestamps, t, noise_trajectory = self.get_timestamps(noise_traj_pts, sph_len)

	
		dbg_print(f'src: {src_traj_pts}, noise: {noise_pos}\n')
		#breakpoint()

		acoustic_scene = AcousticScene(
			room_sz = cfg['room_sz'],
			T60 = cfg['t60'],
			beta = beta,
			SNR = cfg['snr'],
			array_setup = self.array_setup,
			mic_pos = cfg['mic_pos'],
			source_signal = sph[0],
			noise_signal = noi[0],
			fs = self.fs,
			t = t,
			traj_pts = cfg['src_traj_pts'],
			timestamps = src_timestamps,
			trajectory = src_trajectory,
			DOA = cart2sph(src_trajectory - array_pos), #[:,1:3], 
			noise_pos = noise_pos,
			noise_traj_pts = noise_traj_pts,
			noise_timestamps = noise_timestamps
		)

		
		mic_signals, dp_signals, noise_reverb = acoustic_scene.simulate_scenario()   #acoustic_scene.simulate()

		mic_signals = torch.from_numpy(mic_signals.T)
		dp_signals = torch.from_numpy(dp_signals.T)
		noise_reverb = torch.from_numpy(noise_reverb.T)


		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, dp_signals, doa = t(mic_signals, dp_signals, acoustic_scene)
			return mic_signals, dp_signals, doa #, noise_reverb                 # noise_reverb is time domain signal: just for listening 

		return mic_signals, dp_signals, acoustic_scene.DOA    #, noise_reverb - for debug only
		

	def get_timestamps(self, traj_pts, sig_len):
		nb_points = traj_pts.shape[0]
		
		# Interpolate trajectory points
		timestamps = np.arange(nb_points) * sig_len / self.fs / nb_points
		t = np.arange(sig_len)/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		return timestamps, t, trajectory

if __name__=="__main__":
	snr = 5
	t60 = 0.2
	dataset_file = f'dataset_file_circular_motion_snr_{snr}_t60_{t60}.txt' # 'dataset_file_10sec.txt'
	train_dataset = MovingSourceDataset(dataset_file, array_setup_10cm_2mic, size =1, transforms=None) #NetworkInput_Seg(320, 160, 64000, 16000))
	#breakpoint()
	train_loader = DataLoader(train_dataset, batch_size = 1, num_workers=0)
	for _batch_idx, val in enumerate(train_loader):
		print(f"mix_sig: {val[0].shape}, {val[0].dtype}, {val[0].device} \
		        tgt_sig: {val[1].shape}, {val[1].dtype}, {val[1].device} \
				doa: {val[2].shape}, {val[2].dtype}, {val[2].device} \n")

		
		break

