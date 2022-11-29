import numpy as np
from collections import namedtuple
import torch
import torchaudio
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import gpuRIR

# Acoustic Scene Datasets
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

two_mic_array_setup = ArraySetup(arrayType='planar', 
	orV = np.array([0.0, 0.0, 1.0]),
	mic_pos = np.array((
						( 0.05, 0.00, 0.00),
						(-0.05, 0.00, 0.00),
						)), 
	mic_orV = np.tile(np.array([[0.0, 0.0, 1.0]]), (2,1)),
	mic_pattern = 'omni'
)

class StaticAcousticScene:
	""" Acoustic scene class.
	It contains everything needed to simulate a moving sound source moving recorded
	with a microphone array in a reverberant room.
	It can also store the results from the DOA estimation.
	"""
	def __init__(self, room_sz, T60, beta, SNR, array_setup, mic_pos, source_signal, noise_signal, fs, traj_pts, timestamps,
				 trajectory, t, DOA, noise_pos, noise_traj_pts):
		self.room_sz = room_sz				# Room size
		self.T60 = T60						# Reverberation time of the simulated room
		self.beta = beta					# Reflection coefficients of the walls of the room (make sure it corresponds with T60)
		self.SNR = SNR						# Signal to (omnidirectional) Noise Ration to simulate
		self.array_setup = array_setup		# Named tuple with the characteristics of the array
		self.mic_pos = mic_pos				# Position of the center of the array
		self.source_signal = source_signal  # Source signal
		self.noise_signal = noise_signal    # Noise signal
		self.fs = fs						# Sampling frequency of the source signal and the simulations
		self.traj_pts = traj_pts 			# Trajectory points to simulate
		self.timestamps = timestamps		# Time of each simulation (it does not need to correspond with the DOA estimations)
		self.trajectory = trajectory		# Continuous trajectory
		self.t = t							# Continuous time
		self.DOA = DOA 						# Continuous DOA
		self.noise_pos = noise_pos          # Currently Static noise
		self.noise_traj_pts = noise_traj_pts
		self._point_source_noise = True
		self.need_direct_path_signals = True

	def simulate(self):
		""" Get the array recording using gpuRIR to perform the acoustic simulations.
		"""
		msg = f"SNR: {self.SNR}, t60: {self.T60}"
		dbg_print(msg)
		if self.T60 == 0:
			Tdiff = 0.1
			Tmax = 0.1
			nb_img = [1,1,1]
		else:
			Tdiff = gpuRIR.att2t_SabineEstimator(12, self.T60) # Use ISM until the RIRs decay 12dB
			Tmax = gpuRIR.att2t_SabineEstimator(40, self.T60)  # Use diffuse model until the RIRs decay 40dB
			if self.T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
			nb_img = gpuRIR.t2n( Tdiff, self.room_sz )

		nb_mics  = len(self.mic_pos)
		nb_traj_pts = len(self.traj_pts)
		nb_gpu_calls = min(int(np.ceil( self.fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9 )), nb_traj_pts)
		traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls+1) ).astype(int)

		RIRs_list = [ gpuRIR.simulateRIR(self.room_sz, self.beta,
						 self.traj_pts[traj_pts_batch[0]:traj_pts_batch[1],:], self.mic_pos,
						 nb_img, Tmax, self.fs, Tdiff=Tdiff,
						 orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern) ]
		for i in range(1,nb_gpu_calls):
			RIRs_list += [	gpuRIR.simulateRIR(self.room_sz, self.beta,
						 	self.traj_pts[traj_pts_batch[i]:traj_pts_batch[i+1],:], self.mic_pos,
						 	nb_img, Tmax, self.fs, Tdiff=Tdiff,
						 	orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern) ]
		RIRs = np.concatenate(RIRs_list, axis=0)
		mic_signals = gpuRIR.simulateTrajectory(self.source_signal, RIRs, timestamps=self.timestamps, fs=self.fs)
		mic_signals = mic_signals[0:len(self.t),:]


		dp_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.traj_pts, self.mic_pos, [1,1,1], 0.1, self.fs,
									orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
		dp_signals = gpuRIR.simulateTrajectory(self.source_signal, dp_RIRs, timestamps=self.timestamps, fs=self.fs)
		dp_signals = dp_signals[:mic_signals.shape[0],:]

		#torchaudio.save("speech.wav", torch.from_numpy(self.source_signal).unsqueeze(dim=0).to(torch.float32), 16000)
		#torchaudio.save("speech_reverb.wav", torch.from_numpy(mic_signals).T.to(torch.float32), 16000)
		#torchaudio.save("speech_directpath.wav", torch.from_numpy(dp_signals).T.to(torch.float32), 16000)

		if 0: #default
			# Omnidirectional noise
			ac_pow = np.mean([acoustic_power(dp_signals[:,i]) for i in range(dp_signals.shape[1])])
			noise = np.sqrt(ac_pow/10**(self.SNR/10)) * np.random.standard_normal(mic_signals.shape)
			mic_signals += noise
		else:
			if self._point_source_noise:

				noise_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.noise_pos, self.mic_pos, nb_img, Tmax, self.fs, Tdiff=Tdiff, orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
			else:  #diffuse noise
				pass

			noise_reverb = gpuRIR.simulateTrajectory(self.noise_signal, noise_RIRs) 

			
			# handling shape and snr : channel 0
			noise_reverb = noise_reverb[0:len(self.t),:]

			scale_noi = np.sqrt(np.sum(mic_signals[:,0]**2) / (np.sum(noise_reverb[:,0]**2) * (10**(self.SNR/10))))
			mic_signals = mic_signals + noise_reverb * scale_noi

			if 1:
				# normalize the root mean square of the mixture to a constant
				sph_len = mic_signals.shape[0]#*mic_signals.shape[1]          #All mics
				c = 1.0 * np.sqrt(sph_len / np.sum(mic_signals[:,0]**2))
				mic_signals *= c
				dp_signals *= c


		# Apply the propagation delay to the VAD information if it exists
		if hasattr(self, 'source_vad'):
			vad = gpuRIR.simulateTrajectory(self.source_vad, dp_RIRs, timestamps=self.timestamps, fs=self.fs)
			self.vad = vad[0:len(self.t),:].mean(axis=1) > vad[0:len(self.t),:].max()*1e-3

		#torchaudio.save("scaled_noisy_reverb.wav", torch.from_numpy(noise_reverb * scale_noi).T.to(torch.float32), 16000)
		return mic_signals, dp_signals, noise_reverb

class StaticSourceDataset(Dataset):
	""" Dataset Acoustic Scenes with random trajectories.
	The length of the dataset is the length of the source signals dataset.
	When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
	"""
	def __init__(self, sourceDataset, noiseDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points, src_mic_dist,transforms=None):
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

		self.src_mic_dist = src_mic_dist if type(src_mic_dist) is Parameter else Parameter(src_mic_dist)
		self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)
		self.nb_points = nb_points
		self.fs = sourceDataset.fs

		self.transforms = transforms
		
	def __len__(self):
		"include each microphone pair separately"
		return len(self.sourceDataset) #*self.num_mic_pairs

	def __getitem__(self, idx):

		acoustic_scene = self.getRandomScene(idx)
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


		#arrar rotation


		src_mic_dist = self.src_mic_dist.getValue()

		rel_src_pos = [src_mic_dist*np.cos(src_theta), src_mic_dist*np.sin(src_theta), 0.0]

		rel_noise_pos = [src_mic_dist*np.cos(noise_theta), src_mic_dist*np.sin(noise_theta), 0.0]

		src_pos = array_pos + rel_src_pos
		noise_pos = array_pos + rel_noise_pos






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
			DOA = cart2sph(trajectory - array_pos)[:,1:3], 
			noise_pos = noise_pos,
			noise_traj_pts = noise_traj_pts    
		)
		acoustic_scene.source_vad = vad

		return acoustic_scene
