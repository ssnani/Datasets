import numpy as np
import os
import webrtcvad


from collections import namedtuple

import warnings
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

import gpuRIR
import math

from locata_utils import *
from debug import dbg_print
import torchaudio

class AcousticScene:
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

				c = 1.0 * np.sqrt(sph_len / (np.sum(mic_signals[:,0]**2) + 1e-8))

				mic_signals *= c
				dp_signals *= c


		# Apply the propagation delay to the VAD information if it exists
		if hasattr(self, 'source_vad'):
			vad = gpuRIR.simulateTrajectory(self.source_vad, dp_RIRs, timestamps=self.timestamps, fs=self.fs)
			self.vad = vad[0:len(self.t),:].mean(axis=1) > vad[0:len(self.t),:].max()*1e-3

		#torchaudio.save("scaled_noisy_reverb.wav", torch.from_numpy(noise_reverb * scale_noi).T.to(torch.float32), 16000)
		return mic_signals, dp_signals, noise_reverb

	

	def static_simulate(self):
		# Speech Enhancement
		if self.T60 == 0:
			Tdiff = 0.1
			Tmax = 0.1
			nb_img = [1,1,1]
		else:
			Tdiff = gpuRIR.att2t_SabineEstimator(12, self.T60) # Use ISM until the RIRs decay 12dB
			Tmax = gpuRIR.att2t_SabineEstimator(40, self.T60)  # Use diffuse model until the RIRs decay 40dB
			if self.T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
			nb_img = gpuRIR.t2n( Tdiff, self.room_sz )

		noise_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.noise_pos, self.mic_pos, nb_img, Tmax, self.fs, Tdiff=Tdiff, orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
		noise_reverb = gpuRIR.simulateTrajectory(self.noise_signal, noise_RIRs) 


		src_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.src_pos, self.mic_pos, nb_img, Tmax, self.fs, Tdiff=Tdiff, orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
		src_reverb = gpuRIR.simulateTrajectory(self.source_signal, src_RIRs)


		dp_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.src_pos, self.mic_pos, [1,1,1], 0.1, self.fs,
									orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
		dp_src_signal = gpuRIR.simulateTrajectory(self.source_signal, dp_RIRs)

		return src_reverb, dp_src_signal, noise_reverb

	def adjust_to_snr(self, mic_signals, dp_signals, noise_reverb):
		scale_noi = np.sqrt(np.sum(mic_signals[:,0]**2) / (np.sum(noise_reverb[:,0]**2) * (10**(self.SNR/10))))
		mic_signals = mic_signals + noise_reverb * scale_noi

		# normalize the root mean square of the mixture to a constant
		sph_len = mic_signals.shape[0]#*mic_signals.shape[1]          #All mics

		c = 1.0 * np.sqrt(sph_len / (np.sum(mic_signals[:,0]**2) + 1e-8))

		mic_signals *= c
		dp_signals *= c

		return mic_signals, dp_signals



	def get_rmsae(self, exclude_silences=False):
		""" Returns the Root Mean Square Angular Error (degrees) of the DOA estimation.
		The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
		"""
		if not exclude_silences:
			return rms_angular_error_deg(torch.from_numpy(self.DOAw_pred).double(),
										 torch.from_numpy(self.DOAw).double() )
		else:
			silences = self.vad.mean(axis=1) < 2/3
			DOAw_pred = torch.from_numpy(self.DOAw_pred[np.invert(silences), :]).double()
			self.DOAw_pred[silences, :] = np.NaN
			DOAw = torch.from_numpy(self.DOAw[np.invert(silences), :]).double()
			return rms_angular_error_deg(DOAw_pred, DOAw)

	def findMapMaximum(self, exclude_silences=False):
		""" Generates the field DOAw_est_max with the DOA estimation using the SRP-PHAT maximums
		and returns its RMSAE (in degrees) if the field DOAw exists with the DOA groundtruth.
		The scene need to have the field maps with the SRP-PHAT map of each window.
		You can choose whether to include the silent frames into the RMSAE computation or not.
		"""
		max_flat_idx = self.maps.reshape((self.maps.shape[0], -1)).argmax(1)
		theta_max_idx, phi_max_idx = np.unravel_index(max_flat_idx, self.maps.shape[1:])

		# Index to angle (radians)
		if self.array_setup.arrayType == 'planar':
			theta = np.linspace(0, np.pi/2, self.maps.shape[1])
		else:
			theta= np.linspace(0, np.pi, self.maps.shape[1])
		phi = np.linspace(-np.pi, np.pi, self.maps.shape[2]+1)
		phi = phi[:-1]
		DOAw_srpMax = np.stack((theta[theta_max_idx], phi[phi_max_idx]), axis=-1)
		self.DOAw_srpMax = DOAw_srpMax

		if not exclude_silences:
			if hasattr(self, 'DOAw'):
				return rms_angular_error_deg(torch.from_numpy(self.DOAw_srpMax),
														torch.from_numpy(self.DOAw))
		else:
			silences = self.vad.mean(axis=1) < 2/3
			self.DOAw_srpMax[silences] = np.NaN
			if hasattr(self, 'DOAw'):
				return rms_angular_error_deg(torch.from_numpy(DOAw_srpMax[np.invert(silences), :]),
														 torch.from_numpy(self.DOAw[np.invert(silences), :]) )

	def plotScene(self, view='3D'):
		""" Plots the source trajectory and the microphones within the room
		"""
		assert view in ['3D', 'XYZ', 'XY', 'XZ', 'YZ']

		fig = plt.figure()

		if view == '3D' or view == 'XYZ':
			ax = Axes3D(fig)
			ax.set_xlim3d(0, self.room_sz[0])
			ax.set_ylim3d(0, self.room_sz[1])
			ax.set_zlim3d(0, self.room_sz[2])

			ax.scatter(self.traj_pts[:,0], self.traj_pts[:,1], self.traj_pts[:,2])
			ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1], self.mic_pos[:,2])
			ax.text(self.traj_pts[0,0], self.traj_pts[0,1], self.traj_pts[0,2], 'start')

			ax.set_title('$T_{60}$' + ' = {:.3f}s, SNR = {:.1f}dB'.format(self.T60, self.SNR))
			ax.set_xlabel('x [m]')
			ax.set_ylabel('y [m]')
			ax.set_zlabel('z [m]')

		else:
			ax = fig.add_subplot(111)
			plt.gca().set_aspect('equal', adjustable='box')

			if view == 'XY':
				ax.set_xlim(0, self.room_sz[0])
				ax.set_ylim(0, self.room_sz[1])
				ax.scatter(self.traj_pts[:,0], self.traj_pts[:,1])
				ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1])
				ax.text(self.traj_pts[0,0], self.traj_pts[0,1], 'start')
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('x [m]')
				ax.set_ylabel('y [m]')
			elif view == 'XZ':
				ax.set_xlim(0, self.room_sz[0])
				ax.set_ylim(0, self.room_sz[2])
				ax.scatter(self.traj_pts[:,0], self.traj_pts[:,2])
				ax.scatter(self.mic_pos[:,0], self.mic_pos[:,2])
				ax.text(self.traj_pts[0,0], self.traj_pts[0,2], 'start')
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('x [m]')
				ax.set_ylabel('z [m]')
			elif view == 'YZ':
				ax.set_xlim(0, self.room_sz[1])
				ax.set_ylim(0, self.room_sz[2])
				ax.scatter(self.traj_pts[:,1], self.traj_pts[:,2])
				ax.scatter(self.mic_pos[:,1], self.mic_pos[:,2])
				ax.text(self.traj_pts[0,1], self.traj_pts[0,2], 'start')
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('y [m]')
				ax.set_ylabel('z [m]')

		plt.show()

	def plotDOA(self):
		""" Plots the groundtruth DOA
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.t, self.DOA * 180/np.pi)

		ax.legend(['Elevation', 'Azimuth'])
		ax.set_xlabel('time [s]')
		ax.set_ylabel('DOA [º]')

		plt.show()

	def plotEstimation(self, legned_loc='best'):
		""" Plots the DOA groundtruth and its estimation.
		The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
		If the scene has the field DOAw_srpMax with the SRP-PHAT estimation, it also plots it.
		"""
		fig = plt.figure()
		gs = fig.add_gridspec(7, 1)
		ax = fig.add_subplot(gs[0,0])
		ax.plot(self.t, self.source_signal)
		plt.xlim(self.tw[0], self.tw[-1])
		plt.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

		ax = fig.add_subplot(gs[1:,0])
		ax.plot(self.tw, self.DOAw * 180/np.pi)
		plt.gca().set_prop_cycle(None)
		ax.plot(self.tw, self.DOAw_pred * 180/np.pi, '--')
		if hasattr(self, 'DOAw_srpMax'):
			plt.gca().set_prop_cycle(None)
			ax.plot(self.tw, self.DOAw_srpMax * 180 / np.pi, 'x', markersize=4)

		plt.legend(['Elevation', 'Azimuth'], loc=legned_loc)
		plt.xlabel('time [s]')
		plt.ylabel('DOA [º]')

		silences = self.vad.mean(axis=1) < 2/3
		silences_idx = silences.nonzero()[0]
		start, end = [], []
		for i in silences_idx:
			if not i - 1 in silences_idx:
				start.append(i)
			if not i + 1 in silences_idx:
				end.append(i)
		for s, e in zip(start, end):
			plt.axvspan((s-0.5)*self.tw[1], (e+0.5)*self.tw[1], facecolor='0.5', alpha=0.5)

		plt.xlim(self.tw[0], self.tw[-1])
		plt.show()

	def plotMap(self, w_idx):
		""" Plots the SRP-PHAT map of the window w_idx.
		If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also plot them.
		"""
		maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)

		thetaMax = np.pi / 2 if self.array_setup.arrayType == 'planar' else np.pi
		theta = np.linspace(0, thetaMax, maps.shape[-2])
		phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

		map = maps[w_idx, ...]
		DOA = self.DOAw[w_idx, ...] if hasattr(self, 'DOAw') else None
		DOA_pred = self.DOAw_pred[w_idx, ...] if hasattr(self, 'DOAw_pred') else None
		DOA_srpMax = self.DOAw_srpMax[w_idx, ...] if hasattr(self, 'DOAw_srpMax') else None

		plot_srp_map(theta, phi, map, DOA, DOA_pred, DOA_srpMax)

	def animateScene(self, fps=10, file_name=None):
		""" Creates an animation with the SRP-PHAT maps of each window.
		The scene need to have the field maps with the SRP-PHAT map of each window.
		If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also includes them.
		"""
		maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)
		thetaMax = np.pi/2 if self.array_setup=='planar' else np.pi
		theta = np.linspace(0, thetaMax, maps.shape[-2])
		phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

		DOAw = self.DOAw if hasattr(self, 'DOAw') else None
		DOAw_pred = self.DOAw_pred if hasattr(self, 'DOAw_pred') else None
		DOAw_srpMax = self.DOAw_srpMax if hasattr(self, 'DOAw_srpMax') else None

		animate_trajectory(theta, phi, maps, fps, DOAw, DOAw_pred, DOAw_srpMax, file_name)