from utils import Parameter
from locata_utils import *
from debug import dbg_print

import numpy as np
import random
import gpuRIR
from AcousticScene import AcousticScene

from collections import namedtuple
import torch
import torchaudio

#Ablation studies
class TestConfig():
	def __init__(self, array_setup, test_snr, test_t60, static_prob, src_mic_dist, non_linear_prob, nb_points, same_plane, noi_mic_dist=None):
		
		#Creating Static Dataset 
		self.snr = Parameter(test_snr)
		self.t60 = Parameter(test_t60)

		self.room_size = Parameter([3, 3, 2.5],[10, 8, 6])
		self.abs_weights = Parameter([0.5]*6, [1.0]*6)  # Random absorption weights ratios between walls
		self.array_setup = array_setup

		self.array_pos = Parameter([0.4, 0.4, 0.1],[0.6, 0.6, 0.5])  # For front and back of the array Circular motion

		#self.array_pos = Parameter([0.4, 0.1, 0.1],[0.6, 0.9, 0.5])  # Infront of the mic array


		self.static_prob = static_prob
		self.non_linear_motion_prob = non_linear_prob
		self.nb_points = nb_points

		self._same_plane = same_plane

		self.src_mic_dist = Parameter(src_mic_dist)
		self.noi_mic_dist = Parameter(src_mic_dist) if noi_mic_dist is None else Parameter(noi_mic_dist)
		self.theta_grid = Parameter(0, np.radians(360))            #degrees -> radians


	def get_pos(self, r, azimuth):
		"""
		Support for In plane only
		"""
		src_pos = np.array([r*np.cos(azimuth), r*np.sin(azimuth), 0.0])
		return src_pos

	def gen_circular_motion_pos(self, scenario="src_moving"):
		"""
		Circular motion: Non Overlaping Source Trajectory and noise
		Allowed Source angular motion: [10 - 180] degrees
		Minimun Source to Noise Angular distance: 5 degrees

		Output: src_pos: [nb_points, 3]
				noi_pos: [1, 3]
		"""
		
		MIN_SRC_MOV = 10
		MAX_SRC_MOV = 180
		MIN_SRC_NOI_DIST = 5

		src_mic_dist = self.src_mic_dist.getValue()
		noi_mic_dist = self.noi_mic_dist.getValue()

		if "source_moving"==scenario: 
			# Generate random theta on cirle
			src_start_theta = self.theta_grid.getValue()
			src_end_theta = src_start_theta + np.radians(MIN_SRC_MOV) + random.random()* np.radians((MAX_SRC_MOV - MIN_SRC_MOV) )    # Min src movement in terms of angle : 10 degrees
			delta_src_movement = np.degrees(np.abs(src_end_theta - src_start_theta))
			noise_theta = src_end_theta + np.radians(MIN_SRC_NOI_DIST) + random.random()* np.radians((360 - delta_src_movement - MIN_SRC_NOI_DIST*2 ))    # Min src to noise dist in terms of angle : 5 degrees

			#print(f"src_start: {np.degrees(src_start_theta) % 360}, src_end: {np.degrees(src_end_theta) % 360}, noise: {np.degrees(noise_theta) % 360}")

			noise_pos = self.get_pos(noi_mic_dist, noise_theta)

			src_theta_traj = np.linspace(src_start_theta, src_end_theta, self.nb_points)
			src_pos = np.array( [ self.get_pos(src_mic_dist, theta) for theta in src_theta_traj ] )	

			noise_pos = np.expand_dims(noise_pos, axis=0)

		elif "noise_moving"==scenario:
			# Generate random theta on cirle
			noi_start_theta = self.theta_grid.getValue()
			noi_end_theta = noi_start_theta + np.radians(MIN_SRC_MOV) + random.random()* np.radians((MAX_SRC_MOV - MIN_SRC_MOV) )    # Min src movement in terms of angle : 10 degrees
			delta_src_movement = np.degrees(np.abs(noi_end_theta - noi_start_theta))
			src_theta = noi_end_theta + np.radians(MIN_SRC_NOI_DIST) + random.random()* np.radians((360 - delta_src_movement - MIN_SRC_NOI_DIST*2 ))    # Min src to noise dist in terms of angle : 5 degrees

			#print(f"src_start: {np.degrees(src_start_theta) % 360}, src_end: {np.degrees(src_end_theta) % 360}, noise: {np.degrees(noise_theta) % 360}")

			src_pos = self.get_pos(src_mic_dist, src_theta)

			noi_theta_traj = np.linspace(noi_start_theta, noi_end_theta, self.nb_points)
			noise_pos = np.array( [ self.get_pos(noi_mic_dist, theta) for theta in noi_theta_traj ] )	

			src_pos = np.expand_dims(src_pos, axis=0)
		else:
			pass


		return src_pos, noise_pos

	def gen_static_scene_on_circle_pos(self):
		"""
		Circular Scene: 
		Minimun Source to Noise Angular distance: 5 degrees

		Output: src_pos: [1, 3]
				noi_pos: [1, 3]
		"""
		MIN_SRC_NOI_DIST = 5

		src_mic_dist = self.src_mic_dist.getValue()
		noi_mic_dist = self.noi_mic_dist.getValue()
	
		# Generate random theta on cirle
		src_theta = self.theta_grid.getValue()
		noi_theta = src_theta + np.radians(MIN_SRC_NOI_DIST) + np.radians(random.random()*(360 - MIN_SRC_NOI_DIST*2))    # Min src to noise dist in terms of angle : 5 degrees
		
		src_pos = self.get_pos(src_mic_dist, src_theta)
		noise_pos = self.get_pos(noi_mic_dist, noi_theta)

		src_pos = np.expand_dims(src_pos, axis=0)
		noise_pos = np.expand_dims(noise_pos, axis=0)

		return src_pos, noise_pos


	def gen_semi_circular_motion_pos(self):
		"""
		Semi Circular motion: Non Overlaping Source Trajectory and noise
		Allowed Source angular motion: [10 - 180] degrees
		Minimun Source to Noise Angular distance: 5 degrees

		Output: src_pos: [nb_points, 3]
				noi_pos: [1, 3]
		"""
		
		MIN_SRC_MOV = 10
		MAX_SRC_MOV = 180
		MIN_SRC_NOI_DIST = 5

		src_mic_dist = self.src_mic_dist.getValue()

		# Generate random theta on cirle
		src_start_theta = self.theta_grid.getValue()
		src_end_theta = src_start_theta + np.radians(MIN_SRC_MOV) + random.random()* np.radians((MAX_SRC_MOV - MIN_SRC_MOV) )    # Min src movement in terms of angle : 10 degrees
		delta_src_movement = np.degrees(np.abs(src_end_theta - src_start_theta))
		noise_theta = src_end_theta + np.radians(MIN_SRC_NOI_DIST) + random.random()* np.radians((360 - delta_src_movement - MIN_SRC_NOI_DIST*2 ))    # Min src to noise dist in terms of angle : 5 degrees

		#print(f"src_start: {np.degrees(src_start_theta) % 360}, src_end: {np.degrees(src_end_theta) % 360}, noise: {np.degrees(noise_theta) % 360}")

		noise_pos = self.get_pos(src_mic_dist, noise_theta)

		src_theta_traj = np.linspace(src_start_theta, src_end_theta, self.nb_points)
		src_pos = np.array( [ self.get_pos(src_mic_dist, theta) for theta in src_theta_traj ] )	

		noise_pos = np.expand_dims(noise_pos, axis=0)

		return src_pos, noise_pos
	"""
	def gen_linear_motion_pos(self):
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

		traj_pts = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
		if random.random() <= self.non_linear_motion_prob:                          # Non-Linear Motion 
			traj_pts += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
			dbg_print("Moving Source Non-Linear\n")
		else:
			dbg_print("Moving Source Linear\n")

		noise_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

		return src_pos, noise_pos
	"""

	def _create_acoustic_scene_config(self, scene_type, scenario=None):

		# Room
		room_sz = self.room_size.getValue()
		T60 = self.t60.getValue()
		abs_weights = self.abs_weights.getValue()
		beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights)

		# Microphones
		array_pos = self.array_pos.getValue() * room_sz
		mic_pos = array_pos + self.array_setup.mic_pos

		#source_trajectory
		if "Static" == scene_type: #random.random() < self.static_prob: 
			src_pos, noise_pos = self.gen_static_scene_on_circle_pos()

		elif "CircularMotion" == scene_type:

			src_pos, noise_pos = self.gen_circular_motion_pos(scenario)

		else:
			print("Needs to be Implemented \n")

		#global pos
		noise_pos += array_pos
		src_pos += array_pos

		# *** assert src_pos, noise_pos inside room
		assert not ((src_pos >= room_sz).any() or (src_pos <= 0).any())
		assert not ((noise_pos >= room_sz).any() or (noise_pos <= 0).any())

		snr = self.snr.getValue()

		scene_dict = {}

		scene_dict['snr'] = snr
		scene_dict['t60'] = T60
		scene_dict['beta'] = beta
		scene_dict['room_sz'] = room_sz
		scene_dict['abs_weights'] = abs_weights
		scene_dict['mic_pos'] = mic_pos
		scene_dict['array_pos'] = array_pos # mic center
		scene_dict['src_traj_pts'] = src_pos
		scene_dict['noise_pos'] = noise_pos
		scene_dict['nb_points'] = src_pos.shape[0]

		return scene_dict

	"""
	def simulate_acoustic_scene(self, scene_dict, source_signal, noise_signal):

		self.fs = 16000

		nb_points = scene_dict['nb_points']
		traj_pts = scene_dict['src_traj_pts']

		# Interpolate trajectory points
		timestamps = np.arange(nb_points) * len(source_signal) / self.fs / nb_points
		t = np.arange(len(source_signal))/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		acoustic_scene = AcousticScene(
			room_sz = scene_dict['room_sz'],
			T60 = scene_dict['t60'],
			beta = scene_dict['beta'],
			SNR = scene_dict['snr'],
			array_setup = self.array_setup,
			mic_pos = scene_dict['mic_pos'],
			source_signal = source_signal, #sph[0],
			noise_signal = noise_signal, #noi[0],
			fs = self.fs,
			t = t,
			traj_pts = scene_dict['src_traj_pts'],
			timestamps = timestamps,
			trajectory = trajectory,
			DOA = cart2sph(trajectory - scene_dict['array_pos'])[:,1:3], 
			noise_pos = scene_dict['noise_pos'],
			noise_traj_pts = None    
		)

		src_reverb, dp_src_signal, noise_reverb = acoustic_scene.static_simulate()
		mic_signals, dp_signals = acoustic_scene.adjust_to_snr(mic_signals=src_reverb, dp_signals=dp_src_signal, noise_reverb=noise_reverb)

		return mic_signals, dp_signals, acoustic_scene
	"""

if __name__=="__main__":

	# Named tuple with the characteristics of a microphone array and definitions of the LOCATA arrays:
	ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern')

	array_setup_10cm_2mic = ArraySetup(arrayType='planar', 
										orV = np.array([0.0, 1.0, 0.0]),
										mic_pos = np.array((( 0.05,  0.000, 0.000),
															(-0.05,  0.000, 0.000))), 
										mic_orV = np.array(((0.0, 1.0, 0.0),
															(0.0, 1.0, 0.0))), 
										mic_pattern = 'omni'
									)

	snr = 5
	t60 = 0.4
	src_mic_dist = 2.0
	noi_mic_dist = 1.0
	scenario = "source_moving"
	test_config = TestConfig(array_setup=array_setup_10cm_2mic, test_snr=snr, test_t60=t60, static_prob=0.0, src_mic_dist = src_mic_dist, noi_mic_dist = noi_mic_dist, non_linear_prob=0.0, nb_points=16, same_plane=True)

	for idx in range(0, 21):									
		retry_flag = True
		retry_count = 0
		while(retry_flag):
			try:
				static_config_dict = test_config._create_acoustic_scene_config("Static")
				circular_motion_config_dict = test_config._create_acoustic_scene_config("CircularMotion", scenario)
				retry_flag = False
			except AssertionError:
				retry_flag = True
				retry_count += 1

		#breakpoint()
		pp_str = f'./configs/{scenario}/snr_{snr}_t60_{t60}/src_mic_dist_{src_mic_dist}_noi_mic_dist_{noi_mic_dist}/'
		torch.save(circular_motion_config_dict, f'{pp_str}config_circular_motion_{idx}.pt')
		torch.save(static_config_dict, f'{pp_str}config_static_circular_{idx}.pt')
		print(f"Successful Execution {idx}\n")


