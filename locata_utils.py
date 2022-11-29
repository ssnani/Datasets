import numpy as np
import torch

# %% Util functions

def acoustic_power(s):
	""" Acoustic power of after removing the silences.
	"""
	w = 512  # Window size for silent detection
	o = 256  # Window step for silent detection

	# Window the input signal
	s = np.ascontiguousarray(s)
	sh = (s.size - w + 1, w)
	st = s.strides * 2
	S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

	window_power = np.mean(S ** 2, axis=-1)
	th = 0.01 * window_power.max()  # Threshold for silent detection
	return np.mean(window_power[np.nonzero(window_power > th)])

def cart2sph(cart):
	xy2 = cart[:,0]**2 + cart[:,1]**2
	sph = np.zeros_like(cart)
	sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
	sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
	sph[:,2] = np.arctan2(cart[:,1], cart[:,0])
	return sph

def angular_error(the_pred, phi_pred, the_true, phi_true):
	""" Angular distance between spherical coordinates.
	"""
	aux = torch.cos(the_true) * torch.cos(the_pred) + \
		  torch.sin(the_true) * torch.sin(the_pred) * torch.cos(phi_true - phi_pred)

	return torch.acos(torch.clamp(aux, -0.99999, 0.99999))

def mean_square_angular_error(y_pred, y_true):
	""" Mean square angular distance between spherical coordinates.
	Each row contains one point in format (elevation, azimuth).
	"""
	the_true = y_true[:, 0]
	phi_true = y_true[:, 1]
	the_pred = y_pred[:, 0]
	phi_pred = y_pred[:, 1]

	return torch.mean(torch.pow(angular_error(the_pred, phi_pred, the_true, phi_true), 2), -1)

def rms_angular_error_deg(y_pred, y_true):
	""" Root mean square angular distance between spherical coordinates.
	Each input row contains one point in format (elevation, azimuth) in radians
	but the output is in degrees.
	"""
	return torch.sqrt(mean_square_angular_error(y_pred, y_true)) * 180 / pi
