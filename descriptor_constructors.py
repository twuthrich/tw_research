'''
This script processes the camera imagery into optical flow descriptors
7/27/2018 
Tori Wuthrich

'''

import pdb
import rosbag
import numpy as np
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
import scipy
import scipy.stats
import yaml
import sys
from descriptor_building import *
import os
from plot_grid_descriptor_utils import *
from plot_polar_descriptor_utils import *

def cart2pol(x, y):
	r = np.reshape(np.sqrt(x**2 + y**2), (-1, 1))
	theta = np.arctan2(y, x)
	theta_pos = np.array([[t + 2*np.pi if t<0 else t for t in theta]]).T
	return(r, theta_pos)


# Descriptor 1: Inner circle with radius of 90. 
# Middle Discarded, outside partitioned into 8
def build_descriptor_1(x, y, vx, vy):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	polar_statistic = params_dict['polar_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	r_circle = 90
	thetaedges = np.linspace(0, 2*np.pi, 9)

	# Convert points to polar
	(r, theta) = cart2pol(x - image_w/2.0, -(y - image_h/2.0))

	# Initialize lists to store descriptor values
	(vx_d, vy_d)  = ([], [])

	section_sum = 0
	for i in range(len(thetaedges) - 1):

		# Find indices of coordinates that are in each region
		i_theta = np.intersect1d(np.where(theta > thetaedges[i])[0], np.where(theta <= thetaedges[i+1])[0])
		i_r = np.where(r >= r_circle)[0]
		i_section = np.intersect1d(i_theta, i_r)
		section_sum += len(i_section)

		# Calculate the statistic specified in .yaml
		if np.shape(i_section)[0] == 0:
			(x_stat, y_stat) = (0, 0)
		elif polar_statistic == "median":
			(x_stat, y_stat) = (np.median(vx[i_section]), np.median(vy[i_section]))
		elif polar_statistic == "mean":
			(x_stat, y_stat) = (np.mean(vx[i_section]), np.mean(vy[i_section]))
		else:
			Exception("Statistic must be either median or mean")

		# Update lists storing descriptor values
		vx_d.append(x_stat)
		vy_d.append(y_stat)

	# Reshape to size for descriptors
	vxs = np.reshape(vx_d, (-1, 1, 1))
	vys = np.reshape(vy_d, (-1, 1, 1))

	descriptor = np.concatenate((vxs, vys), axis=2)

	return descriptor

# Descriptor 1: Inner circle with radius of 90. 
# Middle Discarded, outside partitioned into 8
# For use when only upper part of image is being used
def build_descriptor_1_upper(x, y, vx, vy):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	polar_statistic = params_dict['polar_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	valid_indices = np.where(y < image_h/2)
	y = y[valid_indices]
	x = x[valid_indices]
	vx = vx[valid_indices]
	vy = vy[valid_indices]
	image_h = image_h/2




	r_circle = 45
	thetaedges = np.linspace(0, 2*np.pi, 9)

	# Convert points to polar
	(r, theta) = cart2pol(x - image_w/2.0, -(y - image_h/2.0))

	# Initialize lists to store descriptor values
	(vx_d, vy_d)  = ([], [])

	section_sum = 0
	for i in range(len(thetaedges) - 1):

		# Find indices of coordinates that are in each region
		i_theta = np.intersect1d(np.where(theta > thetaedges[i])[0], np.where(theta <= thetaedges[i+1])[0])
		i_r = np.where(r >= r_circle)[0]
		i_section = np.intersect1d(i_theta, i_r)
		section_sum += len(i_section)

		# Calculate the statistic specified in .yaml
		if np.shape(i_section)[0] == 0:
			(x_stat, y_stat) = (0, 0)
		elif polar_statistic == "median":
			(x_stat, y_stat) = (np.median(vx[i_section]), np.median(vy[i_section]))
		elif polar_statistic == "mean":
			(x_stat, y_stat) = (np.mean(vx[i_section]), np.mean(vy[i_section]))
		else:
			Exception("Statistic must be either median or mean")

		# Update lists storing descriptor values
		vx_d.append(x_stat)
		vy_d.append(y_stat)

	# Reshape to size for descriptors
	vxs = np.reshape(vx_d, (-1, 1, 1))
	vys = np.reshape(vy_d, (-1, 1, 1))

	descriptor = np.concatenate((vxs, vys), axis=2)

	return descriptor

# Descriptor 2: Inner circle with radius of 90. 
# Middle NOT Discarded, outside partitioned into 8
def build_descriptor_2(x, y, vx, vy):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	polar_statistic = params_dict['polar_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	r_circle = 90
	thetaedges = np.linspace(0, 2*np.pi, 9)

	# Convert points to polar
	(r, theta) = cart2pol(x - image_w/2.0, -(y - image_h/2.0))

	# Initialize lists to store descriptor values
	(vx_d, vy_d)  = ([], [])

	section_sum = 0
	for i in range(len(thetaedges) - 1):

		# Find indices of coordinates that are in each region
		i_theta = np.intersect1d(np.where(theta > thetaedges[i])[0], np.where(theta <= thetaedges[i+1])[0])
		i_r = np.where(r >= r_circle)[0]
		i_section = np.intersect1d(i_theta, i_r)
		section_sum += len(i_section)

		# Calculate the statistic specified in .yaml
		if np.shape(i_section)[0] == 0:
			(x_stat, y_stat) = (0, 0)
		elif polar_statistic == "median":
			(x_stat, y_stat) = (np.median(vx[i_section]), np.median(vy[i_section]))
		elif polar_statistic == "mean":
			(x_stat, y_stat) = (np.mean(vx[i_section]), np.mean(vy[i_section]))
		else:
			Exception("Statistic must be either median or mean")

		# Update lists storing descriptor values
		vx_d.append(x_stat)
		vy_d.append(y_stat)

	# Calculate stat for middle sections
	i_section = np.where(r < r_circle)[0]

	# Calculate the statistic specified in .yaml
	if np.shape(i_section)[0] == 0:
		(x_stat, y_stat) = (0, 0)
	elif polar_statistic == "median":
		(x_stat, y_stat) = (np.median(vx[i_section]), np.median(vy[i_section]))
	elif polar_statistic == "mean":
		(x_stat, y_stat) = (np.mean(vx[i_section]), np.mean(vy[i_section]))
	else:
		Exception("Statistic must be either median or mean")
	
	vx_d.append(x_stat)
	vy_d.append(y_stat)

	# Reshape to size for descriptors
	vxs = np.reshape(vx_d, (-1, 1, 1))
	vys = np.reshape(vy_d, (-1, 1, 1))

	descriptor = np.concatenate((vxs, vys), axis=2)

	return descriptor


# Descriptor 3: 4 circle and 8 lines, middle cut out
def build_descriptor_3(x, y, vx, vy):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	polar_statistic = params_dict['polar_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	l_max = 184.0/np.sin(np.pi/4)
	r_max = np.sqrt((image_h/2.0)**2 + (image_w/2.0)**2)

	redges = np.append(np.linspace(0, l_max, 5), r_max)[1:]
	thetaedges = np.linspace(0, 2*np.pi, 9)

	# Convert points to polar
	(r, theta) = cart2pol(x - image_w/2.0, -(y - image_h/2.0))
	# Initialize lists to store descriptor values
	(vx_d, vy_d)  = ([], [])

	section_sum = 0
	# outer loop through radii
	for i_r in range(len(redges) - 1):
		r_inner = redges[i_r]
		r_outer = redges[i_r+1]

		r_indices = np.intersect1d(np.where(r > r_inner)[0], np.where(r < r_outer)[0])

		# inner loop through angles
		for i_theta in range(len(thetaedges) - 1):
			theta_lo = thetaedges[i_theta]
			theta_hi = thetaedges[i_theta+1]

			theta_indices = np.intersect1d(np.where(theta > theta_lo)[0], np.where(theta <= theta_hi)[0])

			section_indices = np.intersect1d(theta_indices, r_indices)
			section_sum += len(section_indices)

			if np.shape(section_indices)[0] == 0:
				(x_stat, y_stat) = (0, 0)
			elif polar_statistic == "median":
				(x_stat, y_stat) = (np.median(vx[section_indices]), np.median(vy[section_indices]))
			elif polar_statistic == "mean":
				(x_stat, y_stat) = (np.mean(vx[section_indices]), np.mean(vy[section_indices]))
			else:
				Exception("Statistic must be either median or mean")

			# CHECKS 
			if len(section_indices) > 0:
				assert np.max(r[section_indices]) < r_outer
				assert np.min(r[section_indices]) > r_inner
				assert np.max(theta[section_indices] < theta_hi)
				assert np.min(theta[section_indices] > theta_lo)

			# Update lists storing descriptor values
			vx_d.append(x_stat)
			vy_d.append(y_stat)

	# CHECKS Make sure that sections not in the image are zero
	assert vx_d[25] == 0
	assert vx_d[26] == 0
	assert vx_d[29] == 0
	assert vx_d[30] == 0

	assert vy_d[25] == 0
	assert vy_d[26] == 0
	assert vy_d[29] == 0
	assert vy_d[30] == 0

	# Delete all sections that are outside of the image frame
	del_indices = [25, 26, 29, 30]
	vx_d = np.delete(vx_d, del_indices)
	vy_d = np.delete(vy_d ,del_indices)

	# Reshape to size for descriptors
	vxs = np.reshape(vx_d, (-1, 1, 1))
	vys = np.reshape(vy_d, (-1, 1, 1))

	descriptor = np.concatenate((vxs, vys), axis=2)

	return descriptor

def build_descriptor_3_sim(x, y, vx, vy):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	polar_statistic = params_dict['polar_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	l_max = 240.0/np.sin(np.pi/4)
	r_max = np.sqrt((image_h/2.0)**2 + (image_w/2.0)**2)

	redges = np.append(np.linspace(0, l_max, 5), r_max)[1:]
	thetaedges = np.linspace(0, 2*np.pi, 9)

	# Convert points to polar
	(r, theta) = cart2pol(x - image_w/2.0, -(y - image_h/2.0))
	# Initialize lists to store descriptor values
	(vx_d, vy_d)  = ([], [])

	section_sum = 0
	# outer loop through radii
	for i_r in range(len(redges) - 1):
		r_inner = redges[i_r]
		r_outer = redges[i_r+1]

		r_indices = np.intersect1d(np.where(r > r_inner)[0], np.where(r < r_outer)[0])

		# inner loop through angles
		for i_theta in range(len(thetaedges) - 1):
			theta_lo = thetaedges[i_theta]
			theta_hi = thetaedges[i_theta+1]

			theta_indices = np.intersect1d(np.where(theta > theta_lo)[0], np.where(theta <= theta_hi)[0])

			section_indices = np.intersect1d(theta_indices, r_indices)
			section_sum += len(section_indices)

			if np.shape(section_indices)[0] == 0:
				(x_stat, y_stat) = (0, 0)
			elif polar_statistic == "median":
				(x_stat, y_stat) = (np.median(vx[section_indices]), np.median(vy[section_indices]))
			elif polar_statistic == "mean":
				(x_stat, y_stat) = (np.mean(vx[section_indices]), np.mean(vy[section_indices]))
			else:
				Exception("Statistic must be either median or mean")

			# CHECKS 
			if len(section_indices) > 0:
				assert np.max(r[section_indices]) < r_outer
				assert np.min(r[section_indices]) > r_inner
				assert np.max(theta[section_indices] < theta_hi)
				assert np.min(theta[section_indices] > theta_lo)

			# Update lists storing descriptor values
			vx_d.append(x_stat)
			vy_d.append(y_stat)

	# CHECKS Make sure that sections not in the image are zero
	assert vx_d[25] == 0
	assert vx_d[26] == 0
	assert vx_d[29] == 0
	assert vx_d[30] == 0

	assert vy_d[25] == 0
	assert vy_d[26] == 0
	assert vy_d[29] == 0
	assert vy_d[30] == 0

	# Delete all sections that are outside of the image frame
	del_indices = [25, 26, 29, 30]
	vx_d = np.delete(vx_d, del_indices)
	vy_d = np.delete(vy_d ,del_indices)

	# Reshape to size for descriptors
	vxs = np.reshape(vx_d, (-1, 1, 1))
	vys = np.reshape(vy_d, (-1, 1, 1))

	descriptor = np.concatenate((vxs, vys), axis=2)

	return descriptor

# Descriptor 3: 4 circle and 8 lines, middle cut out
def build_descriptor_3_upper(x, y, vx, vy):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	polar_statistic = params_dict['polar_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	valid_indices = np.where(y < image_h/2)
	y = y[valid_indices]
	x = x[valid_indices]
	vx = vx[valid_indices]
	vy = vy[valid_indices]
	image_h = image_h/2

	l_max = 92.0/np.sin(np.pi/4)
	r_max = np.sqrt((image_h/2.0)**2 + (image_w/2.0)**2)

	redges = np.append(np.linspace(0, l_max, 5), r_max)[1:]
	thetaedges = np.linspace(0, 2*np.pi, 9)

	# Convert points to polar
	(r, theta) = cart2pol(x - image_w/2.0, -(y - image_h/2.0))
	# Initialize lists to store descriptor values
	(vx_d, vy_d)  = ([], [])

	section_sum = 0
	# outer loop through radii
	for i_r in range(len(redges) - 1):
		r_inner = redges[i_r]
		r_outer = redges[i_r+1]

		r_indices = np.intersect1d(np.where(r > r_inner)[0], np.where(r < r_outer)[0])

		# inner loop through angles
		for i_theta in range(len(thetaedges) - 1):
			theta_lo = thetaedges[i_theta]
			theta_hi = thetaedges[i_theta+1]

			theta_indices = np.intersect1d(np.where(theta > theta_lo)[0], np.where(theta <= theta_hi)[0])

			section_indices = np.intersect1d(theta_indices, r_indices)
			section_sum += len(section_indices)

			if np.shape(section_indices)[0] == 0:
				(x_stat, y_stat) = (0, 0)
			elif polar_statistic == "median":
				(x_stat, y_stat) = (np.median(vx[section_indices]), np.median(vy[section_indices]))
			elif polar_statistic == "mean":
				(x_stat, y_stat) = (np.mean(vx[section_indices]), np.mean(vy[section_indices]))
			else:
				Exception("Statistic must be either median or mean")

			# CHECKS 
			if len(section_indices) > 0:
				assert np.max(r[section_indices]) < r_outer
				assert np.min(r[section_indices]) > r_inner
				assert np.max(theta[section_indices] < theta_hi)
				assert np.min(theta[section_indices] > theta_lo)

			# Update lists storing descriptor values
			vx_d.append(x_stat)
			vy_d.append(y_stat)

	# CHECKS Make sure that sections not in the image are zero
	assert vx_d[25] == 0
	assert vx_d[26] == 0
	assert vx_d[29] == 0
	assert vx_d[30] == 0

	assert vy_d[25] == 0
	assert vy_d[26] == 0
	assert vy_d[29] == 0
	assert vy_d[30] == 0

	# Delete all sections that are outside of the image frame
	del_indices = [25, 26, 29, 30]
	vx_d = np.delete(vx_d, del_indices)
	vy_d = np.delete(vy_d ,del_indices)

	# Reshape to size for descriptors
	vxs = np.reshape(vx_d, (-1, 1, 1))
	vys = np.reshape(vy_d, (-1, 1, 1))

	descriptor = np.concatenate((vxs, vys), axis=2)

	return descriptor


# Descriptor 4: Same as (3), but not using the lower half
def build_descriptor_4(x, y, vx, vy):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	polar_statistic = params_dict['polar_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	l_max = 184.0/np.sin(np.pi/4)
	r_max = np.sqrt((image_h/2.0)**2 + (image_w/2.0)**2)

	redges = np.append(np.linspace(0, l_max, 5), r_max)[1:]
	thetaedges = np.linspace(0, np.pi, 5)

	# Convert points to polar
	(r, theta) = cart2pol(x - image_w/2.0, -(y - image_h/2.0))
	# Initialize lists to store descriptor values
	(vx_d, vy_d)  = ([], [])

	section_sum = 0
	# outer loop through radii
	for i_r in range(len(redges) - 1):
		r_inner = redges[i_r]
		r_outer = redges[i_r+1]

		r_indices = np.intersect1d(np.where(r > r_inner)[0], np.where(r < r_outer)[0])

		# inner loop through angles
		for i_theta in range(len(thetaedges) - 1):
			theta_lo = thetaedges[i_theta]
			theta_hi = thetaedges[i_theta+1]

			theta_indices = np.intersect1d(np.where(theta > theta_lo)[0], np.where(theta <= theta_hi)[0])

			section_indices = np.intersect1d(theta_indices, r_indices)
			section_sum += len(section_indices)

			if np.shape(section_indices)[0] == 0:
				(x_stat, y_stat) = (0, 0)
			elif polar_statistic == "median":
				(x_stat, y_stat) = (np.median(vx[section_indices]), np.median(vy[section_indices]))
			elif polar_statistic == "mean":
				(x_stat, y_stat) = (np.mean(vx[section_indices]), np.mean(vy[section_indices]))
			else:
				Exception("Statistic must be either median or mean")

			# CHECKS 
			if len(section_indices) > 0:
				assert np.max(r[section_indices]) < r_outer
				assert np.min(r[section_indices]) > r_inner
				assert np.max(theta[section_indices] < theta_hi)
				assert np.min(theta[section_indices] > theta_lo)

			# Update lists storing descriptor values
			vx_d.append(x_stat)
			vy_d.append(y_stat)

	# CHECKS Make sure that sections not in the image are zero
	assert vx_d[13] == 0
	assert vx_d[14] == 0

	assert vy_d[13] == 0
	assert vy_d[14] == 0

	# Delete all sections that are outside of the image frame
	del_indices = [13, 14]
	vx_d = np.delete(vx_d, del_indices)
	vy_d = np.delete(vy_d ,del_indices)

	# Reshape to size for descriptors
	vxs = np.reshape(vx_d, (-1, 1, 1))
	vys = np.reshape(vy_d, (-1, 1, 1))

	descriptor = np.concatenate((vxs, vys), axis=2)

	return descriptor







