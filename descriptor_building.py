#!/usr/bin/env python

'''
Function that takes the optical flow and associated (x,y) coordinates
and calculates the  descriptor 
'''

import  yaml
import numpy as np
import scipy, scipy.stats
import pdb

def build_grid_descriptor(x, y, vx, vy):

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	grid_statistic = params_dict['grid_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']
	grids = params_dict['grids']

	upperOnly = params_dict['upperOnly']

	###########################################

	if upperOnly:
		valid_indices = np.where(y < image_h/2)
		y = y[valid_indices]
		x = x[valid_indices]
		vx = vx[valid_indices]
		vy = vy[valid_indices]
		image_h = image_h/2

	bin_range = [[0, image_w], [0, image_h]]
	bins_n = [grids, grids]

	# get descriptor - mean of vx and vy in every square
	stat, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, statistic=grid_statistic, \
		values =(vx, vy), bins=grids, range=bin_range, expand_binnumbers=True)


	# Change nan's from empty squares to zeros 
	vx_d = np.nan_to_num(stat[0, :, :])
	vy_d = np.nan_to_num(stat[1, :, :])

	descriptors = np.dstack((vx_d, vy_d))

	return descriptors


def cart2pol(x, y):
	r = np.reshape(np.sqrt(x**2 + y**2), (-1, 1))
	theta = np.arctan2(y, x)
	theta_pos = np.array([[t + 2*np.pi if t<0 else t for t in theta]]).T
	return(r, theta_pos)

'''
This function builds optical flow descriptors for a single frame 
using the circular pattern
'''
def build_circle_descriptor(x, y, vx, vy, descriptors, redges, thetaedges, i):

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	polar_statistic = params_dict['polar_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']
	partition_center = params_dict['partition_center']

	###########################################

	# Convert x y to polar with new origin @ image center
	(r, theta) = cart2pol(x - image_w/2, y - image_h/2)

	# Lists to keep track of the statistics
	vxs = []
	vys = []

	# for each set of r's:
	for i_r in range(len(redges)-1):

		# Find range of r's for current section
		r_lo = redges[i_r]
		r_hi = redges[i_r + 1]

		# If not partitioning center, just check indices in r range
		if not partition_center and i_r == 0:
			section_indices = np.intersect1d(np.where(r >= r_lo)[0], np.where(r < r_hi)[0])
			# if no indices in section, set to zero
			if np.size(section_indices) == 0:
				(vx_stat, vy_stat) = (0, 0)
			else:		
				vx_stat = np.median(vx[section_indices])
				vy_stat = np.median(vy[section_indices])
			# Record statistics
			vxs.extend([vx_stat])
			vys.extend([vy_stat])

		else:
			# for each set of theta's:
			for i_t in range(len(thetaedges)-1):

				# Find range of theta's for current section
				theta_lo = thetaedges[i_t] 
				theta_hi = thetaedges[i_t + 1]

				# find indices in that range
				r_indices = np.intersect1d(np.where(r >= r_lo)[0], np.where(r < r_hi)[0])
				theta_indices = np.intersect1d(np.where(theta >= theta_lo)[0], np.where(theta < theta_hi)[0])
				section_indices = np.intersect1d(r_indices, theta_indices)

				# if no indices in section, set to zero
				if np.size(section_indices) == 0:
					(vx_stat, vy_stat) = (0, 0)
				else:
					if polar_statistic == "median":	
						vx_stat = np.median(vx[section_indices])
						vy_stat = np.median(vy[section_indices])
					elif polar_statistic == "mean":
						vx_stat = np.mean(vx[section_indices])
						vy_stat = np.mean(vy[section_indices])
					else:
						Exception("polar_statistic must be either median or mean")

				# Record statistics
				vxs.extend([vx_stat])
				vys.extend([vy_stat])

	# Update descriptors and return
	vxs = np.reshape(vxs, (-1, 1, 1))
	vys = np.reshape(vys, (-1, 1, 1))

	descriptors[:, :, :, i] = np.concatenate((vxs, vys), axis=2)
	return descriptors

def calc_edges():

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']
	n_circles = params_dict['n_circles']
	n_lines = params_dict['n_lines']

	###########################################
	redges = np.linspace(0, image_h/2, n_circles+1)

	# radius in outer section can be up to the max possible given image frame size
	r_max_image = np.array([np.linalg.norm([image_h/2.0, image_w/2.0])])
	redges = np.concatenate((redges, r_max_image))
	thetaedges = np.linspace(0, 2*np.pi, n_lines+1)

	# Create array of zeros for descriptor
	l_d = (n_circles+1)*n_lines
	return (redges, thetaedges)