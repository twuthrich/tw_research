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
from descriptor_constructors import *

def gen_descriptors(bag, name) :

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']
	grids = params_dict['grids']
	flatten_descriptor = params_dict['flatten_descriptor']
	descriptor_type = params_dict['descriptor_type']
	error_threshold = params_dict['error_threshold']
	sim = params_dict['sim']

	lk_params = dict(winSize=(params_dict['winSizeX'],params_dict['winSizeY']),\
		maxLevel=params_dict['maxLevel'])

	feature_params = dict(mask=None, \
		maxCorners=params_dict['maxCorners'],\
		qualityLevel=params_dict['qualityLevel'], 
		minDistance=params_dict['minDistance'], \
		blockSize=params_dict['blockSize'])

	# Subpixel accuracy parameters
	useSubPix = params_dict['useSubPix']

	subPixWinSize = params_dict['subPixWinSize']
	subPixZeroZone = params_dict['subPixZeroZone']
	subPixCount = params_dict['subPixCount']
	subPixEPS = params_dict['subPixEPS']

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, subPixCount, subPixEPS)

	# Image Cropping
	upperOnly = params_dict['upperOnly']

	# TODO: Update these to fit your application
	of_directory = "/home/toriw/tw_research/processed_bags/of"
	image_topic_name = "/airsim/image_raw"
	position_topic_name = "/acePositionData"

	###########################################

	mocap_dict = {}

	# Loop through bag, create dictionary time keys: time and values: msg
	with rosbag.Bag(bag, 'r') as bag: 
		for topic, msg, t, in bag.read_messages():
	
			if topic == image_topic_name:
				mocap_dict[t.to_sec()] = msg

	# Set previous values for first run
	sorted_keys = sorted(mocap_dict.keys())
	prev_time = sorted_keys[0]
	prev_img = CvBridge().imgmsg_to_cv2(mocap_dict[prev_time], "mono8")

	p0 = cv2.goodFeaturesToTrack(prev_img, **feature_params)

	# Sub-pixel 
	if useSubPix:
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, subPixCount, subPixEPS)
		p0 = cv2.cornerSubPix(prev_img, p0, (subPixWinSize, subPixWinSize), (subPixZeroZone, subPixZeroZone), criteria)

	descriptor_list = []


	# Set up grid descriptor 
	if descriptor_type == "grid":
		descriptors = np.zeros((grids, grids, 2, len(mocap_dict.keys())-1))

	for i in range(len(sorted_keys[1:])):

		cur_time = sorted_keys[i+1]

		dt = cur_time - prev_time

		# Convert image to cv2 
		cur_img = CvBridge().imgmsg_to_cv2(mocap_dict[cur_time], "mono8")

		# Calculate new positions of features from previous frame
		p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img, cur_img, p0, None, **lk_params)
		hi_error_indices = np.where(err > error_threshold)[0]
		lost_indices = np.where(st == 0)[0]
		del_indices = np.unique(np.concatenate((hi_error_indices, lost_indices)))

		# Take only p1 & corresponding p0 where tracker worked
		p1_filtered = np.delete(p1, del_indices, axis=0)
		p0_filtered = np.delete(p0, del_indices, axis=0)

		# Calculate OF from good p0s only
		test_p1, test_st, test_err = cv2.calcOpticalFlowPyrLK(prev_img, cur_img, p0_filtered, None, **lk_params)

		# Make sure propagating good p0's has no error 

		# Finite difference => optical flow
		flow = (p1_filtered - p0_filtered)/dt

		# Get features locations & velocities
		(x, y) = (p0_filtered[:, 0, 0], p0_filtered[:, 0, 1])
		(vx, vy) = (flow[:, 0, 0], flow[:, 0, 1])

		# Build descriptors 
		if descriptor_type == 'grid':
			descriptor_list.append(build_grid_descriptor(x, y, vx, vy))

		elif descriptor_type == 1:
			if upperOnly:
				descriptor_list.append(build_descriptor_1_upper(x, y, vx, vy))
			else:
				descriptor_list.append(build_descriptor_1(x, y, vx, vy))

		elif descriptor_type == 2:
			descriptor_list.append(build_descriptor_2(x, y, vx, vy))
		
		elif descriptor_type == 3:
			if upperOnly:
				descriptor_list.append(build_descriptor_3_upper(x, y, vx, vy))
			elif sim:
				descriptor_list.append(build_descriptor_3_sim(x, y, vx, vy))

			else:
				descriptor_list.append(build_descriptor_3(x, y, vx, vy))

		elif descriptor_type == 4:
			descriptor_list.append(build_descriptor_4(x, y, vx, vy))
			
		else:
			Exception('Must choose a valid descriptor type')
				
	    # Reset for next round
		p0 = cv2.goodFeaturesToTrack(cur_img, **feature_params)

		if useSubPix:
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, subPixCount, subPixEPS)
			p0 = cv2.cornerSubPix(prev_img, p0, (subPixWinSize, subPixWinSize), (subPixZeroZone, subPixZeroZone), criteria)


		prev_img = cur_img
		prev_time = cur_time

	# Stack arrays together along 4th axis
	descriptors = np.stack(tuple(descriptor_list), axis=3)

	# Save vel
	np.save(os.path.join(of_directory, name), descriptors)


if __name__ == "__main__":
	bag = str(sys.argv[1])
	name = str(sys.argv[2])
	gen_descriptors(bag, name)