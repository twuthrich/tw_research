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
from descriptor_plotters import *

def visualize_of(bag, name) :

	descriptors = np.load("/home/toriw/research/processed_bags/of/" + name + ".npy")

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']
	grids = params_dict['grids']
	fill_in_zeros = params_dict['fill_in_zeros']
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

	upperOnly = params_dict['upperOnly']

	# Subpixel accuracy parameters
	useSubPix = params_dict['useSubPix']

	subPixWinSize = params_dict['subPixWinSize']
	subPixZeroZone = params_dict['subPixZeroZone']
	subPixCount = params_dict['subPixCount']
	subPixEPS = params_dict['subPixEPS']

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, subPixCount, subPixEPS)

	# TODO: Update these to fit your application
	of_directory = "/home/me/tw_research/processed_bags/of"
	image_topic_name = "/imageraw"
	position_topic_name = "/mocap/pose"


	
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
	t0 = prev_time
	prev_img = CvBridge().imgmsg_to_cv2(mocap_dict[prev_time], "mono8")

	p0 = cv2.goodFeaturesToTrack(prev_img, **feature_params)

	if useSubPix:
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, subPixCount, subPixEPS)
			p0 = cv2.cornerSubPix(prev_img, p0, (subPixWinSize, subPixWinSize), (subPixZeroZone, subPixZeroZone), criteria)

	# To save a video
	# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (144,256))

	image_names = [str(num).zfill(4) for num in range(len(sorted_keys[1:]))]

	for i in range(len(sorted_keys[1:])):
		cur_time = sorted_keys[i+1]
		dt = cur_time - prev_time

		cur_img = CvBridge().imgmsg_to_cv2(mocap_dict[cur_time], "mono8")


		p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img, cur_img, p0, None, **lk_params)
		hi_error_indices = np.where(err > error_threshold)[0]
		lost_indices = np.where(st == 0)[0]
		del_indices = np.unique(np.concatenate((hi_error_indices, lost_indices)))

		# Take only p1 & corresponding p0 where tracker worked
		p1_filtered = np.delete(p1, del_indices, axis=0)
		p0_filtered = np.delete(p0, del_indices, axis=0)

		vx_d = descriptors[:, :, 0, i]
		vy_d = descriptors[:, :, 1, i]
		
		# Call plotting function based on which type of descriptor is being used
		if descriptor_type == 'grid':
			print("Elapsed Time: ", cur_time - t0)
			
			draw_descriptor_grid(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt, image_names[i])
		
		elif descriptor_type == 1:
			print("Elapsed Time: ", cur_time - t0)

			if upperOnly:
				draw_descriptor_1_upper(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt)
			elif sim:
				draw_descriptor_1_sim(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt, image_names[i])
			else:
				draw_descriptor_1(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt)
		
		elif descriptor_type == 2:
			draw_descriptor_2(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt)
		
		elif descriptor_type == 3:
			print("Elapsed Time: ", cur_time - t0)

			if upperOnly:
				draw_descriptor_3_upper(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt)	
			elif sim:
				draw_descriptor_3_sim(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt, image_names[i])	
			else:
				draw_descriptor_3(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt)

		elif descriptor_type == 4:
			draw_descriptor_4(cur_img, vx_d, vy_d, p0_filtered, p1_filtered, dt)

		# Reset image, time, and features for next iteration
		prev_img = cur_img
		prev_time = cur_time
		p0 = cv2.goodFeaturesToTrack(cur_img, **feature_params)

		if useSubPix:
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, subPixCount, subPixEPS)
			p0 = cv2.cornerSubPix(prev_img, p0, (subPixWinSize, subPixWinSize), (subPixZeroZone, subPixZeroZone), criteria)

	# For saving a video
	# out.release()

	cv2.destroyAllWindows()

if __name__ == "__main__":
	bag = str(sys.argv[1])
	name = str(sys.argv[2])
	visualize_of(bag, name)