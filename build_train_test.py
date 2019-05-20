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
import sklearn
from sklearn.model_selection import train_test_split


def build_train_test(data_list, output_name):
	# data_list = data_list.split()
	
	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	test_percentage = params_dict['test_percentage']
	descriptor_type = params_dict['descriptor_type']

	flatten_descriptor = params_dict['flatten_descriptor']

	grids = params_dict['grids']

	predict_vx = params_dict['predict_vx']
	predict_vy = params_dict['predict_vy']
	predict_vz = params_dict['predict_vz']
	predict_vthx = params_dict['predict_vthx']
	predict_vthy = params_dict['predict_vthy']
	predict_vthz = params_dict['predict_vthz']

	# TODO: Change these paths to fit your application
	of_path = "/home/toriw/tw_research/processed_bags/of/"
	v_path = "/home/toriw/tw_research/processed_bags/v/"
	x_test_path = "/home/toriw/tw_research/processed_bags/X_test"
	y_test_path = "/home/toriw/tw_research/processed_bags/Y_test"
	x_train_path = "/home/toriw/tw_research/processed_bags/X_train"
	y_train_path = "/home/toriw/tw_research/processed_bags/Y_train"

	#############################################



	# Loop through all names and load arrays into list
	x_list = []
	y_list = []
	
	for name in data_list:
		x_array = np.load(of_path + name + ".npy")
		y_array = np.load(v_path + name + ".npy")
		x_list.append(x_array)
		y_list.append(y_array)
	zero_list = []

	# Pre-allocate zero arrays for the flattened descriptors
	for descriptors in x_list:
		if descriptor_type == 'grid':
			descriptors_f = np.zeros((2*grids*grids, np.shape(descriptors)[3]))	

		elif descriptor_type == 1 or descriptor_type == 2 or descriptor_type == 3 or descriptor_type == 4:
			l = np.shape(descriptors)[0]*np.shape(descriptors)[2]
			descriptors_f = np.zeros((l, np.shape(descriptors)[3]))
		zero_list.append(descriptors_f)


	# Flatten descriptors and put in a list
	descriptor_list = []
	for i in range(len(zero_list)):
		unflattened_descriptors = x_list[i]
		for j in range(np.shape(unflattened_descriptors)[3]):
			zero_list[i][:, j] = np.reshape(unflattened_descriptors[:, :, :, j].flatten().T, (-1,))

	# Concatenated descriptor data
	all_descriptors = np.concatenate(tuple(zero_list), axis=1)

	# Concatenated velocity data
	all_velocities = np.concatenate(tuple(y_list), axis=0)

	# Take only parts corresponding to desired prediction target(s) (add False to start because first entry is time)
	predict_bools = [False, predict_vx, predict_vy, predict_vz, predict_vthx,\
		predict_vthy, predict_vthz]

	target_velocities = all_velocities[:, np.where(predict_bools)[0]]
	s = "Data set contains %d samples\n" %(np.shape(target_velocities)[0])
	print s
	# If test_percentage is 1, write all data to test folders
	if test_percentage == 1:
		X_test = all_descriptors.T
		Y_test = target_velocities
		np.save(os.path.join(x_test_path, output_name), X_test)
		np.save(os.path.join(y_test_path, output_name), Y_test)

	# If test_percentage is 0, write all data to training folders
	elif test_percentage == 0:
		X_train = all_descriptors.T
		Y_train = target_velocities

		np.save(os.path.join(x_train_path, output_name), X_train)
		np.save(os.path.join(y_train_path, output_name), Y_train)

	# Otherwise, split using sklearn 
	else:

		# Split the training and testing data
		X_train, X_test, Y_train, Y_test = train_test_split(all_descriptors.T, target_velocities, \
			test_size=test_percentage)

		train_test_split(all_descriptors.T, np.reshape(target_velocities, (-1)),test_size=test_percentage)

		# Write outputs to folders
		np.save(os.path.join(x_train_path, output_name), X_train)
		np.save(os.path.join(y_train_path, output_name), Y_train)
		np.save(os.path.join(x_test_path, output_name), X_test)
		np.save(os.path.join(y_test_path, output_name), Y_test)
 	
if __name__ == "__main__":
	data_list = sys.argv[1].split()
	output_name = sys.argv[2]
	build_train_test(data_list, output_name)