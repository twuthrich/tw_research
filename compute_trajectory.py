'''
This script plots various analysis metrics including 
trajectories resulting from propagating results
'''

import pdb
import numpy as np
import yaml
import sys
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import neural_network
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tf

# Function to calculate positions when given velocity and times
def propagate_v(dt, v, x_0):
	pose = [np.array([x_0])]
	for i in range(len(dt)):
		pose.append(x_0 + dt[i]*v[i])
		x_0 = pose[-1]
	return pose

# 
def compute_trajectory(model_name, data_set, rosbag):

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	model_type = params_dict['model_type']
	n_layers = params_dict['n_layers']
	scaler_type = params_dict['scaler_type']

	predict_vx = params_dict['predict_vx']
	predict_vy = params_dict['predict_vy']
	predict_vz = params_dict['predict_vz']
	predict_vthx = params_dict['predict_vthx']
	predict_vthy = params_dict['predict_vthy']
	predict_vthz = params_dict['predict_vthz']

	## TODO: Change these paths for your application
	v_path = "/home/toriw/research/processed_bags/v/"
	pose_path = "/home/toriw/tw_research/processed_bags/raw_pose/"
	x_test_path = "/home/toriw/tw_research/processed_bags/X_test/"
	gt_traj_path = "/home/toriw/tw_research/processed_bags/gt_v_traj/"
	est_traj_path = "/home/toriw/tw_research/processed_bags/v_hat_traj/"
	est_v_path = "/home/toriw/tw_research/processed_bags/v_hat/"
	#############################################
	predict_bools = [False, predict_vx, predict_vy, predict_vz, predict_vthx,\
		predict_vthy, predict_vthz]

	
	# Load the model
	with open('processed_bags/models/%s.pkl'%model_name, 'rb') as fid:
		model = cPickle.load(fid)

	# Load data used to create model
	with open('processed_bags/scalers/%s.pkl'%model_name, 'rb') as fid:
		scaler = cPickle.load(fid)
	

	v_filtered = np.load(v_path + data_set + ".npy")
	aruco_poses = np.load(pose_path + data_set + ".npy")
	descriptors = np.load(x_test_path + data_set + ".npy")

	x_gt = aruco_poses[:, 1]
	y_gt = aruco_poses[:, 2]
	z_gt = aruco_poses[:, 3]

	temp = np.copy(aruco_poses)

	# Extract Times
	t_image = v_filtered[:,0]
	t_aruco = aruco_poses[:,0]

	dt_image = np.diff(t_image)

	# Extract gt velocities in directions predicted
	v_filtered_target = np.reshape(v_filtered[:, np.where(predict_bools)[0]], (-1, np.sum(predict_bools)))

	# Extract other velocities
	vx = np.reshape(v_filtered[:, 1], (-1)) 
	vy = np.reshape(v_filtered[:, 2], (-1)) 
	vz = np.reshape(v_filtered[:, 3], (-1)) 
	vthx = np.reshape(v_filtered[:, 4], (-1))
	vthy = np.reshape(v_filtered[:, 5], (-1))
	vthz = np.reshape(v_filtered[:, 6], (-1))

	# Use the model to predict velocity
	X_test_scaled = scaler.transform(descriptors)	
	v_hat = model.predict(X_test_scaled)

	##### camera to mocap (for converting velocity)
	Rxm90 = tf.transformations.rotation_matrix(-np.pi/2, (1, 0, 0));
	Rz90 = tf.transformations.rotation_matrix(np.pi/2, (0, 0, 1));

	R = Rz90[0:3, 0:3].dot(Rxm90[0:3, 0:3])

	### mocap to camera (starting position)
	Rx90 = tf.transformations.rotation_matrix(np.pi/2, (1, 0, 0));
	Rzm90 = tf.transformations.rotation_matrix(-np.pi/2, (0, 0, 1));

	R2 = Rx90[0:3, 0:3].dot(Rzm90[0:3, 0:3])

	v_filtered_target = R.dot(np.stack((vx, vy, vz)))
	v_filtered_target = v_filtered_target.T

	v_hat = R.dot(v_hat.T)
	v_hat = v_hat.T

	x_0 = aruco_poses[np.argmin(np.abs(t_aruco - t_image[0])), 0+1]
	y_0 = aruco_poses[np.argmin(np.abs(t_aruco - t_image[0])), 1+1]
	z_0 = aruco_poses[np.argmin(np.abs(t_aruco - t_image[0])), 2+1]

	# Propagate GT and estimated velocities
	x_0 = aruco_poses[np.argmin(np.abs(t_aruco - t_image[0])), 0+1]
	x_hat = propagate_v(dt_image, v_hat[:,0], x_0)
	x_prop_gt = propagate_v(dt_image, v_filtered_target[:,0], x_0)

	y_0 = aruco_poses[np.argmin(np.abs(t_aruco - t_image[0])), 1+1]
	y_hat = propagate_v(dt_image, v_hat[:,1], y_0)
	y_prop_gt = propagate_v(dt_image, v_filtered_target[:,1], y_0)

	z_0 = aruco_poses[np.argmin(np.abs(t_aruco - t_image[0])), 2+1]
	z_hat = propagate_v(dt_image, v_hat[:,2], z_0)
	z_prop_gt = propagate_v(dt_image, v_filtered_target[:,2], z_0)


	x_name = data_set + "_x"
	y_name = data_set + "_y"
	z_name = data_set + "_z"

	name = model_name + "_" + data_set

	# Save gt propagated trajectories
	np.save(os.path.join(gt_traj_path, x_name), x_prop_gt)
	np.save(os.path.join(gt_traj_path, y_name), y_prop_gt)
	np.save(os.path.join(gt_traj_path, z_name), z_prop_gt)

	# Save estimated propagated trajectories
	np.save(os.path.join(est_traj_path, x_name), x_hat)
	np.save(os.path.join(est_traj_path, y_name), y_hat)
	np.save(os.path.join(est_traj_path, z_name), z_hat)

	# Save estimated velocities
	np.save(os.path.join(est_v_path, name), v_hat)

	print("Saved new trajectories")
if __name__ == "__main__":
	model_name = sys.argv[1]
	data_set = sys.argv[2]
	rosbag = sys.argv[3]
	compute_trajectory(model_name, data_set, rosbag)