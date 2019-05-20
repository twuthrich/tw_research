from camera_frame_mocap_to_velocity import *
from camera_to_of import *
from build_train_test import *
from compute_trajectory import *
import pdb
import numpy as np
import tf

# Read in a trajectory, and a gt trajectory, and calculate & return
# The %TTD Error

def get_ttd(data_set):

	# TODO: Update these paths to reflect your applications
	gt_traj_path = "/home/toriw/tw_research/processed_bags/gt_v_traj/"
	est_traj_path = "/home/toriw/tw_research/processed_bags/v_hat_traj/"

	gt_traj_x = np.load(gt_traj_path + data_set + "_x" + ".npy")
	gt_traj_y = np.load(gt_traj_path + data_set + "_y" + ".npy")
	gt_traj_z = np.load(gt_traj_path + data_set + "_z" + ".npy")

	est_traj_x = np.load(est_traj_path + data_set + "_x" + ".npy")
	est_traj_y = np.load(est_traj_path + data_set + "_y" + ".npy")
	est_traj_z = np.load(est_traj_path + data_set + "_z" + ".npy")

	gt_traj_x[0] = gt_traj_x[0][0]
	gt_traj_y[0] = gt_traj_y[0][0]
	gt_traj_z[0] = gt_traj_z[0][0]

	est_traj_x[0] = est_traj_x[0][0]
	est_traj_y[0] = est_traj_y[0][0]
	est_traj_z[0] = est_traj_z[0][0]

	# 3-D Trajectory Plotting
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.set_xlabel('X', size=15)
	ax.set_ylabel('Y', size=15)
	ax.set_zlabel('Z', size=15)

	plt.title("3-D Test Trajectory\n 4.1 %TTD Error ", size=20)

	fig.hold(True)
	plt.grid(True)


	# For lines
	ax.plot(est_traj_x, est_traj_y, est_traj_z, c="g")

	ax.plot(gt_traj_x, gt_traj_y, gt_traj_z, c="b")

	ax.scatter3D(gt_traj_x[0], gt_traj_y[0], gt_traj_z[0], c='orange', s=300)

	ax.scatter3D(est_traj_x[-1], est_traj_y[-1], est_traj_z[-1], c='g', s=300)
	
	ax.scatter3D(gt_traj_x[-1], gt_traj_y[-1], gt_traj_z[-1], c='b', s=300)

	plt.legend(['Estimated Trajectory', 'Ground Truth Trajectry', 'Starting Point',\
		'Estimated Endpoint', 'Actual Endpoint'], numpoints=1)

	plt.axis('equal')
	plt.show()


	# Determine GT Total Traveled Distance
	dx = np.reshape(np.diff(gt_traj_x), (-1,1))
	dy = np.reshape(np.diff(gt_traj_y), (-1,1))
	dz = np.reshape(np.diff(gt_traj_z) , (-1,1)) 

	dx_hat = np.reshape(np.diff(est_traj_x), (-1,1))
	dy_hat = np.reshape(np.diff(est_traj_y), (-1,1))
	dz_hat = np.reshape(np.diff(est_traj_z) , (-1,1)) 

	dx_sq = np.square(dx)
	dy_sq = np.square(dy)
	dz_sq = np.square(dz)

	dx_sq_hat = np.square(dx_hat)
	dy_sq_hat = np.square(dy_hat)
	dz_sq_hat = np.square(dz_hat)

	d_sq = np.sum(np.concatenate((dx_sq, dy_sq, dz_sq), axis=1), axis=1)
	ttd_gt = np.sum(np.power(d_sq, 0.5), axis=0)

	# Get vectors to actual and estimated endpoints
	start = np.array([est_traj_x[0], est_traj_y[0], est_traj_z[0]])
	end_est = np.array([est_traj_x[-1], est_traj_y[-1], est_traj_z[-1]])
	end_gt = np.array([gt_traj_x[-1], gt_traj_y[-1], gt_traj_z[-1]])

	delta_gt = end_gt - start
	delta_est = end_est - start

	ttd_error = np.linalg.norm(delta_gt - delta_est)/ttd_gt
	print(data_set, " : ", ttd_error*100)
	
	return ttd_error

if __name__ == "__main__":
	data_set = sys.argv[1]
	get_ttd(data_set)
	
