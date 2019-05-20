#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
import os

def visualize_velocity(name):
	# TODO: change these paths to fit your application
	velocity_path = "/home/toriw/tw_research/processed_bags/v/"
	position_path = "/home/toriw/tw_research/processed_bags/raw_pose/"
	###

	v_in = np.load(velocity_path + name + ".npy")
	raw_pose_in = np.load(position_path + name + ".npy")

	(t_v, vx, vy, vz, ox, oy, oz) = (v_in[:, 0], v_in[:, 1], v_in[:, 2], v_in[:, 3], \
		v_in[:, 4], v_in[:, 5], v_in[:, 6])

	(t_p, x_r, y_r, z_r, thx_r, thy_r, thz_r) = (raw_pose_in[:, 0], raw_pose_in[:, 1], raw_pose_in[:, 2],\
		raw_pose_in[:, 3], raw_pose_in[:, 4], raw_pose_in[:, 5], raw_pose_in[:, 6])

	t_p = t_p - np.min(t_p);
	t_v = t_v - np.min(t_v);
	
	# Plot Results of Position Correction
	plt.rcParams['axes.grid'] = True
	fig, axarr = plt.subplots(2, 3)
	fig.suptitle('Velocity Data [m]')

	axarr[0, 0].plot(t_v, vx)
	axarr[0, 0].set_xlabel('Time [s]')
	axarr[0, 0].set_ylabel('X Velocity [m/s]')

	axarr[0, 1].plot(t_v, vy)
	axarr[0, 1].set_xlabel('Time [s]')
	axarr[0, 1].set_ylabel('Y Velocity [m/s]')

	axarr[0, 2].plot(t_v, vz)
	axarr[0, 2].set_xlabel('Time [s]')
	axarr[0, 2].set_ylabel('Z Velocity [m/s]')

	axarr[1, 0].plot(t_v, ox)
	axarr[1, 0].set_xlabel('Time [s]')
	axarr[1, 0].set_ylabel('Roll Velocity [rad/s]')

	axarr[1, 1].plot(t_v, oy)
	axarr[1, 1].set_xlabel('Time [s]')
	axarr[1, 1].set_ylabel('Pitch Velocity [rad/s]')

	axarr[1, 2].plot(t_v, oz)
	axarr[1, 2].set_xlabel('Time [s]')
	axarr[1, 2].set_ylabel('Yaw Velocity [rad/s]')

	plt.show()

	# Plot Results of Position Correction
	plt.rcParams['axes.grid'] = True
	fig, axarr = plt.subplots(2, 3)
	fig.suptitle('Position Data [m]')

	axarr[0, 0].plot(t_v, x_r)
	axarr[0, 0].set_xlabel('Time [s]')
	axarr[0, 0].set_ylabel('X Position [m]')

	axarr[0, 1].plot(t_v, y_r)
	axarr[0, 1].set_xlabel('Time [s]')
	axarr[0, 1].set_ylabel('Y Position [m]')

	axarr[0, 2].plot(t_v, z_r)
	axarr[0, 2].set_xlabel('Time [s]')
	axarr[0, 2].set_ylabel('Z Position [m]')

	axarr[1, 0].plot(t_v, thx_r)
	axarr[1, 0].set_xlabel('Time [s]')
	axarr[1, 0].set_ylabel('Roll Position [rad]')

	axarr[1, 1].plot(t_v, thy_r)
	axarr[1, 1].set_xlabel('Time [s]')
	axarr[1, 1].set_ylabel('Pitch Position [rad]')

	axarr[1, 2].plot(t_v, thz_r)
	axarr[1, 2].set_xlabel('Time [s]')
	axarr[1, 2].set_ylabel('Yaw Position [rad]')

	plt.show()


if __name__ == "__main__":
	name = str(sys.argv[1])
	visualize_velocity(name)