#!/usr/bin/env python

'''
- This script takes in the ros bag, and returns the camera velocity in the camera frame

- 7/23/18
Tori Wuthrich
'''

import pdb
import rosbag
import numpy as np
import tf
import sys
import yaml
import os
import tf
from mpl_toolkits.mplot3d import axes3d


def camera_frame_mocap_to_velocity(bag, name):

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)


	window_size = params_dict['window_size']
	sim = params_dict['sim']

	# TODO: Update these paths and topic names to fit your application
	raw_pose_directory = "/home/me/tw_research/processed_bags/raw_pose"
	v_directory = "/home/me/tw_research/processed_bags/v"
	image_topic_name = "/imageraw"
	position_topic_name = "/mocap/pose"

	###########################################

	mocap_times = []
	(x, y, z, roll, pitch, yaw) = ([], [], [], [], [], [])
	image_times = []
	z_ang_cmd = []

	# Loop through bag
	with rosbag.Bag(bag, 'r') as bag: 
		for topic, msg, t, in bag.read_messages():

			if topic == position_topic_name:

				# Get current position (convert from quat. ) & time
				o = msg.pose.orientation
				orientation = np.reshape(np.array(tf.transformations.\
				euler_from_quaternion((o.x, o.y, o.z, o.w))), (-1, 1))
				
				p = msg.pose.position
				
				# Append to column vector
				mocap_times.append(t.to_sec())

				roll.append(orientation[0, 0])
				pitch.append(orientation[1, 0])
				yaw.append(orientation[2, 0])

				x.append(p.x)
				y.append(p.y)
				z.append(p.z)

			# Keep track of times when images were recorded (image topic name may vary)
			if topic == image_topic_name:
				image_times.append(t.to_sec())
		
	# Eliminate Starting zeros & unwrap to eliminate jumps @ +/- 2pi
	roll = np.unwrap(np.array([roll]).T, axis=0)
	pitch = np.unwrap(np.array([pitch]).T, axis=0)
	yaw = np.unwrap(np.array([yaw]).T, axis=0)
	x = np.array([x]).T
	y = np.array([y]).T
	z = np.array([z]).T

	raw_pose = np.concatenate((x, y, z, roll, pitch, yaw), axis=1)

	# Diff poses & time to get velocities
	dt_mocap = np.array([np.diff(mocap_times, n=1, axis=0)]).T

	droll = np.diff(roll, n=1, axis=0)
	dpitch = np.diff(pitch, n=1, axis=0)
	dyaw = np.diff(yaw, n=1, axis=0)

	dx = np.diff(x, n=1, axis=0)
	dy = np.diff(y, n=1, axis=0)
	dz = np.diff(z, n=1, axis=0)

	# Raw finite difference
	raw_omega = np.concatenate((droll/dt_mocap, dpitch/dt_mocap, dyaw/dt_mocap), axis=1)
	raw_v = np.concatenate((dx/dt_mocap, dy/dt_mocap, dz/dt_mocap), axis=1)

	omega = np.zeros((3, len(image_times)-1))
	v = np.zeros((3, len(image_times)-1))

	image_times = np.array([image_times]).T
	mocap_times = np.array([mocap_times]).T

	# Loop through image times [1:]
	# Start at 1 so as to have the same size as the descriptors (descriptors look backwards)
	for i in range(np.shape(image_times)[0]-1):

		# Find time at current image + next image
		t_cur_image = image_times[i, 0]
		t_next_image = image_times[i+1, 0]

		# Calculate the midpoint between current and next image times
		t_mid_image = t_cur_image + (t_next_image - t_cur_image)/2.0

		# Find times on the edge of the desired window
		t_hi = min(np.max(mocap_times), t_mid_image + window_size/2.0)
		t_lo = max(t_hi - window_size, min(np.min(mocap_times), np.min(image_times)))

		# Find actual mocap times that correspond most closely to the window edges
		i_lo = np.argmin(np.abs(mocap_times - t_lo))
		i_hi = np.argmin(np.abs(mocap_times - t_hi))
		

		# Get selection of mocap times & angles
		mocap_points = mocap_times[i_lo:i_hi+1, 0]
		roll_points = roll[i_lo:i_hi+1, 0]
		pitch_points = pitch[i_lo:i_hi+1, 0]
		yaw_points = yaw[i_lo:i_hi+1, 0]
		x_points = x[i_lo:i_hi+1, 0]
		y_points = y[i_lo:i_hi+1, 0]
		z_points = z[i_lo:i_hi+1, 0]

		# Fit a polynomial of degree deg, then get the slope => velocity
		tx_vel = np.polyder(np.polyfit(mocap_points, roll_points, deg=1), m=1)
		ty_vel = np.polyder(np.polyfit(mocap_points, pitch_points, deg=1), m=1)
		tz_vel = np.polyder(np.polyfit(mocap_points, yaw_points, deg=1), m=1)

		x_vel = np.polyder(np.polyfit(mocap_points, x_points, deg=1), m=1)
		y_vel = np.polyder(np.polyfit(mocap_points, y_points, deg=1), m=1)
		z_vel = np.polyder(np.polyfit(mocap_points, z_points, deg=1), m=1)

		i_cur_image = np.argmin(np.abs(mocap_times - t_cur_image))

		# Get the RPY of the camera at the point most closely corresponding to the first of the two images
		roll_current = roll[i_cur_image]
		pitch_current = pitch[i_cur_image]
		yaw_current = yaw[i_cur_image]
		x_current = x[i_cur_image]
		y_current = y[i_cur_image]
		z_current = z[i_cur_image]

		# Form the rotation matrix between world and camera
		Rx = tf.transformations.rotation_matrix(roll_current, (1, 0, 0));
		Ry = tf.transformations.rotation_matrix(pitch_current, (0, 1, 0));
		Rz = tf.transformations.rotation_matrix(yaw_current, (0, 0, 1));


		if sim:
			Rxm90 = tf.transformations.rotation_matrix(-np.pi/2, (1, 0, 0));
			Rzm90 = tf.transformations.rotation_matrix(-np.pi/2, (0, 0, 1));

			c_R_b = Rxm90[0:3, 0:3].dot(Rzm90[0:3, 0:3])

			m_R_b = (Rz*Ry*Rx)[0:3, 0:3]
			b_R_m = m_R_b.transpose()

			c_R_m = c_R_b.dot(b_R_m)

		else:
			Rx90 = tf.transformations.rotation_matrix(np.pi/2, (1, 0, 0));
			Rzm90 = tf.transformations.rotation_matrix(-np.pi/2, (0, 0, 1));

			c_R_b = Rx90[0:3, 0:3].dot(Rzm90[0:3, 0:3])

		m_R_b = (Rz*Ry*Rx)[0:3, 0:3]
		b_R_m = m_R_b.transpose()

		c_R_m = c_R_b.dot(b_R_m)

		# Convert velocities into the camera frame using R_cw
		v_world = np.array([x_vel, y_vel, z_vel])
		omega_world = np.array([tx_vel, ty_vel, tz_vel])
		
		v_cam = c_R_m.dot(v_world)
		omega_cam = c_R_m.dot(omega_world)

		# Reshape and store
		v[:,i] = np.reshape(v_cam, (-1, 3))
		omega[:, i] = np.reshape(omega_cam, (-1, 3))
		
	# Change to column vector
	vel = np.concatenate((v, omega), axis=0)
	vel = vel.T	

	# Concatenate pose / vel with time for plotting / analysis
	vel = np.concatenate((image_times[:-1], vel), axis=1)
	raw_pose = np.concatenate((mocap_times, raw_pose), axis=1)

	# Save vel
	np.save(os.path.join(raw_pose_directory, name), raw_pose)
	np.save(os.path.join(v_directory, name), vel)
	
	# Print information
	print_str_0 = "\n"
	print_str_1 = "Velocity processing complete.\n\nProcessed Velocity: \"/processed_bags/v/%s.npy\"\n" %(name)
	print_str_2 = "Raw Pose: \"/processed_bags/raw_pose/%s.npy\"\n" %(name)
	print_str_4 = "Bag contains %d samples\n" %(np.shape(vel)[0])


if __name__ == "__main__":
	bag = str(sys.argv[1])
	name = str(sys.argv[2])
	camera_frame_mocap_to_velocity(bag, name)
