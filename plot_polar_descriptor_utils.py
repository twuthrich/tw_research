import pdb
import rosbag
import numpy as np
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import yaml

def calc_polar_edges():

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

	#######
	redges = np.linspace(0, image_h/2, n_circles+1)

	# radius in outer section can be up to the max possible given image frame size
	r_max_image = np.array([np.linalg.norm([image_h/2.0, image_w/2.0])])
	redges = np.concatenate((redges, r_max_image))
	thetaedges = np.linspace(0, 2*np.pi, n_lines+1)

	# Create array of zeros for descriptor
	l_d = (n_circles+1)*n_lines
	return (redges, thetaedges)

# For visualization - this function calculates the points on the edge of the image
# in order to plot the lines from the center outwards
def calc_edge_points(thetaedges):

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	#######

	x_edge = np.zeros((np.shape(thetaedges)[0]-1, 1))
	y_edge = np.zeros((np.shape(thetaedges)[0]-1, 1))

	for i in range(np.shape(thetaedges)[0]-1):
		theta = thetaedges[i]

		if fix_y(theta, image_h, image_w):
			# Scale by y to find point on perimeter
			y = np.sign(np.sin(theta))*image_h/2
			x = y*np.cos(theta)/np.sin(theta)
		
		else:
			# Scale by x to find point on perimeter
			x = np.sign(np.cos(theta))*image_w/2
			y = x*np.sin(theta)/np.cos(theta)
		x_edge[i] = x
		y_edge[i] = y

	x_edge += np.ones_like(x_edge)*image_w/2
	y_edge += np.ones_like(y_edge)*image_h/2
	
	return (x_edge, y_edge)

def calc_inner_points(r, thetaedges):

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']
	partition_center = params_dict['partition_center']

	#######
	# If center is partitioned, just use the origin
	if partition_center:
		x_inner = image_w/2*np.ones((np.shape(thetaedges)[0]-1, 1))
		y_inner = image_h/2*np.ones((np.shape(thetaedges)[0]-1, 1))
	# Otherwise, need points on innermost circle
	else:
		x_inner = r*np.cos(thetaedges)[:-1]
		y_inner = r*np.sin(thetaedges)[:-1]

		x_inner += image_w/2*np.ones_like(x_inner)
		y_inner += image_h/2*np.ones_like(y_inner)

	return(x_inner, y_inner)

	
	# For each lines, 

# This function calculates the (x,y) points at which to plot the descriptors
# for the visualization for the circular case
def calc_cicular_ticks(redges, thetaedges):

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
	partition_center = params_dict['partition_center']

	#######
	# Find the radial & angular midpoints
	r_points = np.diff(redges)[:-1]/2.0 + redges[:-2]
	theta_points = thetaedges[:-1] + np.diff(thetaedges)[0]/2.0

	# Define arrays for plotting points
	if not partition_center:
		l_d = (n_circles-1)*n_lines + 1		
	else:
		l_d = (n_circles)*n_lines

	(r_plot, theta_plot) = (np.zeros((l_d, 1)), np.zeros((l_d, 1)))

	# Arrange plotting points in proper order according to descriptor
	ctr = 0
	for i in range(np.shape(r_points)[0]):
		# If not partitioning center, just plot one point at the middle
		if (not partition_center) and i == 0:
			r_plot[ctr, 0] = 0
			theta_plot[ctr, 0] = 0
			ctr += 1
		else:
			for j in range(np.shape(theta_points)[0]):
				r_plot[ctr, 0] = r_points[i]
				theta_plot[ctr, 0] = theta_points[j]
				ctr += 1

	# Convert points to cartesian image frame
	(x_plot, y_plot) = pol2cart(r_plot, theta_plot)
	x_plot = np.reshape(x_plot + image_w/2, (-1, 1))
	y_plot = np.reshape(y_plot + image_h/2, (-1, 1))

	# Manually add points for the outermost sections when there are 8 pie slices
	if np.shape(theta_points)[0] == 8 and image_h == 480:
		# outer points in frame with origin @ image center 
		x_outer = np.array([[280, 170, -170, -280, -280, -170, 170, 280]]).T
		y_outer = np.array([[120, 205, 205, 120, -120, -205, -205, -120]]).T

	elif np.shape(theta_points)[0] == 4 and image_h == 480:
		# outer points in frame with origin @ image center 
		x_outer = np.array([[280, -280, -280, 280]]).T
		y_outer = np.array([[160, 160, -160, -160]]).T

	elif np.shape(theta_points)[0] == 4 and image_h == 368:
		# outer points in frame with origin @ image center 
		x_outer = np.array([[252, -252, -252, 252]]).T
		y_outer = np.array([[92, 92, -92, -92]]).T

	elif np.shape(theta_points)[0] == 8 and image_h == 368:
		# outer points in frame with origin @ image center 
		x_outer = np.array([[252, 130, -130, -252, -252, -130, 130, 252]]).T
		y_outer = np.array([[92, 157, 157, 92, -92, -157, -157, -92]]).T


	else:
		raise Exception("Descriptor must contain either 4 or 8 lines")

	# Convert to traditional image coords. 
	x_outer += image_w/2*np.ones_like(x_outer)
	y_outer += image_h/2*np.ones_like(y_outer)

	# Concatenate all plotting points
	x_plot = np.concatenate((x_plot, x_outer), axis=0)
	y_plot = np.concatenate((y_plot, y_outer), axis=0)

	# Convert every element to an int for use in openCv
	x_plot = np.array([[np.int(x) for x in x_plot]]).T
	y_plot = np.array([[np.int(y) for y in y_plot]]).T
	return (x_plot, y_plot)

def draw_descriptor_field_circular(img, vx_d, vy_d, redges, thetaedges, points_0, points_1, dt):

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
	partition_center = params_dict['partition_center']

	#######

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	# for each angle, find the corresponding point on the edge of the rectangle
	color_black = [0, 0, 0]
	color_red = [0, 0, 255] # bgr colorspace
	color_green = [0, 255, 0]
	color_blue = [255, 0, 0]
	
	# Calculate the points that the lines will pass through
	(x_edge, y_edge) = calc_edge_points(thetaedges)
	(x_inner, y_inner) = calc_inner_points(redges[1], thetaedges)
	
	# Plot the descriptor boundaries of lines and circles
	for i in range(np.shape(x_edge)[0]):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_edge[i]), np.int(y_edge[i])), color_black, 2)

	for i in range(np.shape(redges[1:])[0]):
		cv2.circle(img, (image_w/2, image_h/2), np.int(redges[i+1]), color_black, 2)

	# Calculate the coodinates at which to plot the descriptors
	(x_ticks, y_ticks) = calc_cicular_ticks(redges, thetaedges)

	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x_ticks[p], y_ticks[p]), 2, color_black, -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*10*dt)), np.int(np.rint(y+vy*10*dt))), color_red, 2)

    # Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), color_green, 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, color_blue, -1)
		# cv2.circle(img, (points_1[p,0, 0], points_1[p,0, 1]), 2, color_black, -1)

	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)

def cart2pol(x, y):
	r = np.reshape(np.sqrt(x**2 + y**2), (-1, 1))
	theta = np.arctan2(y, x)
	theta_pos = np.array([[t + 2*np.pi if t<0 else t for t in theta]]).T
	return(r, theta_pos)

# This function takes in an angle, and returns a 1 if the line should
# be scaled by x, and a 0 if it should be scaled by y
def fix_y(theta, image_h, image_w):
	# For this test need angle between [-pi, pi]
	if np.abs(theta) > np.pi:
		theta -= 2*np.pi

	phi = np.arctan2(image_h/2, image_w/2)
	if (np.abs(theta) >= phi and np.abs(theta) <= np.pi - phi):
		return True
	return False

# This function converts radial coordinates to cartesian
def pol2cart(r, theta):
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	return (x, y)