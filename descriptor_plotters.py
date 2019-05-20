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




def draw_descriptor_grid(img, vx, vy, points_0, points_1, dt, idx):

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	grid_statistic = params_dict['grid_statistic']
	image_h = params_dict['image_h']
	image_w = params_dict['image_w']
	grid_spaces = params_dict['grids']

	upperOnly = params_dict['upperOnly']
	write_video = params_dict['write_video']

	###########################################

	if upperOnly:
		image_h = image_h/2

	xedges = np.linspace(0, image_w, grid_spaces+1)
	yedges = np.linspace(0, image_h, grid_spaces+1)

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	# Take only points in upper half if desired
	if upperOnly:
		img = img[0:img.shape[0]/2]

	# Determine x ticks and y ticks
	# Flip y so that image and matrix coordinates will be aligned
	x_ticks = (xedges[1:] + xedges[:-1]) / 2
	y_ticks = np.fliplr(np.reshape(((yedges[1:] + yedges[:-1]) / 2), (-1, 1)))

	# Plotting params
	color_red = [0, 0, 255] # bgr colorspace
	color_blue = [255, 0, 0]
	color_black = [0, 0, 0]
	color_white = [255, 255, 255]
	color_green = [0, 255, 0]

	# Loop through vx and vy and plot lines
	for i in range(grid_spaces):
		for j in range(grid_spaces):
			vx_ij = vx[i,j]
			vy_ij = vy[i,j]
			x = x_ticks[i]
			y = y_ticks[j]
			n = 0.01*np.linalg.norm([vx, vy])
			cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx_ij*2*dt)), np.int(np.rint(y+vy_ij*2*dt))), color_red, 2)

	# Plot grid lines
	for x in xedges:
		cv2.line(img, (np.int(x), np.int(yedges[0])), (np.int(x), np.int(yedges[-1])), color_white, 1)
	for y in yedges:
		cv2.line(img, (np.int(xedges[0]), np.int(y)), (np.int(xedges[-1]), np.int(y)), color_white, 1) 

	# Plot features
	if True:
		for p in range(np.shape(points_0)[0]):
			cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), color_green, 1)
			cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 1, color_blue, -1)

			cv2.circle(img, (points_1[p,0, 0], points_1[p,0, 1]), 1, color_white, -1)
	
	cv2.waitKey(0)

def draw_descriptor_1(img, vx_d, vy_d, points_0, points_1, dt):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	r_circle = 90

	angles = np.linspace(0, 7*np.pi/4, 8)
	radii = r_circle*np.ones_like(angles)

	(x_inner, y_inner) = pol2cart(radii, angles) 
	x_inner += (image_w/2.0)*np.ones_like(x_inner)
	y_inner += (image_h/2.0)*np.ones_like(y_inner)

	x_outer = np.array([640, 504, 320, 163, 0, 136, 320, 504])
	y_outer = np.array([184, 368, 368, 368, 184, 0, 0, 0])

	x_ticks = np.array([520, 400, 240, 120, 120, 240, 400, 520])
	y_ticks = np.array([132, 62, 62, 132, 236, 306, 306, 236])

	# Plot section boundaries
	for i in range(len(x_inner)):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_outer[i]), np.int(y_outer[i])), [0, 0, 0], 2)
	
	cv2.circle(img, (image_w/2, image_h/2), r_circle, [0, 0, 0], 2)

	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x_ticks[p], y_ticks[p]), 2, [0, 0, 0], -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*50*dt)), np.int(np.rint(y+vy*50*dt))), [0, 0, 255], 2)
	# Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), [0, 255, 0], 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, [255, 0, 0], -1)

	# Draw circle of radius 90
	print("number of poings in view: ", np.shape(points_0)[0])
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)

def draw_descriptor_1_sim(img, vx_d, vy_d, points_0, points_1, dt, idx):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	r_circle = 90

	angles = np.linspace(0, 7*np.pi/4, 8)
	radii = r_circle*np.ones_like(angles)

	(x_inner, y_inner) = pol2cart(radii, angles) 
	x_inner += (image_w/2.0)*np.ones_like(x_inner)
	y_inner += (image_h/2.0)*np.ones_like(y_inner)

	x_outer = np.array([640, 560, 320, 80, 0, 80, 320, 560])
	y_outer = np.array([240, 480, 480, 480, 240, 0, 0, 0])

	x_ticks = np.array([560, 410, 230, 80, 80, 230, 420, 560])
	y_ticks = np.array([120, 100, 100, 120, 360, 410, 410, 360])

	# Plot section boundaries
	for i in range(len(x_inner)):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_outer[i]), np.int(y_outer[i])), [255, 255, 255], 1)
	
	cv2.circle(img, (image_w/2, image_h/2), r_circle, [255, 255, 255], 1)

	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x_ticks[p], y_ticks[p]), 2, [0, 0, 0], -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*2*dt)), np.int(np.rint(y+vy*2*dt))), [0, 0, 255], 2)
	# Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), [0, 255, 0], 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, [255, 0, 0], -1)


	# Draw circle of radius 90
	print("number of poings in view: ", np.shape(points_0)[0])
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)

def draw_descriptor_1_upper(img, vx_d, vy_d, points_0, points_1, dt):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']


	valid_indices = np.where(points_0[:, 0, 1] < image_h/2)
	points_0 = points_0[valid_indices[0], :, :]
	points_1 = points_1[valid_indices[0], :, :]

	image_h = image_h/2

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	img = img[0:img.shape[0]/2]

	r_circle = 45

	angles = np.linspace(0, 7*np.pi/4, 8)
	radii = r_circle*np.ones_like(angles)

	(x_inner, y_inner) = pol2cart(radii, angles) 
	x_inner += (image_w/2.0)*np.ones_like(x_inner)
	y_inner += (image_h/2.0)*np.ones_like(y_inner)

	x_outer = np.array([640, 412, 320, 228, 0, 228, 320, 412])
	y_outer = np.array([92, 184, 184, 184, 92, 0, 0, 0])

	x_ticks = np.array([480, 360, 280, 160, 160, 280, 360, 480])
	y_ticks = np.array([45, 30, 30, 45, 135, 150, 150, 135])

	# Plot section boundaries
	for i in range(len(x_inner)):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_outer[i]), np.int(y_outer[i])), [0, 0, 0], 2)
	
	cv2.circle(img, (image_w/2, image_h/2), r_circle, [0, 0, 0], 2)

	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x_ticks[p], y_ticks[p]), 2, [0, 0, 0], -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*50*dt)), np.int(np.rint(y+vy*50*dt))), [0, 0, 255], 2)
	# Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), [0, 255, 0], 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, [255, 0, 0], -1)

	# Draw circle of radius 90
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)

def draw_descriptor_2(img, vx_d, vy_d, points_0, points_1, dt):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	r_circle = 90

	angles = np.linspace(0, 7*np.pi/4, 8)
	radii = r_circle*np.ones_like(angles)

	(x_inner, y_inner) = pol2cart(radii, angles) 
	x_inner += (image_w/2.0)*np.ones_like(x_inner)
	y_inner += (image_h/2.0)*np.ones_like(y_inner)

	x_outer = np.array([640, 504, 320, 163, 0, 136, 320, 504])
	y_outer = np.array([184, 368, 368, 368, 184, 0, 0, 0])

	x_ticks = np.array([520, 400, 240, 120, 120, 240, 400, 520, np.int(image_w/2.0)])
	y_ticks = np.array([132, 62, 62, 132, 236, 306, 306, 236,  np.int(image_h/2.0)])

	# Plot section boundaries
	for i in range(len(x_inner)):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_outer[i]), np.int(y_outer[i])), [0, 0, 0], 2)
	
	cv2.circle(img, (image_w/2, image_h/2), r_circle, [0, 0, 0], 2)

	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x, y), 2, [0, 0, 0], -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*10*dt)), np.int(np.rint(y+vy*10*dt))), [0, 0, 255], 2)
		
	# Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), [0, 255, 0], 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, [255, 0, 0], -1)

	# Draw circle of radius 90
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)

def draw_descriptor_3(img, vx_d, vy_d, points_0, points_1, dt):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	# Calculate where to draw lines
	l_max = 184.0/np.sin(np.pi/4)
	r_max = np.sqrt((image_h/2.0)**2 + (image_w/2.0)**2)
	redges = np.append(np.linspace(0, l_max, 5), r_max)[1:]

	thetaedges = np.linspace(0, 7*np.pi/4, 8)
	(x_inner, y_inner) = pol2cart(redges[0]*np.ones_like(thetaedges), thetaedges) 
	(x_inner, y_inner) = pol2cart(redges[0]*np.ones_like(thetaedges), thetaedges) 

	x_inner += (image_w/2.0)*np.ones_like(x_inner)
	y_inner += (image_h/2.0)*np.ones_like(y_inner)

	x_outer = np.array([640, 504, 320, 136, 0, 136, 320, 504])
	y_outer = np.array([184, 368, 368, 368, 184, 0, 0, 0])

	# Plot section boundaries
	for i in range(len(x_inner)):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_outer[i]), np.int(y_outer[i])), [0, 0, 0], 2)
	for r in redges:
		cv2.circle(img, (np.int(image_w/2), np.int(image_h/2)), np.int(r), [0, 0, 0], 2)

	x_ticks = [407, 358, 284, 229, 229, 284, 358, 407, \
				466, 384, 260, 166, 166, 260, 384, 466, \
				529, 459, 184, 112, 112, 184, 459, 529, \
				594, 24, 24, 594]
	y_ticks = [149, 90, 90, 149, 219, 276, 276, 219,   \
				120, 35, 35, 120, 244, 332, 332, 244,  \
				103, 12, 12, 103, 269, 353, 353, 269,   \
				69, 69, 288, 288]



	####
	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x, y), 2, [0, 0, 0], -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*10*dt)), np.int(np.rint(y+vy*10*dt))), [0, 0, 255], 2)
		
	# Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), [0, 255, 0], 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, [255, 0, 0], -1)

	# Draw circle of radius 90
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)

def draw_descriptor_3_sim(img, vx_d, vy_d, points_0, points_1, dt, idx):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	# Calculate where to draw lines
	l_max = 240.0/np.sin(np.pi/4)
	r_max = np.sqrt((image_h/2.0)**2 + (image_w/2.0)**2)
	redges = np.append(np.linspace(0, l_max, 5), r_max)[1:]

	thetaedges = np.linspace(0, 7*np.pi/4, 8)
	(x_inner, y_inner) = pol2cart(redges[0]*np.ones_like(thetaedges), thetaedges) 
	(x_inner, y_inner) = pol2cart(redges[0]*np.ones_like(thetaedges), thetaedges) 

	x_inner += (image_w/2.0)*np.ones_like(x_inner)
	y_inner += (image_h/2.0)*np.ones_like(y_inner)

	x_outer = np.array([640, 560, 320, 80, 0, 80, 320, 560])
	y_outer = np.array([240, 480, 480, 480, 240, 0, 0, 0])

	# Plot section boundaries
	for i in range(len(x_inner)):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_outer[i]), np.int(y_outer[i])), [255, 255, 255], 1)
	for r in redges:
		cv2.circle(img, (np.int(image_w/2), np.int(image_h/2)), np.int(r), [255, 255, 255], 1)

	x_ticks = [430, 358, 284, 229, 229, 284, 358, 407, \
				466, 384, 260, 166, 166, 260, 384, 466, \
				529, 459, 184, 112, 112, 184, 459, 529, \
				594, 24, 24, 594]
	y_ticks = [150, 90, 90, 149, 219, 276, 276, 219,   \
				120, 35, 35, 120, 244, 332, 332, 244,  \
				103, 12, 12, 103, 269, 353, 353, 269,   \
				69, 69, 288, 288]

	x_ticks = [430, 370, 280, 210, 210, 280, 370, 430, \
				500, 390, 250, 140, 140, 250, 390, 500, \
				580, 500, 140, 60, 60, 140, 500, 580, \
				620, 20, 20, 620]
	y_ticks = [180, 120, 120, 180, 300, 360, 360, 300, \
				140, 50, 50, 140, 340, 430, 430, 340, \
				130, 20, 20, 130, 330, 460, 460, 330, \
				20, 20, 460, 460]


	####
	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x, y), 2, [0, 0, 0], -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*2*dt)), np.int(np.rint(y+vy*2*dt))), [0, 0, 255], 2)
		


		
	# Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), [0, 255, 0], 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, [255, 0, 0], -1)

	# Draw circle of radius 90
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)



def draw_descriptor_3_upper(img, vx_d, vy_d, points_0, points_1, dt):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']/2.0
	image_w = params_dict['image_w']

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	img = img[0:img.shape[0]/2]

	# Calculate where to draw lines
	l_max = 92.0/np.sin(np.pi/4)
	r_max = np.sqrt((image_h/2.0)**2 + (image_w/2.0)**2)
	redges = np.append(np.linspace(0, l_max, 5), r_max)[1:]

	thetaedges = np.linspace(0, 7*np.pi/4, 8)
	(x_inner, y_inner) = pol2cart(redges[0]*np.ones_like(thetaedges), thetaedges) 
	(x_inner, y_inner) = pol2cart(redges[0]*np.ones_like(thetaedges), thetaedges) 

	x_inner += (image_w/2.0)*np.ones_like(x_inner)
	y_inner += (image_h/2.0)*np.ones_like(y_inner)

	x_outer = np.array([640, 412, 320, 228, 0, 228, 320, 412])
	y_outer = np.array([92, 184, 184, 184, 92, 0, 0, 0])

	# Plot section boundaries
	for i in range(len(x_inner)):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_outer[i]), np.int(y_outer[i])), [0, 0, 0], 2)
	for r in redges:
		cv2.circle(img, (np.int(image_w/2), np.int(image_h/2)), np.int(r), [0, 0, 0], 2)

	x_ticks = [366, 338, 302, 274, 274, 302, 338, 366, \
		       396, 352, 292, 246, 246, 292, 352, 396, \
			   424, 386, 250, 210, 210, 250, 388, 424, \
			   530, 95, 95, 530]
	y_ticks = [74, 46, 46, 74, 108, 138, 138, 108,   \
			   60, 17, 17, 60, 125, 166, 166, 125,   \
			   50, 10, 10, 50, 130, 178, 178, 130, \
			   44, 44, 150, 150]



	####
	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x, y), 2, [0, 0, 0], -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*10*dt)), np.int(np.rint(y+vy*10*dt))), [0, 0, 255], 2)
		
	# Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), [0, 255, 0], 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, [255, 0, 0], -1)

	# Draw circle of radius 90
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)

def draw_descriptor_4(img, vx_d, vy_d, points_0, points_1, dt):
	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	image_h = params_dict['image_h']
	image_w = params_dict['image_w']

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	# Calculate where to draw lines
	l_max = 184.0/np.sin(np.pi/4)
	r_max = np.sqrt((image_h/2.0)**2 + (image_w/2.0)**2)
	redges = np.append(np.linspace(0, l_max, 5), r_max)[1:]

	thetaedges = np.linspace(0, 7*np.pi/4, 8)
	(x_inner, y_inner) = pol2cart(redges[0]*np.ones_like(thetaedges), thetaedges) 
	(x_inner, y_inner) = pol2cart(redges[0]*np.ones_like(thetaedges), thetaedges) 

	x_inner += (image_w/2.0)*np.ones_like(x_inner)
	y_inner += (image_h/2.0)*np.ones_like(y_inner)

	x_outer = np.array([640, 504, 320, 136, 0, 136, 320, 504])
	y_outer = np.array([184, 368, 368, 368, 184, 0, 0, 0])

	# Plot section boundaries
	for i in range(len(x_inner)):
		cv2.line(img, (np.int(x_inner[i]), np.int(y_inner[i])), (np.int(x_outer[i]), np.int(y_outer[i])), [0, 0, 0], 2)
	for r in redges:
		cv2.circle(img, (np.int(image_w/2), np.int(image_h/2)), np.int(r), [0, 0, 0], 2)
	
	
	x_ticks = [407, 358, 284, 229, \
				466, 384, 260, 166, \
				529, 459, 184, 112, \
				594, 24]

	y_ticks = [149, 90, 90, 149, \
				120, 35, 35, 120, \
				103, 12, 12, 103, \
				69, 69]


	####
	# Plot the descriptor lines
	for p in range(np.shape(x_ticks)[0]):
		x = x_ticks[p]
		y = y_ticks[p]
		vx = vx_d[p, 0]
		vy = vy_d[p, 0]
		cv2.circle(img, (x, y), 2, [0, 0, 0], -1)
		cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx*10*dt)), np.int(np.rint(y+vy*10*dt))), [0, 0, 255], 2)
		
	# Plot features & their track from previous frame
	for p in range(np.shape(points_0)[0]):
		cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), [0, 255, 0], 1)
		cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 2, [255, 0, 0], -1)

	# Draw circle of radius 90
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)

def cart2pol(x, y):
	r = np.reshape(np.sqrt(x**2 + y**2), (-1, 1))
	theta = np.arctan2(y, x)
	theta_pos = np.array([[t + 2*np.pi if t<0 else t for t in theta]]).T
	return(r, theta_pos)

def pol2cart(r, theta):
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	return (x, y)



	

