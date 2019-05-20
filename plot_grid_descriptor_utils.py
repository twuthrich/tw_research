import pdb
import numpy as np
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib
import scipy
import yaml

def draw_descriptor_field(img, vx, vy, xedges, yedges, grid_spaces, points_0, points_1, dt, out):

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
	
	pdb.set_trace()


	###########################################

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
			cv2.line(img, (np.int(x), np.int(y)), (np.int(np.rint(x+vx_ij*10*dt)), np.int(np.rint(y+vy_ij*10*dt))), color_red, 2)

	# Plot grid lines
	for x in xedges:
		cv2.line(img, (np.int(x), np.int(yedges[0])), (np.int(x), np.int(yedges[-1])), color_black, 1)
	for y in yedges:
		cv2.line(img, (np.int(xedges[0]), np.int(y)), (np.int(xedges[-1]), np.int(y)), color_black, 1) 

	# Plot features
	if True:
		for p in range(np.shape(points_0)[0]):
			cv2.line(img, (points_0[p, 0, 0], points_0[p, 0, 1]), (points_1[p, 0, 0], points_1[p, 0, 1]), color_green, 1)
			cv2.circle(img, (points_0[p,0, 0], points_0[p,0, 1]), 1, color_blue, -1)

			cv2.circle(img, (points_1[p,0, 0], points_1[p,0, 1]), 1, color_white, -1)
	out.write(img)
	cv2.imshow('descriptor field', img)
	cv2.waitKey(0)
	return out


