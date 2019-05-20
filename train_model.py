'''
This script trains and saves a model
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
from descriptor_constructors import *

def train_model(model_name, data_set):

	###### Read .yaml & set parameters #######

	with open("params.yaml", 'r') as stream:
		try:
			params_dict = yaml.load(open('params.yaml'))
		except yaml.YAMLError as exc:
			print(exc)

	model_type = params_dict['model_type']
	n_layers = params_dict['n_layers']
	scaler_type = params_dict['scaler_type']

	## TODO: Update these paths for you application
	x_train_path = "/home/toriw/tw_research/processed_bags/X_train/"
	y_train_path = "/home/toriw/tw_research/processed_bags/Y_train/"
	#############################################

	X_train = np.load(x_train_path + data_set + ".npy")
	Y_train = np.load(y_train_path + data_set + ".npy")

	# Scale
	if scaler_type == "std":
		scaler = preprocessing.StandardScaler().fit(X_train)
	elif scaler_type == "minmax":
		scaler = preprocessing.MinMaxScaler().fit(X_train)

	X_train_scaled = scaler.transform(X_train)

	# Create / Fit / Neural Net
	if model_type == "NN":
		scaled_model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(n_layers,))
		scaled_model.fit(X_train_scaled, Y_train)
	
	# Create / Fit Random Forest Model
	elif model_type == "RF":

		scaled_model = RandomForestRegressor()
		
		# make input 1-D if only predicting one feature
		if np.shape(Y_train)[1] == 1:
			Y_train = np.reshape(Y_train, (-1))
		scaled_model.fit(X_train_scaled, Y_train)


	# Save the model and the scaler
	with open('processed_bags/models/%s.pkl'%model_name, 'wb') as fid:
		cPickle.dump(scaled_model, fid)

	with open('processed_bags/scalers/%s.pkl'%model_name, 'wb') as fid:
		cPickle.dump(scaler, fid)

	print_str_1 = "Model Trained.\n"
	print_str_2 = "Model: \"/processed_bags/models/%s.pkl\"\n" %(model_name)
	print_str_3 = "Scaler: \"/processed_bags/scalers/%s.pkl\"\n" %(model_name)

	print(print_str_1)
	print(print_str_2)	
	print(print_str_3)	

if __name__ == "__main__":
	model_name = sys.argv[1]
	data_set = sys.argv[2]
	train_model(model_name, data_set)
