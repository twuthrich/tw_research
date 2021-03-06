This file serves as a documented example of how to take ros bag files, turn them into a dataset, and train and test a model using that dataset. Depending on your use case, you may want to split up the processing into seperate files, but this should provide an overview of how to leverage all the capabilities provided in this package. 

The following import statements are necessary: 

```
from camera_frame_mocap_to_velocity import *
from camera_to_of import *
from build_train_test import *
from compute_trajectory import *
from analyze_ttd import *
from analytical_estimate import *
import pdb
import numpy as np
from train_model import *
```

## Step 1: Define which bags are used for training and testing
If we have train_1.bag, train_2.bag, train_3.bag, test_1.bag, and test_2.bag, our lists would be as follows:

```
train_list = [train_1, train_2, train_3]
test_list = [test_1, test_2]
data_list = [train_1, train_2, train_3, test_1, test_2]
```

We save the name of the model we'll be training and testing
```
model_name = 'my_model'
train_list_name = 'my_training_set'
```

We save the path to the location of the ros bags
```
bag_path = "/home/my_name/ros_bags"
```

## Step 2: Process ros bags into optical flow and velocity
```
for i in range(len(all_list)):
 	prefix = all_list[i]

 	s = "Processing #%d out of %d \n" %(i+1, len(all_list))
 	print s

 	bag = bag_path + prefix + ".bag"

 	camera_frame_mocap_to_velocity(bag, prefix)
 	gen_descriptors(bag, prefix)
```



## Step 3: Build test sets
Assuming we want to test the trajectory on multiple, small datasets, the ones listed in test_list we use the following code. 
Before running this, make sure to update the test_percentage in params.yaml to 1. 

```
for i in range(len(test_list)):
	set_name = test_list[i]
	build_train_test([set_name], set_name)
```

## Step 4: Build a training set 
Do not forget to flip test_percentage back to 0. 

```
build_train_test(train_list, model_name)
```

## Step 5: Create a model
```
train_model(model_name, train_list_name)
```

## Step 6: Compute trajectory on each test segment
This code shows how to compute the estimated trajectory for a given dataset. 

```
for i in range(len(test_list)):
	item = test_list[i]
	s = "Computing trajectory on %d out of %d \n" %(i+1, len(test_list))
	print s
	compute_trajectory(model_name, item, None)
```

## Step 7: Compute the TTD error on each test section & save a text file
This code shows how to compute the total-traveled distance error between the estimated and actual trajectories, on each of the trajectories contained in test_list.  

```
trajectory_names = test_list
ttd_error_list = []
for i in range(len(trajectory_names)):
	ttd_error = get_ttd(trajectory_names[i])
	ttd_error_list.append(ttd_error)
```

This code saves a text file with the list of total traveled distance errors achieved on the set of test trajetories. 

```
txt_name = model_name + ".txt"
np.savetxt(txt_name, ttd_error_list, delimiter =',')
```

# Visualizations
## Velocity & Optical flow
Somtimes, we may want to check the velocity data output by a particular bag. Say we have a bag, sim_0_vert.bag, which we process using either as part of a batch (as described above) or as an individual bag (as shown here):

```
camera_frame_mocap_to_velocity("\home\toriw\Documents\rosbags\sim_0_vert.bag", "sim_0_vert") 

camera_to_of("\home\toriw\Documents\rosbags\sim_0_vert.bag", "sim_0_vert") 
```

We can now visualize the velocity in the form of a graph, and the optical flow in the form of a movie by making the following calls:
```
visualize_velocity("sim_0_vert")

visualize_of("\home\toriw\Documents\rosbags\sim_0_vert.bag", "sim_0_vert") 
```

Remember to update the parameters in params.yaml to reflect the image size and descriptor type before running the optical flow visualization. 

The analyze_ttd code produces plots of the actual vs estimated trajectory. 

# Other Notes
The polar descriptors, and their associated plotting functions have been designed specifically for a particular image size. If your image size varies from 368x640, and you wish to use polar descriptors, you will need to update the functions to relect new image dimensions. 

# Example Data
Example test data and a model are provided here. The model is named drone_sim, and the test set is named hallway_test. We can run the following to generate a plot of the estimated trajectory when using drone_sim model on the hallway_test dataset. 

```
compute_trajectory("drone_sim", "hallway_test", "None")
analyze_ttd("hallway_test")
```