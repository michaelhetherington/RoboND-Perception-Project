# Project: Perception Pick & Place

## Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

## Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Project Overview
In this project I had to create a perception pipeline that would enable a simulated PR2 Robot to pick and place a set of objects placed in front of it on a table. The steps were:

1. Create the perception pipeline
2. Determine which basket each object was to be placed in based on a pick list
3. Determine which arm (left or right) was most appropriate for picking and placing the object

The final code that contains my perception pipeline and classification is named [project_template.py](https://github.com/michaelhetherington/RoboND-Perception-Project/blob/master/pr2_robot/scripts/project_template.py).

My final yaml output files are:
  - [output_1.yaml](https://github.com/michaelhetherington/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output_1.yaml) with success rate of 100%
  - [output_2.yaml](https://github.com/michaelhetherington/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output_2.yaml) with success rate of 80%
  - [output_3.yaml](https://github.com/michaelhetherington/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output_3.yaml) with a success rate of 87.5%

The following image is an example of the objects being recognised by the PR2 Robot camera in RViz for Test World 1.

![demo-1](https://github.com/michaelhetherington/RoboND-Perception-Project/blob/master/writeup_images/recog_test_robot.png)

![demo-2](https://github.com/michaelhetherington/RoboND-Perception-Project/blob/master/writeup_images/recog_test_camera.png)

The following discussed the finer details of my code implementation.

### 1. Perception Pipeline
The structure and code for the perception pipeline followed Exercises 1 to 3 from the Udacity coursework.

#### Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting.
The steps for completing the filtering and RANSAC plane fitting, as outlined and commented throughout my code, are:
  1. Covert ROS msg to PCL data
  2. Statistical Outlier Filtering - to remove the noise in the scene
    - number of neighbouring points: __outlier_filter.set_mean_k(5)__
    - threshold scale factor: __x = 5__
  3. Voxel grid down sampling
    - Leaf size: __LEAF_SIZE = 0.01__
  4. Passthrough filter x3 for x-, y- and z-axes to narrow the target range down to the table top.
    - Z-axis: __axis_min = 0.60__ and __axis_max = 1.00__ 
    - Y-axis: __axis_min = -0.50__ and __axis_max = 0.50__
    - X-axis: __axis_min = 0.30__ and __axis_max = 0.85__
  5. RANSAC Plane Segmentation
    - Model type: __set_model_type(pcl.SACMODEL_PLANE)__
    - Max distance for a point to be in model: __max_distance = 0.01__
    
#### Complete Exercise 2 steps. Pipeline for clustering for segmentation.
The steps for completing the clustering for segmentation, as outlined and commented throughout my code, are:
  1. Extract inliers and outliers - determines which points are objects and which are the table
  2. Euclidean clustering
    - Make KD-tree
    - Cluster tolerance: __set_ClusterTolerance(0.05)__
    - Min cluster size: __set_MinClusterSize(50)__
    - Max cluster size: __set_MaxClusterSize(2000)__
  3. Create cluster-cloud masking to make each object cluster a homogenous yet different colour
  
#### Complete Exercise 3 steps. Train model data and classify objects in Test World.
The steps for completing the model training and cluster classification, as outlined and commented throughout my code, are:
  1. Capture the features of the objects that can be found in the scene. This was undertaken with helper code from Exercise 3.
  2. Train the classification model using the training_set data. Load the model data in to the code.
  3. Classify the clusters by first computing feature vectors and then using the trained classification model (previous step) to classify the objects
  4. Publish messages to ROS containing labels for the objects in the Test World scene. Must create a publishing service.

#### Create function to request PickPlace service
This function has the purpose of providing instructions to the PR2 Robot to pick objects in the Test World, which are included on the pick list, and placing them in the correct basket as specified in the pick list.
  1. Read in parameters from ROS - __object_list__ and __dropbox__
  2. Parse parameters in to individual variables using list comprehension
  3. Loop through list of detected objects and compute centroid of object (pick_pose)
  4. Determine which arm picks each object based on which basket it must be placed in (arm_name)
  5. Determine location where object must be placed (place_pose)
  6. Assign test world value to test scene variable (test_scene_num)
  7. Make yaml dictionaries to be appended to the list of yaml dictionaries (dict_list)
  8. Output yaml summary files
  9. Repeat for other Test Worlds

### Discussion

The key observations I made while undertaking this project were:
  - to effectively filter the point clouds for the objects it was necessary to have filters in all three orthogonal axes
  - RANSAC plane filtering doesn't appear to be a significant inclusion. I've included it as it was in the exercises but after experimenting I didn't find that changing the filtering parameters made a difference. This may be because my passthrough filters already excluded the table
  - list comprehension methods were an effective way to parse the test world parameters in to individual variables
  - I'd like to have found a way to extract the test world number from the launch file for automatic population of the test_scene_num variable
  - yaml dictionaries have specific data types they can accept and if there is a mismatch then the code fails
  - implementing the PickPlace service worked in that the robot was able to move to the locations of the objects but was not however always successful at gripping the object. In case where the object was successfully gripped then the robot was able to place the object in the correct basket
  - throughout this project I learned how to use the git functions through the Linux command line so that I could clone, commit and push changes directly without having to use Google Chrome or similar
  
Future improvements:
  - Within the Pose() data types it is possible to assign object orientation. This is a significant area for improvement as it would enable the robot end effectors to orientate more accurately to be able to pick the objects
  - The code currently only places 
  - Implement PR2 motion and collision mapping
  - Undertake the challenge



