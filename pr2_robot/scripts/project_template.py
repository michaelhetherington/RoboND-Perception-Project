#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    # The environment used in this project contains noise
    # Using a statistical outlier filter we can remove this noise before
    # working on the remainder of the input
    # First, create the filter object
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighbouring points to analyse for any given point
    outlier_filter.set_mean_k(5) # experiment
    # Set threshold scale factor
    x = 0.5 # experiment
    # Any point with a mean distance larger than global
    # (mean distance + x*std_dev) will be considered an outlier
    outlier_filter.set_std_dev_mul_thresh(x) # with x = 1.0 an outlier must be more than one SD from mean
    # Call filter function
    cloud_filtered = outlier_filter.filter()
    
    # TODO: Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    # Select a leaf/voxel size - note that 1.0 is very large.
    # The leaf size is measured in meters. Therefore a size of 1 is a cubic meter.
    LEAF_SIZE = 0.01 # experiment with this
    # Set the voxel/leaf size. 
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    # TODO: PassThrough Filter
    # Create a passthrough filter object for the z-axis
    passthrough_z = cloud_filtered.make_passthrough_filter()
    # Assign axis and range of passthrough filter object.
    filter_axis = 'z'
    passthrough_z.set_filter_field_name(filter_axis)
    axis_min = 0.60 # experiment with this
    axis_max = 1.00 # experiment with this
    passthrough_z.set_filter_limits(axis_min, axis_max)
    # Filter the downsampled point cloud with the passthrough object to get resultant cloud.
    cloud_filtered = passthrough_z.filter()

    # Create a passthrough filter for the y-axis - across robot
    passthrough_y = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_y.set_filter_field_name(filter_axis)
    axis_min = -0.50 # experiment
    axis_max = 0.50 # experiment
    passthrough_y.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough_y.filter()

    # Create a passthrough filter for the x-axis - in front of robot
    passthrough_x = cloud_filtered.make_passthrough_filter()
    filter_axis = 'x'
    passthrough_x.set_filter_field_name(filter_axis)
    axis_min = 0.30 # experiment
    axis_max = 0.85 # experiment
    passthrough_x.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough_x.filter()

    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    # Set the model to fit the objects to
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Set the max distance for a point to be considered to be fitting the model
    max_distance = 0.01 # experiement with this
    seg.set_distance_threshold(max_distance)

    # TODO: Extract inliers and outliers
    # Call the segment function to obtain the set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()
    pcl_cloud_objects = cloud_filtered.extract(inliers, negative=True)
    pcl_cloud_table = cloud_filtered.extract(inliers, negative=False)

    # TODO: Euclidean Clustering
    # Convert XYZRGB point cloud to XYZ as Euclidean Clustering cannot use colour information
    white_cloud = XYZRGB_to_XYZ(pcl_cloud_objects)
    # Construct the k-d tree
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as a minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.05) #experiment
    ec.set_MinClusterSize(50) #experiment
    ec.set_MaxClusterSize(2000) #experiment
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered indices
    cluster_indices = ec.Extract()
    # cluster_indices now contains a list of indices for each cluster (a list of lists)

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # This cloud will contain points for each of the segmented objects, with each set of points having a unique colour.
    # Assign a colour corresponding to each segmented object in the camera view
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud object containing all clusters, each with a unique colour
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(pcl_cloud_objects)
    ros_cloud_table = pcl_to_ros(pcl_cloud_table)
    ros_cloud_clusters = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_cloud.publish(ros_cloud_clusters)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        sample_cloud = pcl_cloud_objects.extract(pts_list)
        sample_cloud = pcl_to_ros(sample_cloud)

        # Compute the associated feature vector
        chists = compute_color_histograms(sample_cloud, using_hsv=True)
        normals = get_normals(sample_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)        

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = sample_cloud
        detected_objects.append(do)

        rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    test_scene_num = Int32()
    arm_name = String()
    object_name = String()
    object_group = String()
    pick_pose = Pose()
    place_pose = Pose()
    
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    
    # TODO: Parse parameters into individual variables
    object_names = [object_list_param[i]['name'] for i in range(0, len(object_list_param))]
    object_groups = [object_list_param[i]['group'] for i in range(0, len(object_list_param))]
    dropbox_names = [dropbox_param[i]['name'] for i in range(0, len(dropbox_param))]
    dropbox_groups = [dropbox_param[i]['group'] for i in range(0, len(dropbox_param))]
    dropbox_positions = [dropbox_param[i]['position'] for i in range(0, len(dropbox_param))]

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list for object labels and centroids
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    for object in object_list:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])

    # TODO: ensure test_scene_num is correct for the scene being simulated
    test_scene_num.data  = int(1) # change this to suit the world being simulated

    # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
    dict_list = []
    for i in range(0, len(object_list_param)):
        # Populate various ROS messages
        object_name.data = str(labels[i])
        # TODO: Create 'pick_pose' for the object
        pick_pose.position.x = np.asscalar(centroids[i][0])
        pick_pose.position.y = np.asscalar(centroids[i][1])
        pick_pose.position.z = np.asscalar(centroids[i][2])
        
        # TODO: Assign the arm to be used for pick_place
        if object_groups[i] == 'red':
            arm_name.data = str('left')
        else:
            arm_name.data = str('right')
            
        # TODO: Create 'place_pose' for the object
        place_pose.position.x = dropbox_positions[dropbox_groups.index(object_groups[i])][0]
        place_pose.position.y = dropbox_positions[dropbox_groups.index(object_groups[i])][1]
        place_pose.position.z = dropbox_positions[dropbox_groups.index(object_groups[i])][2]
        
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict) 

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

#        try:
#            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
#            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

#            print ("Response: ",resp.success)

#        except rospy.ServiceException, e:
#            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('output_2.yaml', dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_cloud = rospy.Publisher("/pcl_clusters", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/obj_labels", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    # The model must be trained separately using the code from sensor_stick Exercise 3
    # Specifically the capture_features.py file must be altered to include the
    # relevant list of objects to be recognised. Then it must be run concurrently with the
    # roslaunch sensor_stick training.launch environment
    # The models will be loaded according to the new list as they are sourced in Gazebo.
    # Once the capture_features.py script has created the training_set it is time to
    # train the model using train_svm.py which is also taken from the sensor_stick/scripts folder
    # used in Exercise 3. train_svm.py will create the model.sav file which will be used by
    # the object recognition code
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
