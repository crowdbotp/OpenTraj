import sys
packages_path_from_ros = '/opt/ros/kinetic/lib/python2.7/dist-packages'
ros_is_installed = (packages_path_from_ros in sys.path)
if ros_is_installed:
    sys.path.remove(packages_path_from_ros) # in order to import cv2 under python3
import cv2
print('cv2 is imported!')
if ros_is_installed:
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
