#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

import driving_functions as drv
from plate_transform_functions import get_raw_plate

from std_msgs.msg import String
class image_converter:

  def __init__(self):
    self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=10)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    
    self.plate_pub = rospy.Publisher('/license_plate', String, queue_size=10)

    self.s3_last_error = 0
    
    #timing
    self.last_time = 0

    #plot vars
    self.first_plot = True
    self.plot = None

    #section int
    self.section = 1

    self.first_plate_publish_flag = 0

  def callback(self, data):
    
    #timing
    start_time = time.time()
    #print('elapsed_time: ' + str(start_time - self.last_time))
    self.last_time = start_time

    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    flag = 0
    vel_ang = 0
    vel_lin = 0

    ## driving:
    if(self.section is 1):
        vel_lin, vel_ang, flag, _ = drv.section1_driving(cv_image)

    elif(self.section is 2):
        vel_lin, vel_ang, flag, _ = drv.section2_driving(cv_image)
    
    elif(self.section is 3):
        vel_lin, vel_ang, flag, _, new_last_error = drv.section3_driving(cv_image, self.s3_last_error)
        self.s3_last_error = new_last_error
    else:
        pass

    vel = Twist()
    vel.angular.z = vel_ang
    vel.linear.x = vel_lin

    self.section = self.section + flag

    self.vel_pub.publish(vel)
    print('sec: ' + str(self.section))
    print('  ')
   
    ## cnn
    raw_plate = get_raw_plate(cv_image)
    if raw_plate is not None:
        print("found plate!")
        pass 
        #do processing here
    
        #plate is valid;


    team_ID = "Team14"
    team_password = "h8rdc03d"
    plate_location = '1' #^from above
    plate_ID = 'YY66' #from above

    if(self.first_plate_publish_flag == 0 ):
        publish_string = team_ID + ',' + team_password + ',' + '0' + ',' + 'XX99'
        self.first_plate_publish_flag = 1
        self.plate_pub.publish(publish_string)
    elif(0):
        publish_string = team_ID + ',' + team_password + ',' + plate_location + ',' + plate_ID
        self.plate_pub.publish(publish_string)



    

    ##image processing reference:
    # gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
    # mask_edge = cv2.inRange(gray_im, 240, 280)
    # mask_road = cv2.inRange(gray_im, 78, 82.5)
    # mask_crosswalk = cv2.inRange(cv_image, (0, 0, 240), (15, 15, 255))

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)