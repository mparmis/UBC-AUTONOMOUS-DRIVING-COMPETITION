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
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from keras.models import model_from_json

import driving_functions as drv
from plate_transform_functions import get_raw_plate
from cnn_utils import convert_pic

import os

from std_msgs.msg import String
label_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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
    self.section = 3

    self.first_plate_publish_flag = 0

    self.gogogo = False

    self.sess = tf.Session()
    self.graph = tf.get_default_graph()
    set_session(self.sess)

    model_path = './cnn/model_saves/model_test_5'#leav off .[extension]
    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.loaded_model = model_from_json(loaded_model_json)
    self.loaded_model.load_weights(model_path+ ".h5")
    self.loaded_model._make_predict_function()
    
    print('model_loaded from disk')
    
    self.save_im_path = './full_stack/pics/'
    os.remove(file) for file in os.listdir(self.save_im_path) if file.endswith('.png')
    print('files cleared')

  def callback(self, data):
    
    #init vals: too tired to make more elegant
    team_ID = "Team14"
    team_password = "h8rdc03d"
    plate_location = '1' #^from above
    plate_ID = 'YY66' #from above


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
    elif(self.section is 4):
        vel_lin, vel_ang, flag, gogogo_flag = drv.section4_driving(cv_image, self.gogogo)
        self.gogogo = gogogo_flag
        print("gogoflag: " + str(gogogo_flag))

    else:
        pass

    vel = Twist()
    vel.angular.z = vel_ang
    vel.linear.x = vel_lin

    self.section = self.section + flag

    self.vel_pub.publish(vel)
    print('sec: ' + str(self.section))

    ## cnn
    all_high_conf_flag = False
    raw_plate = get_raw_plate(cv_image)
    if raw_plate is not None:
        print("found plate!")
        #pass 
        #do processing here
        ims_processed, sub_ims_raw = convert_pic(raw_plate)
        
        #print("SHAPE: " + str(ims_processed.shape))
        with self.graph.as_default():
          set_session(self.sess)
          y_predict = self.loaded_model.predict(ims_processed)
        
        y_val = []
        y_index = []
        all_high_conf_flag = True
        for i in range(y_predict.shape[0]):
          p_i = np.argmax(y_predict[i])
          y_val.append(y_predict[i][p_i])
          y_index.append(p_i)
          if (y_predict[i][p_i] < 0.7):
            all_high_conf_flag = False       
        plate_ID = label_options[y_index[0]] + label_options[y_index[1]] + label_options[y_index[2]] + label_options[y_index[3]]
        plate_location = label_options[y_index[4]]  
        if all_high_conf_flag:
          print("FOUND GOOD PLATE")

        print("plate: " + str(plate_ID))
        print("pos: "+ str(plate_location))
        print("yvals: " + str(y_val))
    if(self.first_plate_publish_flag == 0 ):
        publish_string = team_ID + ',' + team_password + ',' + '0' + ',' + 'XX99'
        self.first_plate_publish_flag = 1
        self.plate_pub.publish(publish_string)
    elif(all_high_conf_flag):
        publish_string = team_ID + ',' + team_password + ',' + plate_location + ',' + plate_ID
        self.plate_pub.publish(publish_string)
        print('published plate and stall!')
    print('  ')
    

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