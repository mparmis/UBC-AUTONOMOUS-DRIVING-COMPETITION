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
    self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=2)#was 10
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    
    self.plate_pub = rospy.Publisher('/license_plate', String, queue_size=10)

    self.s3_last_xbar = 0
    self.s3_last_error = 0
    self.s7_last_error = 0
    self.s7_last_xbar = 0
    #timing
    self.last_time = 0
    self.last_time_ROSPI = 0

    #plot vars
    self.first_plot = True
    self.plot = None

    #section int
    self.section = 3#1

    self.first_plate_publish_flag = 0

    self.s3_cycles = 0
    self.crosswalks_passed = 2#0
    self.ICS_seen_intersection = False
    self.turn_enough_to_inner = False

    self.found_plate_flag = False

    self.gogogo = False
    self.seen_ped = False
    self.passed_second_blue_line = False

    self.seen_sec6_truck  = False
    self.sec6_gogogo = False
    self.sec6_gogogo_straight = True


    self.sec3n_seen_car_flag =  False
    self.sec3n_seeing_car_flag = False

    self.dict_plate_vals = {}


    self.sess = tf.Session()
    self.graph = tf.get_default_graph()
    set_session(self.sess)

    #number model
    num_model_path = './cnn/model_saves_num/model_test_5'#leav off .[extension]
    json_file = open(num_model_path + '.json', 'r')
    num_loaded_model_json = json_file.read()
    json_file.close()
    self.num_loaded_model = model_from_json(num_loaded_model_json)
    self.num_loaded_model.load_weights(num_model_path+ ".h5")
    self.num_loaded_model._make_predict_function()
    print('numbers model_loaded from disk')
    
    #letters model
    letter_model_path = './cnn/model_saves/model_test_5'#leav off .[extension]
    json_file = open(letter_model_path + '.json', 'r')
    letter_loaded_model_json = json_file.read()
    json_file.close()
    self.letter_loaded_model = model_from_json(letter_loaded_model_json)
    self.letter_loaded_model.load_weights(letter_model_path+ ".h5")
    self.letter_loaded_model._make_predict_function()
    print('letters model_loaded from disk')
    
    #position
    pos_model_path = './cnn/model_saves_pos/model_test_5'#leav off .[extension]
    json_file = open(pos_model_path + '.json', 'r')
    pos_loaded_model_json = json_file.read()
    json_file.close()
    self.pos_loaded_model = model_from_json(pos_loaded_model_json)
    self.pos_loaded_model.load_weights(pos_model_path+ ".h5")
    self.pos_loaded_model._make_predict_function()
    print('letters model_loaded from disk')


    self.save_i = 0
    self.save_im_path = './full_stack/pics/'
    
    filelist = [ f for f in os.listdir( self.save_im_path) if f.endswith(".png") ]
    for f in filelist:
      os.remove(os.path.join(self.save_im_path, f))
    print('files cleared')

  def callback(self, data):
    x_1=rospy.get_time()
    #init vals: too tired to make more elegant
    team_ID = "Team11"
    team_password = "h8rdc03d"
    plate_location = '1' #^from above
    plate_ID = 'YY66' #from above

    #timing
    start_time = time.time()
    print('elapsed_time:  PY' + str(start_time - self.last_time))
    self.last_time = start_time
    start_time_ros = rospy.get_time()
    print('elapsed_time:  ROS' + str(start_time - self.last_time))
    self.last_time_ROSPI = start_time_ros


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
        vel_lin, vel_ang, flag, _, new_last_error = drv.section3_driving(self, cv_image, self.s3_last_error)
        self.s3_last_error = new_last_error
    elif(self.section is 4):
        vel_lin, vel_ang = drv.section4_driving(self, cv_image) 
        print("gogoflag: " + str(self.gogogo))
    elif(self.section is 5):
        #vel_lin, vel_ang = drv.section5(self, cv_image)
        vel_lin, vel_ang = drv.new_section_5_internal_line_following(self, cv_image)
    elif(self.section is 6):
        pass
        #vel_lin, vel_ang = drv.section6(self, cv_image)
    elif(self.section is 7):
        vel_lin, vel_ang = drv.section7(self, cv_image)
    else:
        pass

    vel = Twist()
    vel.angular.z = vel_ang
    vel.linear.x = vel_lin


    print('velL: ' + str(vel_lin) + " velA: " + str(vel_ang))
    self.section = self.section + flag

    self.vel_pub.publish(vel)
    print('sec: ' + str(self.section))

    ## cnn
    all_high_conf_flag = False
    raw_plate = get_raw_plate(cv_image)
    if raw_plate is not None:
        
        # try:
        #   cv2.imwrite(self.save_im_path + str(self.save_i) + '.png', raw_plate)
        #   self.save_i = self.save_i + 1
        # except:
        #   print("failed to save plate")
        print("found plate!")
        #pass 
        #do processing here
        ims_processed, sub_ims_raw = convert_pic(raw_plate)
        
        #print("SHAPE: " + str(ims_processed.shape))
        #numbers
        with self.graph.as_default():
          set_session(self.sess)
          y_predict_nums = self.num_loaded_model.predict(ims_processed)
        
        #letters
        with self.graph.as_default():
          set_session(self.sess)
          y_predict_letters = self.letter_loaded_model.predict(ims_processed)

        #pos
        with self.graph.as_default():
          set_session(self.sess)
          y_predict_pos = self.pos_loaded_model.predict(ims_processed)


        y_val = []
        y_index = []
        all_high_conf_flag = True
        plate_string = ""

        for i in range(y_predict_letters.shape[0]):
          if i <= 1 :
            p_i = np.argmax(y_predict_letters[i])
            y_val.append(y_predict_letters[i][p_i])
            y_index.append(p_i)
            plate_string = plate_string + label_options[p_i]
          elif i is 4:
            p_i = np.argmax(y_predict_pos[i])
            y_val.append(y_predict_pos[i][p_i])
            y_index.append(p_i)
            plate_string = plate_string + label_options[p_i]
          else: 
            p_i = np.argmax(y_predict_nums[i])
            y_val.append(y_predict_nums[i][p_i])
            y_index.append(p_i)
            plate_string = plate_string + label_options[p_i]

        lowest_conf = 1
        for val in y_val:
          if val < lowest_conf:
            lowest_conf = val
          if (val < 0.85): # was 99 then 95
            all_high_conf_flag = False       
        
        #plate_ID = label_options[y_index[0]] + label_options[y_index[1]] + label_options[y_index[2]] + label_options[y_index[3]]
        #plate_location = label_options[y_index[4]]  
        if all_high_conf_flag:
          #print("FOUND GOOD PLATE")
          if self.crosswalks_passed >=2 and plate_string[4] is not 1:
            self.found_plate_flag=True
        #print("plate: " + str(plate_string))
        #print("pos: "+ str(plate_location))
        #print("yvals: " + str(y_val))
    if(self.first_plate_publish_flag == 0 ):
        publish_string = team_ID + ',' + team_password + ',' + '0' + ',' + 'ZZ99'
        self.first_plate_publish_flag = 1
        self.plate_pub.publish(publish_string)
    #elif(all_high_conf_flag):
    elif(raw_plate is not None):
        plate_location = plate_string[4]
        plate_ID = plate_string[0:4]
        publish_string = team_ID + ',' + team_password + ',' + plate_location + ',' + plate_ID
        if (y_val[4] > 0.965): #plate location confidence check
          if plate_location not in self.dict_plate_vals:
            self.dict_plate_vals[plate_location] = lowest_conf
            self.plate_pub.publish(publish_string)
            print("PUBLISHED")
            print('first: published plate and stall!: ' + str(publish_string))
            print("yvals: " + str(y_val))
          else:
            curr_conf = self.dict_plate_vals[plate_location]
            if curr_conf < lowest_conf:
              self.dict_plate_vals[plate_location] = lowest_conf
              self.plate_pub.publish(publish_string)
              print("PUBLISHED")
              print('second: published plate and stall!: ' + str(publish_string))
              print("yvals: " + str(y_val))
        # self.plate_pub.publish(publish_string)
        # print('published plate and stall!: ' + str(publish_string))
    x_2=rospy.get_time()
    print("TIME: " + str(x_2 - x_1)) 
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