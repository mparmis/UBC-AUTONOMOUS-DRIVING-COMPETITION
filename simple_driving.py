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


class image_converter:

  def __init__(self):
    self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=10)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    
    self.s3_last_error = 0
    
    #timing
    self.last_time = 0

    #plot vars
    self.first_plot = True
    self.plot = None

    #section int
    self.section = 1

  def callback(self,data):
    
    #timing
    start_time = time.time()
    #print('elapsed_time: ' + str(start_time - self.last_time))
    self.last_time = start_time
    
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    
    ##image processing:
    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
   
    mask_edge = cv2.inRange(gray_im, 240, 280)
    mask_road = cv2.inRange(gray_im, 78, 82.5)
    print('sec: ' + str(self.section))

    plot_image = gray_im    

    vel = Twist()

    if(self.section is 1):
      #forward until line
      vel.angular.z = 0
      vel.linear.x = 0.001     
      s1_x = 600
      s1_y = 575
      s1_dy = 125
      s1_y_low = int(s1_y - (s1_dy/2))
      s1_y_high = int(s1_y + (s1_dy/2))

      circled = cv2.circle(cv_image, (int(s1_x), s1_y), 20, (0,255,0), -1)
      plot_image = circled
      val = np.any( np.transpose(mask_edge)[s1_x][s1_y_low:s1_y_high])
      print('vals' + str(np.transpose(mask_edge)[s1_x][s1_y_low:s1_y_high]))
      print('maskval: ' + str(val))
      if val:
        print('s1: edge found!')
        self.section = self.section+1
        vel.angular.z = 0
        vel.linear.x = 0

    elif(self.section is 2):
      vel.angular.z = 0.005
      vel.linear.x = 0   
      s2_x = 800
      s2_y = 500
      s2_dx = 150
      s2_x_low = int(s2_x - (s2_dx/2))
      s2_x_high = int(s2_x + (s2_dx/2))

      circled = cv2.circle(cv_image, (int(s2_x), s2_y), 20, (0,255,0), -1)
      plot_image = circled
      val = np.any(mask_edge[s2_y][s2_x_low:s2_x_high])
      print('vals' + str(mask_edge[s2_y][s2_x_low:s2_x_high]))
      print('maskval: ' + str(val))
      if val:
        print('s2: edge found!')
        self.section = self.section+1
        vel.angular.z = 0
        vel.linear.x = 0

    elif(self.section is 3):
        #follow_outside line
        s3_kp = 0.01 #0.01
        s3_kd = 0.01   #0.1

        submask_edge = np.transpose(np.transpose(mask_edge)[600:-1][:])

        top = 0
        bot = 0
        index_array = np.linspace(600, 600+submask_edge.shape[1]-1, submask_edge.shape[1] )
        #print('indexarray: ' + str(index_array))
        for r in range(550, 719): #range of rows to check
          top += np.sum(np.multiply(submask_edge[r], index_array))
          bot += np.sum(submask_edge[r])
        x_bar = top  / (bot +1)
        print('xbar: ' + str(x_bar))

        tar = 1100
        error = tar -  x_bar
        print('error: ' + str(error))
        ang_vel = s3_kp*(error) - s3_kd*(self.s3_last_error- error)
        self.s3_last_error = error

        if(abs(error) > 55):      
          vel.angular.z = ang_vel
          vel.linear.x = 0.00
        else:
          vel.linear.x = 0.0001
          #vel.angular.z = 0.001

    else:
      pass

    self.vel_pub.publish(vel)
    print('  ')

    # if (gray_im[-2,:] < 80).all():
    #   print("no line found")
    #   error = self.last_error

    #kmeans = KMeans(n_clusters=2)
    #m = mask_egde[200, :].reshape(-1,1)
    #kmeans.fit( mask_edge[200, :].reshape(-1,1))
    #print(kmeans.cluster_centers_)



    #image plot
    # if(self.first_plot):
    #     fg = plt.figure()
    #     ax = fg.gca()
    #     h = ax.imshow(plot_image, cmap='gray')
    #     self.first_plot = False
    #     self.plot = h
    # else:
    #     self.plot.set_data(plot_image)
    #     plt.draw(), plt.pause(1e-3)

    #circled = cv2.circle(gray_im,(int(x_bar), gray_im.shape[0]-im_row), 5, (0,255,0), -1)


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