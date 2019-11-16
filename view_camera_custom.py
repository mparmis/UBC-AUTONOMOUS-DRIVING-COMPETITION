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

class image_converter:

  def __init__(self):
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    self.bridge = CvBridge()
    self.first_plot = True
    self.plot = None
    #self.ax1 = plt.subplot(1,2,1)


  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      print("got image!")
    except CvBridgeError as e:
      print(e)



    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
   
    mask_edge = cv2.inRange(gray_im, 240, 280)
    mask_road = cv2.inRange(gray_im, 78, 82.5)

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
    error = x_bar - tar
    print('error: ' + str(error)) 
  
    circled = cv2.circle(cv_image, (int(x_bar), int((719+500)/2)), 20, (0,255,0), -1)

    plot_image = circled  

    print(" ")

    #matplotlib image show
    if(self.first_plot):
        fg = plt.figure()
        ax = fg.gca()
        h = ax.imshow(plot_image)
        self.first_plot = False
        self.plot = h
    else:
        self.plot.set_data(plot_image)
        plt.draw(), plt.pause(1e-3)

    #cv2 image show
    #cv2.imshow('window', cv_image)



def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
    plt.close('all')
if __name__ == '__main__':
    main(sys.argv)