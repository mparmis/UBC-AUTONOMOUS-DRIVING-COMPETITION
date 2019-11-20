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

    #matplotlib image show
    if(self.first_plot):
        fg = plt.figure()
        ax = fg.gca()
        h = ax.imshow(cv_image)
        self.first_plot = False
        self.plot = h
    else:
        self.plot.set_data(cv_image)
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