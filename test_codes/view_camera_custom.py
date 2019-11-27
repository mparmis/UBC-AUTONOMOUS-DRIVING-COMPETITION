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

class myBox:
    def __init__(self, x, y, dx, dy):
        self.x = int(x)
        self.y = int(y) 
        self.x_low = int(x - (dx/2))
        self.x_high = int(x + (dx/2))
        self.y_low = int(y - (dy/2))
        self.y_high = int(y + (dy/2))

def check_box(mask_im, box):
    return np.sum(mask_im[box.y_low:box.y_high, box.x_low:box.x_high])/255


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

    mask_crosswalk = cv2.inRange(cv_image, (0, 0, 240), (15, 15, 255))
    
    mask_pants = cv2.inRange(cv_image, (50, 40, 25), (90, 70, 43))
    
    mask_tail_lights = cv2.inRange(cv_image, (0, 0, 0), (50, 50, 50))
    
    mask_trees = cv2.inRange(cv_image, (42, 45, 47), (48, 54, 60))
    
    # mask_car = cv2.inRange(cv_image, (119, 17, 17), (125, 23, 23))
    mask_car = cv2.inRange(cv_image, (98, 0, 0), (126, 23, 23))
    mask_car_2 = cv2.inRange(cv_image, (198, 96, 96), (202, 102, 102))

    mask_car = cv2.add(mask_car, mask_car_2)


        # calculate moments of binary image
    M = cv2.moments(mask_edge[:, 600:-1])
    
    # calculate x,y coordinate of center
    cX = 600+ int(M["m10"] / M["m00"])
    cY = 600+ int(M["m01"] / M["m00"])

    print("CX: " +str(cX))
    print("CY: " +str(cY))

    print(cv_image.shape)
    #plot_image = cv_image[400:700, 490:790] 
    plot_image = cv_image


    road1 = myBox(350, 400, 100, 200)
    print('gray box count:' + str((200*50) - check_box(mask_trees, road1)))
    # if (100*100) - check_box(mask_road, road1) < 100: 
    #     print('road fully seen')


    #print(mask_pants)

    total_pants = np.sum(mask_pants[400:700, 490:790])/255 #if greater than 200 move
    print(total_pants)

    print(" ")

    #matplotlib image show
    if(self.first_plot):
        fg = plt.figure()
        ax = fg.gca()
        h = ax.imshow(plot_image, cmap='gray')
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