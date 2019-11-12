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


class image_converter:

  def __init__(self):
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    self.bridge = CvBridge()
    self.image_count=0

  def callback(self, data):

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      print("got image!")
    except CvBridgeError as e:
      print(e)


    
    cv2.imshow("Image window", cv_image)
    #cv2.waitKey(3)
    image_path = '/home/fizzer/Desktop/MyWorkFor353/Images/%d.png' % (self.image_count) 
    cv2.imwrite(image_path,cv_image)
    self.image_count += 1; 



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