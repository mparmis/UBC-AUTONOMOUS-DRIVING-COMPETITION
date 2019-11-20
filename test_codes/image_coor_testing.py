import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('media/test1.png',1)

dx_plate = 120
dy_plate = 180

dx_pos = 240
dy_pos = 300

im1 = img[1320:1320+dy_plate, 40:40+dx_plate]
im2 = img[1320:1320+dy_plate, 140:140+dx_plate]
im3 = img[1320:1320+dy_plate, 340:340+dx_plate]
im4 = img[1320:1320+dy_plate, 445:445+dx_plate]
pos1 = img[740:740+dy_pos, 330:330+dx_pos]

print( str(int(im1.shape[1])) + " "  + str(int(im1.shape[0])))

pos_resize = cv2.resize(pos1, (int(im1.shape[1]), int(im1.shape[0])))

fg = plt.figure()
ax = fg.gca()
h = ax.imshow(im2)
plt.draw(), plt.pause(100)



# 


