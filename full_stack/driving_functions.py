import cv2
import numpy as np



def section1_driving(cv_image):
    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
    mask_edge = cv2.inRange(gray_im, 240, 280)
    
    vel_lin = 0.001
    vel_ang = 0
    flag = 0

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
        flag = 1
        vel_ang = 0
        vel_lin = 0

    return vel_lin, vel_ang, flag, plot_image


def section2_driving(cv_image):
    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
    mask_edge = cv2.inRange(gray_im, 240, 280)

    vel_lin = 0.000
    vel_ang = 0.005
    flag = 0

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
        flag = 1
        vel_lin = 0
        vel_ang = 0
    return vel_lin, vel_ang, flag, plot_image


def section3_driving(cv_image, last_error):
    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
    mask_edge = cv2.inRange(gray_im, 240, 280)
    mask_crosswalk = cv2.inRange(cv_image, (0, 0, 240), (15, 15, 255))
    
    plot_image = mask_crosswalk

    vel_lin = 0
    vel_ang = 0
    flag = 0

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
    ang_vel = s3_kp*(error) - s3_kd*(last_error- error)
    new_last_error = error

    if(abs(error) > 55):      
        vel_ang = ang_vel
        vel_lin = 0.00
    else:
        vel_lin = 0.0001
        #vel.angular.z = 0.001

    #check for crosswalk:
    s3_x = 600
    s3_y = 575
    s3_dy = 125
    s3_dx = 200
    s3_y_low = int(s3_y - (s3_dy/2))
    s3_y_high = int(s3_y + (s3_dy/2))
    s3_x_low = int(s3_x - (s3_dx/2))
    s3_x_high = int(s3_x + (s3_dx/2))

    #val = np.any( np.transpose(mask_edge)[s1_x][s1_y_low:s1_y_high])
    val = np.any( mask_crosswalk[s3_y_low:s3_y_high, s3_x_low:s3_x_high])
    #print(mask_crosswalk[s3_y_low:s3_y_high, s3_x_low:s3_x_high])
    print("val" + str(val))
    if val:
        print('s3: edge found!')
        print('ending')
        flag = 1
        vel_lin = 0
        vel_ang = 0
    
    return vel_lin, vel_ang, flag, plot_image, new_last_error

