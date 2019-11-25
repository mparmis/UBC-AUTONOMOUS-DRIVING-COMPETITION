import cv2
import numpy as np

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

def get_box_crop(im, box):
    return im[box.y_low:box.y_high, box.x_low:box.x_high]

def centroid(im):
    top = 0
    bot = 0
    index_array = np.linspace(0, im.shape[0]-1, im.shape[0] )
    #print('indexarray: ' + str(index_array))
    for r in range(0, im.shape[1]-2): #range of rows to check
        top += np.sum(np.multiply(im[:, r], index_array))
        bot += np.sum(im[:, r])
    x_bar = top  / (bot +1)
    print('xbar: '+ str(x_bar))
    return x_bar
    #print('xbar: ' + str(x_bar))


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
    #print('vals' + str(np.transpose(mask_edge)[s1_x][s1_y_low:s1_y_high]))
    #print('maskval: ' + str(val))
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
    #print('vals' + str(mask_edge[s2_y][s2_x_low:s2_x_high]))
    #print('maskval: ' + str(val))
    if val:
        print('s2: edge found!')
        flag = 1
        vel_lin = 0
        vel_ang = 0
    return vel_lin, vel_ang, flag, plot_image


def section3_driving(self, cv_image, last_error):
    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
    mask_edge = cv2.inRange(gray_im, 240, 280)
    mask_crosswalk = cv2.inRange(cv_image, (0, 0, 240), (15, 15, 255))
    mask_road = cv2.inRange(gray_im, 78, 82.5)

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
    #print('xbar: ' + str(x_bar))

    tar = 1100
    error = tar -  x_bar
    #print('error: ' + str(error))
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
    s3_y = 600
    s3_dy = 20
    s3_dx = 200
    s3_y_low = int(s3_y - (s3_dy/2))
    s3_y_high = int(s3_y + (s3_dy/2))
    s3_x_low = int(s3_x - (s3_dx/2))
    s3_x_high = int(s3_x + (s3_dx/2))

    #val = np.any( np.transpose(mask_edge)[s1_x][s1_y_low:s1_y_high])
    val = np.any( mask_crosswalk[s3_y_low:s3_y_high, s3_x_low:s3_x_high])
    #print(mask_crosswalk[s3_y_low:s3_y_high, s3_x_low:s3_x_high])
    print("val" + str(val))
    if val and self.s3_cycles > 30:
        self.s3_cycles = 0
        print('s3: edge found!')
        print('ending')
        flag = 1
        vel_lin = 0
        vel_ang = 0
    else:
        self.s3_cycles =  self.s3_cycles + 1
    
    if(self.ICS_seen_intersection):
        #wait for white line
        print('witing for white line')
        line2 = myBox(3, 500, 2, 50)
        if check_box(mask_edge, line2):
            print('found edge')
            vel_lin = 0
            vel_ang = 0
            self.section = 5 #enter turning section

    elif(self.crosswalks_passed >= 2 and self.found_plate_flag):
        road1 = myBox(200, 600, 200, 100)
        print('gray box count:' + str((200*100) - check_box(mask_road, road1)))
        if (200*100) - check_box(mask_road, road1) < 5600: # was 5600 
            print('road fully seen')
            self.ICS_seen_intersection = True


    return vel_lin, vel_ang, flag, plot_image, new_last_error

def section4_driving(self, cv_image):

    vel_lin = 0
    vel_ang = 0
    flag = 0

    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
    mask_edge = cv2.inRange(gray_im, 240, 280)
        
    mask_crosswalk = cv2.inRange(cv_image, (0, 0, 240), (15, 15, 255))
    
    mask_pants = cv2.inRange(cv_image, (50, 40, 25), (90, 70, 43))
    total  = np.sum(mask_pants[400:700, 490:790])/255 
    print(total)
    
    if not self.seen_ped:
        #if hasv't seen pedestrian
        if total > 50:
            print('seen and waiting')
            self.seen_ped = True
    else:
        if total < 50 or self.gogogo:
            print('GOGOGO')

            self.gogogo= True
            vel_lin = 1
        
        #go straight until the end of road

            b_BL = myBox(600, 540, 200, 10)
            b_RE = myBox(600, 575, 2, 125)
            
            blue_line_val  = np.any(mask_crosswalk[b_BL.y_low:b_BL.y_high, b_BL.x_low:b_BL.x_high])
            road_edge_val = np.any(mask_edge[b_RE.y_low:b_RE.y_high, b_RE.x_low:b_RE.x_high])
            print('bluelineval: ' + str(blue_line_val))
            print('roadedge val: ' + str(road_edge_val))
            if blue_line_val:
                self.passed_second_blue_line = True
                print('s4: blue line  found!')
                self.section = 3
                vel_ang = 0
                vel_lin = 0
                self.gogogo = False
                self.seen_ped = False
                self.passed_second_blue_line = False
                self.crosswalks_passed = self.crosswalks_passed+1

            # if self.passed_second_blue_line:
            #     if road_edge_val:
            #         print('s4: road edge found!')
            #         self.section = 2
            #         vel_ang = 0
            #         vel_lin = 0
            #         self.gogogo = False
            #         self.seen_ped = False
            #         self.passed_second_blue_line = False
        
    return vel_lin, vel_ang

def section5(self, cv_image):
        
    vel_lin = 0
    vel_ang = 0

    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
    mask_edge = cv2.inRange(gray_im, 240, 280)
    mask_road = cv2.inRange(gray_im, 78, 82.5)
    mask_trees = cv2.inRange(cv_image, (42, 45, 47), (48, 54, 60))

    mask_tail_lights = cv2.inRange(cv_image, (0, 0, 0), (50, 50, 50))
    
    vel_ang = 1 #was -1
    tree_box = myBox(250, 400, 150, 200)
    cond1 = (200*150) - check_box(mask_trees, tree_box)
    if not self.turn_enough_to_inner:
        print('tree count: ' + str(cond1))

    if cond1 < 17000 or self.turn_enough_to_inner:
        print('turned anough')
        self.turn_enough_to_inner = True
        vel_ang = 0
        vel_lin = 1

        #look for line in front
        inner_loop_line_box = myBox(600, 485, 10, 100) #was 450, and 550
        cond2 = check_box(mask_edge, inner_loop_line_box)
        car_box = myBox(600, 440, 10, 80)
        cond3 = check_box(mask_tail_lights, car_box)
        
        print('innerlien count: ' +str(cond2))
        print ('cond3' + str(cond3))
        if cond2 > 2 or cond3 > 10 :
            vel_lin = 0
            vel_ang = 0
            print('found line')
            self.section =  6 #next section : wait for car
    return vel_lin, vel_ang
    

def section6(self, cv_image):

    vel_lin = 0
    vel_ang = 0

    gray_im = np.dot(cv_image[...,:3], [0.299, 0.5487, 0.114])
    mask_edge = cv2.inRange(gray_im, 240, 280)
    mask_road = cv2.inRange(gray_im, 78, 82.5)
    mask_trees = cv2.inRange(cv_image, (42, 45, 47), (48, 54, 60))
    
    mask_tail_lights = cv2.inRange(cv_image, (0, 0, 0), (50, 50, 50))
    
    taillight_box = myBox(640, 540, 300, 180)
    print('full frame: ' + str(np.sum(mask_tail_lights)))
    print('check: ' + str(check_box(mask_tail_lights, taillight_box)))

    if(check_box(mask_tail_lights, taillight_box)):
        print('stared seeing truck')
        self.seen_sec6_truck  = True

    if (self.seen_sec6_truck and not check_box(mask_tail_lights, taillight_box)):
        print('time to gogogo')
        self.sec6_gogogo = True

    if (self.sec6_gogogo):
        vel_ang = 1
        white_line_box = myBox(1000, 400, 40, 100)
        if(check_box(mask_edge, white_line_box)):
            print('found inner line sec 6')
            vel_lin = 0
            vel_ang = 0
            self.section = 7

    return vel_lin, vel_ang
