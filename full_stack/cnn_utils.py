import cv2
import numpy as np


IM_HEIGHT = 69
IM_WIDTH = 40
label_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def split_im(im, yi, xi, dy, dx, final_size):
    #y_i, x_i, dy, dx all arrays of same size
    ims = []
    for i, _ in enumerate(yi):
        im_temp = im[ yi[i] : yi[i] + dy[i], xi[i] : xi[i]+dx[i]]
        ims.append(cv2.resize(im_temp, final_size))
    return ims


def convert_pic(raw_pic):

    im_raw = cv2.resize(raw_pic, (600, 1498)) #<- original training raw im size
    ims_processed = []
    #filtering for filtered image here:

    #splitting into sections

    #for images from gazebo
    # yi = [1105, 1105, 1105, 1105, 590]
    # xi = [40, 140, 340, 445, 330]
    # dy = [180, 180, 180, 180, 300]
    # dx = [ 120, 120, 120, 120, 240]

    yi = [1320, 1320, 1320, 1320, 740]
    xi = [40, 140, 340, 445, 330]
    dy = [180, 180, 180, 180, 300]
    dx = [ 120, 120, 120, 120, 240]

    final_size = (IM_WIDTH, IM_HEIGHT)
    
    sub_ims = split_im(im_raw, yi, xi, dy, dx, final_size)
    sub_ims_raw = split_im(im_raw, yi, xi, dy, dx, final_size)

    for i, img in enumerate(sub_ims):
         ims_processed.append( np.expand_dims( cv2.resize( cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY), (IM_WIDTH, IM_HEIGHT)) , axis=2).astype('float32')/255)

    return ims_processed, sub_ims_raw

