from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2
import os
from PIL import Image
import numpy as np

from skimage.util import random_noise
# load the image
raw_pics_path = 'cnn/raw_pics/'

augment1 = []
augment_names = []


files = [img for img in os.listdir(raw_pics_path) if os.path.isfile(os.path.join(raw_pics_path, img))] 
#random shuffle
np.random.shuffle(files)
#can chan
MAX_NUM_IMAGES = len(files)

for j, img_path in enumerate(files):

    if j == MAX_NUM_IMAGES:
        break
    if j % 50 == 0:
        print('on image: ' + str(j))

#add random shift
for j, img in enumerate(files):
    data = img_to_array(load_img(raw_pics_path+img))
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(width_shift_range=[-20.0,20.0])
    it = datagen.flow(samples, batch_size=1)
# generate samples and plot
    for i in range(4):
        # define subplot
        # pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        augment1.append(batch[0])
        augment_names.append(img)
            

            

    #add random shearing


    #add radom brightness shift
folder_path=raw_pics_path
augment2 = []
    
files = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
for j, img in enumerate(augment1):
    datagen = ImageDataGenerator(brightness_range=[0.7,1.3])
    it = datagen.flow(img, batch_size=1)
    # generate samples and plot
    for i in range(4):
        # define subplot
        # pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        augment2.append(batch[0])


    # resolution Change 

folder_path=raw_pics_path
augment3 = []


files = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))] 
for j, img in enumerate(augment2):
    if j % 2 == 0:
	    frame = img
	    #print(frame.shape[0])
	    width = int(frame.shape[1] * 5/ 100)
	    height = int(frame.shape[0] * 5/ 100)
	    dim = (width, height)
	    f2= cv2.resize(frame, dim)
	    frame2=f2
	    width = int(img.shape[1])
	    height = int(img.shape[0])
	    dim = (width, height)
	    f3= cv2.resize(frame2, dim)
	    final_im = f3
    else:
	    final_im = img
	    augment3.append(final_im)
            



    #filter to binary masks

    #cut images cut

all_pics = []
all_letters = []
  
for i, img in enumerate(augment3):
    #open image F
    grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #note, should be removed if saving images
        
    grayim_scaled = grayim/255
    img = grayim_scaled

    
    dy_plate = 180
    dx_plate = 120

    dx_pos = 240
    dy_pos = 300

    im1 = img[1320:1320+dy_plate, 40:40+dx_plate]
    im2 = img[1320:1320+dy_plate, 140:140+dx_plate]
    im3 = img[1320:1320+dy_plate, 340:340+dx_plate]
    im4 = img[1320:1320+dy_plate, 445:445+dx_plate]
    pos1 = img[740:740+dy_pos, 330:330+dx_pos]

    pos_resize = cv2.resize(pos1, (im1.shape[1], im1.shape[0]))

    l1 = augment_names[i][0]
    l2= augment_names[i][1]
    l3= augment_names[i][2]
    l4= augment_names[i][3]
    l5= augment_names[i][6]

    all_pics.append(im1)
    all_letters.append(l1)
        
    all_pics.append(im2)
    all_letters.append(l2)
        
    all_pics.append(im3)
    all_letters.append(l3)
        
    all_pics.append(im4)
    all_letters.append(l4)
                   
    all_pics.append(pos_resize)
    all_letters.append(l5)    
