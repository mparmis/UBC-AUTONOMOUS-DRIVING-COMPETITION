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


files = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))] 
#random shuffle
np.random.shuffle(files)
#can chan
MAX_NUM_IMAGES = len(files)

for j, img_path in enumerate(files):

    if j == MAX_NUM_IMAGES:
        break
    if j % 50 == 0:
        print('on image: ' + str(j))

    img_raw = cv2.imread(raw_pics_path + img_path)

    #add noise
    #needs testings
    img_noise = random_noise(img, mode='s&p', amount=0.3)

    #add random shift
    for j, img in enumerate(files):
        data = img_to_array(load_img(folder_path+img))
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
            image = batch[0].astype('uint8')
            image_path = './cnn/aug_pics/'+ str(img) + "%d.png" % (i)
            grayim = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            

    #add random shearing

    #add radom brightness shift

    #filter to binary masks

    #cut images cut
    sub_ims = []

    dy_plate = 180
    dx_plate = 120

    dx_pos = 240
    dy_pos = 300

    sub_ims.append(img[1320:1320+dy_plate, 40:40+dx_plate])
    sub_ims.append(img[1320:1320+dy_plate, 140:140+dx_plate]
    sub_ims.append(img[1320:1320+dy_plate, 340:340+dx_plate]
    sub_ims.append(img[1320:1320+dy_plate, 445:445+dx_plate]
    sub_ims.append(img[740:740+dy_pos, 330:330+dx_pos]


    #save all ims
    

    #splitting ims up:


    if j % 50 == 0:
        print('on image: ' + str(j))