import cv2
import csv
import numpy as np
import os
import pyqrcode
import random
import string

from random import randint
from PIL import Image, ImageFont, ImageDraw

path = '/home/fizzer/Desktop/Enph353_JP/353_ws/src/Enph353-JP/Enph353-JP/cnn/' #from above

#save dir
save_dir = path + 'raw_pics/'

with open(path + "plates.csv", 'w') as plates_file:
    csvwriter = csv.writer(plates_file)

    for i in range(0, 20):

        # Pick two random letters
        plate_alpha = ""
        for _ in range(0, 2):
            plate_alpha += (random.choice(string.ascii_uppercase))
        num = randint(0, 99)

        # Pick two random numbers
        plate_num = "{:02d}".format(num)

        # Save plate to file
        csvwriter.writerow([plate_alpha+plate_num])

        # Write plate to image
        blank_plate = cv2.imread(path+'blank_plate.png')

        # To use monospaced font for the license plate we need to use the PIL
        # package.
        # Convert into a PIL image (this is so we can use the monospaced fonts)
        blank_plate_pil = Image.fromarray(blank_plate)
        # Get a drawing context
        draw = ImageDraw.Draw(blank_plate_pil)
        monospace = ImageFont.truetype(path + "ubuntumonor.ttf", 200)
        draw.text((48, 105),plate_alpha + " " + plate_num, (255,0,0), font=monospace)
        # Convert back to OpenCV image and save
        blank_plate = np.array(blank_plate_pil)

        # cv2.putText(blank_plate,
        #             plate_alpha + " " + plate_num, (45, 360),
        #             cv2.FONT_HERSHEY_PLAIN, 11, (255, 0, 0), 7, cv2.LINE_AA)

        # Create parking spot label
        s = "P" + str(i)
        parking_spot = 255 * np.ones(shape=[600, 600, 3], dtype=np.uint8)
        cv2.putText(parking_spot, s, (30, 450), cv2.FONT_HERSHEY_PLAIN, 28,
                    (0, 0, 0), 30, cv2.LINE_AA)
        spot_w_plate = np.concatenate((parking_spot, blank_plate), axis=0)

        # Merge labelled or unlabelled images and save
        #labelled = np.concatenate((QR_img, spot_w_plate), axis=0)
        unlabelled = np.concatenate((255 * np.ones(shape=[600, 600, 3],
                                    dtype=np.uint8), spot_w_plate), axis=0)
        #cv2.imwrite(os.path.join(path + save_dir + "labelled/",
                                 #"plate_" + str(i) + ".png"), labelled)
        cv2.imwrite(os.path.join(save_dir,
                                 "{}{}_{}.png".format(plate_alpha, plate_num,s)), unlabelled)

