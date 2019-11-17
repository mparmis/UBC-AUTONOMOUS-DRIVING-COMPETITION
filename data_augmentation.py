from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2
# load the image
for j in range(0, 201): 
    img = load_img('/home/fizzer/Desktop/MyWorkFor353/License_Plates/%d.png' % (j))
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(brightness_range=[0.7,1.3])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(4):
        # define subplot
        # pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        #pyplot.imshow(image)
        image_path = '/home/fizzer/Desktop/MyWorkFor353/Augmented_Data_Brightness/%d_%d.png' % (j,i) 
        cv2.imwrite(image_path,image)
    # show the figure
    #pyplot.show()