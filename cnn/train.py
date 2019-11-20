from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend

import os
import math
import numpy as np
import re

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image 

import cv2


label_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

IM_HEIGHT = 178
IM_WIDTH = 120

def get_data(folder_path):
    
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.png')]
    np.random.shuffle(files)

    MAX_NUM_IMAGES = len(files)

    x = []
    y = []


    for j, img_path in enumerate(files):
        #open image F
        
        if j == MAX_NUM_IMAGES:
            break
        if j % 50 == 0:
            print('on image: ' + str(j))

        img_raw = cv2.imread(folder_path + img_path)

        #preprocessing to convert image to 0-1 scale
        #adding dim 1 to end of image
        im_mid = (cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY))

        img_processed = np.expand_dims( cv2.resize( cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY), (IM_WIDTH, IM_HEIGHT)), axis=2).astype('float32')/255
       
        if img_processed.shape[0] is not IM_HEIGHT and img_processed.shape[1] is not IM_WIDTH:
            print('error wrong shape:' + str(j))

        letter = img_path[0]
        if j%4 == 0:
            #print(letter) 
            print(label_options.index(letter))
        one_hot = np.zeros(len(label_options))
        one_hot[label_options.index(letter)] = 1
        #one_hot_final = np.expand_dims(one_hot, axis=1)
        
        x.append(img_processed)
        y.append(one_hot)

    x_return = np.stack(x) 
    y_return = np.stack(y)

    return x_return, y_return

x, y = get_data('./cnn/aug_pics/')

print('shape of x data: ' +str(x.shape))
print('shape of y data: ' + str(y.shape))

def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                             input_shape=(IM_HEIGHT, IM_WIDTH, 1)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dense(len(label_options), activation='softmax'))

conv_model.summary()

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

reset_weights(conv_model)
print("--beginning train--")
history_conv = conv_model.fit(x, y, 
                              validation_split=0.15, 
                              epochs=3, 
                              batch_size=16)

plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

print('beginning save of model--')

model_save_name = 'model_test_SECOND'
model_json = conv_model.to_json()
with open("./cnn/model_saves/" + model_save_name+ ".json", "w") as json_file:
    json_file.write(model_json)

conv_model.save_weights('./cnn/model_saves/' + model_save_name + '.h5')
print('--model saved as: ' + model_save_name + "--")

