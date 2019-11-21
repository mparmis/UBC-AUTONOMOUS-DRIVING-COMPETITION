#need to add libs
import cv2
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import os

IM_HEIGHT = 60
IM_WIDTH = 40
label_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def split_im(im, yi, xi, dy, dx, final_size):
    #y_i, x_i, dy, dx all arrays of same size
    ims = []
    for i, _ in enumerate(yi):
        im_temp = im[ yi[i] : yi[i] + dy[i], xi[i] : xi[i]+dx[i]]
        ims.append(cv2.resize(im_temp, final_size))
    return ims

def get_data(folder_path):
    
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.png')]
    np.random.shuffle(files)

    MAX_NUM_IMAGES = len(files)

    x = []
    y = []
    x_raw = []


    for j, img_path in enumerate(files):
        #open image F
        
        if j == MAX_NUM_IMAGES:
            break
        if j % 50 == 0:
            print('on image: ' + str(j))

        img_raw = cv2.imread(folder_path + img_path)
        x_raw.append(img_raw)

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

    return x_return, y_return, x_raw



#loading model:

model_path = './cnn/model_saves/model_test_5'#leav off .[extension]
json_file = open(model_path + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_path+ ".h5")

print('model_loaded from disk')

##INFER:
x_infer, y_infer, y_raw = get_data('cnn/aug_pics_test/')

y_predict = loaded_model.predict(x_infer)

index = 14

print(y_predict[index])
p_i = np.argmax(y_predict[index])
print('prediction: ' + str(label_options[p_i]))
print('mag: ' + str(y_predict[index, p_i]))

#print('ans: ' + str(y_infer[index]))
fg = plt.figure()
ax = fg.gca()
h = ax.imshow(y_raw[index])
plt.draw(), plt.pause(0.1)


img2 = cv2.cvtColor(y_raw[index], cv2.COLOR_BGR2GRAY)

fg = plt.figure()
ax = fg.gca()
h = ax.imshow(img2, cmap='gray')
plt.draw(), plt.pause(100)

