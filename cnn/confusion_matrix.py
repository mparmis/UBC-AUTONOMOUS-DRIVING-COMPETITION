from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

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


def get_data_infer_full(folder_path):
    
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

        img_raw = cv2.resize(cv2.imread(folder_path + img_path), (600, 1498))
            #for images from gazebo
        yi = [1105, 1105] #1105, 1105, 590]
        xi = [40, 140] #340, 445, 330]
        dy = [180, 180] #180, 180, 300]
        dx = [ 120, 120] #120, 120, 240]

        # yi = [1320, 1320, 1320, 1320, 740]
        # xi = [40, 140, 340, 445, 330]
        # dy = [180, 180, 180, 180, 300]
        # dx = [ 120, 120, 120, 120, 240]

        final_size = (IM_WIDTH, IM_HEIGHT)
        
        sub_ims = split_im(img_raw, yi, xi, dy, dx, final_size)

        for im in sub_ims:        
            img_raw = im
            
            x_raw.append(img_raw)

            #preprocessing to convert image to 0-1 scale
            #adding dim 1 to end of image
            

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

    return x_return, y_return#, x_raw


def get_pics(path_to_pic):

    im_rough = cv2.imread(path_to_pic)

    im_raw = cv2.resize(im_rough, (600, 1498)) #<- original training raw im size
        
    ims_processed = []
    #filtering for filtered image here:

    #splitting into sections

    #for images from gazebo
    yi = [1105, 1105] #1105, 1105, 590]
    xi = [40, 140] #340, 445, 330]
    dy = [180, 180] #180, 180, 300]
    dx = [ 120, 120] #120, 120, 240]

    # yi = [1320, 1320, 1320, 1320, 740]
    # xi = [40, 140, 340, 445, 330]
    # dy = [180, 180, 180, 180, 300]
    # dx = [ 120, 120, 120, 120, 240]

    final_size = (IM_WIDTH, IM_HEIGHT)
    
    sub_ims = split_im(im_raw, yi, xi, dy, dx, final_size)
    sub_ims_raw = split_im(im_raw, yi, xi, dy, dx, final_size)

    for i, img in enumerate(sub_ims):
         ims_processed.append( np.expand_dims( cv2.resize( cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY), (IM_WIDTH, IM_HEIGHT)) , axis=2).astype('float32')/255)

    return ims_processed, sub_ims_raw


#loading model:

model_path = '/home/fizzer/Desktop/Enph353_JP/353_ws/src/Enph353-JP/Enph353-JP/cnn/model_saves/model_test_5'#leav off .[extension]
json_file = open(model_path + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_path+ ".h5")

print('model_loaded from disk')

#testing ims:
#ims, ims_raw = get_pics('/home/jwhite2a/Enph353-JP/cnn/raw_pics/AF36_P241.png')

#x_infer, y_infer, y_raw = get_data_infer_full('/home/jwhite2a/Desktop/testing_imgs/')
raw_start_im = cv2.imread('/media/fizzer/ESD-USB/enph353/pics/88.png')
x_infer, y_raw = get_data_infer_full(raw_start_im)

y_predict = loaded_model.predict(x_infer) #********

#index = 1

#print(y_predict[index])
#p_i = np.argmax(y_predict[index])
#print('prediction: ' + str(label_options[p_i]))
#print('mag: ' + str(y_predict[index, p_i]))

#print('ans: ' + str(y_infer[index]))
#fg = plt.figure()
#ax = fg.gca()
#h = ax.imshow(y_raw[index])
#plt.draw(), plt.pause(0.1)


#img2 = cv2.cvtColor(y_raw[index], cv2.COLOR_BGR2GRAY)

#fg = plt.figure()
#ax = fg.gca()
#h = ax.imshow(img2, cmap='gray')
#plt.draw(), plt.pause(100)

x_test, y_test = get_data_infer_full('/home/fizzer/Desktop/Enph353_JP/353_ws/src/Enph353-JP/Enph353-JP/cnn/aug_letters/')
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test/255


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_predict)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig, ax = plt.subplots(figsize=(15, 15))
  
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test.argmax(axis=0), y_predict.argmax(axis=0), classes=label_options,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test.argmax(axis=0), y_predict.argmax(axis=0), classes=label_options, normalize=True,
                      title='Normalized confusion matrix')
plt.figure(num=None, figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
#plt.show()