#need to add libs
import cv2
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

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


def get_pics(path_to_pic):

    im_rough = cv2.imread(path_to_pic)

    im_raw = cv2.resize(im_rough, (600, 1498)) #<- original training raw im size
        
    ims_processed = []
    #filtering for filtered image here:

    #splitting into sections

    #for images from gazebo
    yi = [1105, 1105, 1105, 1105, 590]
    xi = [40, 140, 340, 445, 330]
    dy = [180, 180, 180, 180, 300]
    dx = [ 120, 120, 120, 120, 240]

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

model_path = './cnn/model_saves/model_test_5'#leav off .[extension]
json_file = open(model_path + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_path+ ".h5")

print('model_loaded from disk')

#testing ims:
#ims, ims_raw = get_pics('/home/jwhite2a/Enph353-JP/cnn/raw_pics/AF36_P241.png')
ims, ims_raw = get_pics('/home/jwhite2a/Desktop/testing_imgs/GD55_P1_1.png')
ims = np.stack(ims)

##

# imnew = cv2.imread("cnn/aug_pics/H_440_44.png")
# newim_list = []
# newim_list.append( np.expand_dims( cv2.resize( cv2.cvtColor(imnew, cv2.COLOR_BGR2GRAY), (IM_WIDTH, IM_HEIGHT)) , axis=2).astype('float32')/255)

# y_new = np.stack(newim_list)

# ims_raw = [imnew]



#y_predict = loaded_model.predict(y_new)

y_predict = loaded_model.predict(ims)

index = 1

print(y_predict[index])
p_i = np.argmax(y_predict[index])
print('prediction: ' + str(label_options[p_i]))
print('mag: ' + str(y_predict[index, p_i]))

fg = plt.figure()
ax = fg.gca()
h = ax.imshow(ims_raw[index])
plt.draw(), plt.pause(0.1)


img2 = cv2.cvtColor(ims_raw[index], cv2.COLOR_BGR2GRAY)

fg = plt.figure()
ax = fg.gca()
h = ax.imshow(img2, cmap='gray')
plt.draw(), plt.pause(100)


#print('ans: ' + str(test_ans[index]))