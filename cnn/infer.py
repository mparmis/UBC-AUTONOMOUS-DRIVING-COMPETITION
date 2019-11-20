#need to add libs
import cv2
from keras.models import model_from_json
import numpy as np

def split_im(im, yi, xi, dy, dx, final_size):
    #y_i, x_i, dy, dx all arrays of same size
    ims = []
    for i, _ in enumerate(y_i)
        im_temp = im[ yi[i] : yi[i] + dy[i], xi[i] : xi[i]+dy[i]]
        ims.append(cv2.resize(im_temp, final_size))
    return ims


def get_pics(path_to_pic):

    im_rough = cv2.imread(path_to_pic)
    im_raw = cv2.resize(im_rough, (1498, 600)) #<- original training raw im size
    #filtering for filtered image here:

    im_processed = im_raw

    #splitting into sections

    yi = [1105, 1105, 1105, 1105, 590]
    xi = [40, 140, 340, 445, 330]
    dy = [180, 180, 180, 180, 300]
    dx = [ 120, 120, 120, 120, 240]
    final_size = (178, 120)
    
    sub_ims = split_im(im_processed, )yi, xi, dy, dx, final_size)
    sub_ims_raw = split_im(im_raw, yi, xi, dy, dx, final_size)

    return sub_ims, sub_ims_raw


#loading model:

model_path = './cnn/model_saves/model_test'#leav off .[extension]
json_file = open(model_path + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_path+ ".h5")

print('model_loaded from disk')

#testing ims:
ims, ims_raw = get_pics('/home/jwhite2a/Desktop/testing_imgs/GD55_P1_2.png')

index = 0
y_predict = loaded_model.predict(ims[index])

print(y_predict)
p_i = np.argmax(y_predict)
print('prediction: ' + str(label_options[p_i]))
print('mag: ' + str(y_predict[0, p_i]))
print('ans: ' + str(test_ans[index]))