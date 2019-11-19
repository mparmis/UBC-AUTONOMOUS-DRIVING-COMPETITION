
import cv2
import os


folder_path="/home/fizzer/Desktop/MyWorkFor353/Augmented_Data_Brightness/"

files = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))] 
for j, img in enumerate(files):
    

	frame=cv2.imread(folder_path+img)

	width = int(frame.shape[1] * 5/ 100)
	height = int(frame.shape[0] * 5/ 100)
	dim = (width, height)
	f2= cv2.resize(frame, dim)

	image_path = '/home/fizzer/Desktop/MyWorkFor353/' + str(img) + "2.png"   
	cv2.imwrite(image_path,f2)

	frame2=cv2.imread(image_path)

	width = int(frame2.shape[1] * 2000/ 100)
	height = int(frame2.shape[0] * 2000/ 100)
	dim = (width, height)
	f3= cv2.resize(frame2, dim)

	image_path = '/home/fizzer/Desktop/MyWorkFor353/Augmented_Data_Resized/' + str(img) + "2.png" 
	cv2.imwrite(image_path,f3)

