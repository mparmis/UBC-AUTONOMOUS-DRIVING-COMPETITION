
import cv2
import os


folder_path="/home/fizzer/Desktop/MyWorkFor353/LP_Data/"

files = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))] 
for j, img in enumerate(files):
    

	frame=cv2.imread(folder_path+img)

	width = int(frame.shape[1] / 8 )
	height = int(frame.shape[0] / 8)
	dim = (width, height)
	f2= cv2.resize(frame, dim)

	image_path = '/home/fizzer/Desktop/MyWorkFor353/LP_Data_Low_Res/' + str(img) + "2.png"   
	cv2.imwrite(image_path,f2)