import os
import cv2

input_path = input("Input path : ")
I = int(0)
for i in os.listdir(input_path):
    I = I+1
    tmp = cv2.imread(input_path + "\\" + i)
    if tmp.shape[0] < 100 or tmp.shape[1] < 100:
        os.remove(input_path + "\\" + i)
        print("Del : " + input_path + "\\" + i)
    if I % 10 == 0:
        print(str(I))