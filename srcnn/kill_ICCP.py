import os
import cv2
from skimage import io
path = input("需要处理的目录 : ")
#西瓜6的代码
out_path = input("输出文件夹 : ")
fileList = os.listdir(path)
for i in fileList:
    image = io.imread(path + "\\" + i)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imencode('.png',image)[1].tofile(out_path + "\\" + i)