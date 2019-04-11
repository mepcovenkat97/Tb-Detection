import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

def findedge(image_file,output):
	img = cv2.imread(image_file)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	cv2.imwrite(output,thresh)

#path = input('Enter folder path')
input_dir = glob.glob("/mnt/BA90E00C90DFCD4F/Project/TB/TB-FeatureExtraction/Dataset/Predict/MCUCXR_0301_1.png")
N = len(input_dir)
for i in range(N):
        output_dir = "edge"+str(i)+".png"
        findedge(input_dir[i], output_dir)
