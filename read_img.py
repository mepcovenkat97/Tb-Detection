import cv2
import numpy as np
import csv
from PIL import Image

img = cv2.imread('MCUCXR_0103_0.png');
print (img)
a = np.asarray(img)
for i in a:
   #np.savetxt('img.csv', a, delimiter=',')
   with open("img.csv", 'a') as f:
      writer = csv.writer(f)
      writer.writerow(i)