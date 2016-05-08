# Find out what is the size of the font using the images in the sizes folder and the SSD measure.
#Show the results.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from config import size_10, size_11, size_12, size_14, size_16, text

img = cv2.imread(text, 0)

template = size_16
template2 = cv2.imread(template, 0)
w, h = template2.shape[::-1]
img2 = img.copy()

#Apply template Matching
res = cv2.matchTemplate(img2,template2,cv2.TM_CCOEFF_NORMED)

threshold = 0.9
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
	 cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.jpg',img2)