#Find out what is the font of the text using the images in the fonts folder and the SSD measure.
#Explain the process and show the results.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from config import Ariel, Calibari, PlatinoLinotype, TimesNewRoman, Verdana, text

# template_img_dir = ['Ariel', 'Calibari', 'PlatinoLinotype', 'TimesNewRoman', 'Verdana']

img = cv2.imread(text, 0)

template = Verdana
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




