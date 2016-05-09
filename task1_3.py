#Find out how many times do the letters 't' and 'a' appear in the text, including capital letters. Do
#this by cutting the appropriate templates from the text itself (see previous items for good
#template examples) and use the SSD measure. Try to find a single rule for setting the threshold
#for the matching (it can be a function of the template).
	#a. Show your templates.
	#b. Explain your threshold selection
	#c. Write how many times each of the letters appear. Did you locate all the appearances?

import cv2
import numpy as np
from matplotlib import pyplot as plt
from config import small_a, small_t, big_A, big_T, text


img = cv2.imread(text, 0)

template = big_T
template2 = cv2.imread(template, 0)
w, h = template2.shape[::-1]
img2 = img.copy()

#Apply template Matching
res = cv2.matchTemplate(img2,template2,cv2.TM_CCOEFF_NORMED)

threshold = 0.729816
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
	 cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.jpg',img2)
