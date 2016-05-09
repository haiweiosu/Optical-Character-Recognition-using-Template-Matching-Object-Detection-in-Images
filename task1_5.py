#Help Mr. Smoke-too-much read this text to his friends. Do this by replacing each appearance of
#the letter 'c' with the letter 'k'. Use the given templates and the SSD measure. Show the new
#text. Does it look natural? Can you read it to your partner?

import cv2
import numpy as np
from matplotlib import pyplot as plt
from config import letter_c, letter_k, text

img = cv2.imread(text, 0)

template = letter_c
template2 = cv2.imread(template, 0)

w, h = template2.shape[::-1]

img2 = img.copy()

temp = cv2.imread(letter_k, 0)


#Apply template Matching
res = cv2.matchTemplate(img2,template2,cv2.TM_CCOEFF_NORMED)

threshold = 0.85
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
	# cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	for element1 in range(w):
		for element2 in range(h):
			img2[pt[1] + element2, pt[0] + element1] = temp[element2, element1]


cv2.imwrite('res.jpg',img2)