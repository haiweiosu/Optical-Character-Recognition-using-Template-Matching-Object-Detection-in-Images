import cv2
import numpy as np
from matplotlib import pyplot as plt
from config import Ariel, Calibari, PlatinoLinotype, TimesNewRoman, Verdana, text

template_img_dir = [Ariel, Calibari, PlatinoLinotype, TimesNewRoman, Verdana]

img = cv2.imread(text, 0)

for temp in template_img_dir:
	template = cv2.imread(temp, 0)
	w, h = template.shape[::-1]
	img2 = img.copy()
	method = eval('cv2.TM_CCOEFF')

	#Apply template Matching
	res = cv2.matchTemplate(img2, template, method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	top_left = max_loc
	bootom_right = (top_left[0] + w, top_left[1] + h)

	cv2.rectangle(img2, top_left, bootom_right, 255, 2)

	plt.subplot(121),plt.imshow(res,cmap = 'gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img2,cmap = 'gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	plt.suptitle(temp)

	plt.show()




