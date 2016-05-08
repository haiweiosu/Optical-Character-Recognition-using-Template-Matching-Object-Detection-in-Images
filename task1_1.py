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
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# top_left = max_loc
# bootom_right = (top_left[0] + w, top_left[1] + h)

threshold = 0.9
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
	 cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

# plt.subplot(121),plt.imshow(res,cmap = 'gray')
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img2,cmap = 'gray')
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# plt.suptitle(temp)

cv2.imwrite('res.jpg',img2)




