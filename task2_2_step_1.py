#Implement the method and apply it to the 3 provided datasets. To evaluate the quality of the
#results draw on each query image green bounding-boxes of the detected templates.

#Step 1: Sample P positive samples from your training data of 
#the object(s) you want to detect and extract HOG descriptors from these samples.

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from config import roadtemp, road1, road2, road3, road4, road5, road6, road7, road8, road9, road10
from PIL import Image
import numpy as np
import cv2

im_list = [roadtemp, road1, road2, road3, road4, road5, road6, road7, road8]
hog_image_list = []
test_image_list = [road9, road10]
test_hog_image_list = []

for element in im_list:
	hog = cv2.HOGDescriptor()
	im = cv2.imread(element)
	h = hog.compute(im)
	hog_image_list.append(h)

for element2 in test_image_list:
	hog = cv2.HOGDescriptor()
	im = cv2.imread(element2)
	h = hog.compute(im)
	test_hog_image_list.append(h)

