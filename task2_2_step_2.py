#Step 2

#Sample N negative samples from a negative training set 
#that does not contain any of the objects you want to detect and extract HOG descriptors 
#from these samples as well. In practice N >> P.

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from config import negative_training_1, negative_training_2, negative_training_3, negative_training_4, negative_training_5, negative_training_6
from PIL import Image
import numpy as np
import cv2


im_list = [negative_training_1,negative_training_2, negative_training_3, negative_training_4, negative_training_5, negative_training_6]
neg_image_list = []

for element in im_list:
	hog = cv2.HOGDescriptor()
	im = cv2.imread(element)
	h = hog.compute(im)
	neg_image_list.append(h)

