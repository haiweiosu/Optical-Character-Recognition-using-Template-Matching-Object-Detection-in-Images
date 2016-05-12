#step 4

#Apply hard-negative mining. For each image and each possible scale of 
#each image in your negative training set, apply the sliding window technique 
#and slide your window across the image. At each window compute your HOG descriptors 
#and apply your classifier. If your classifier (incorrectly) classifies a given window 
#as an object (and it will, there will absolutely be false-positives), 
#record the feature vector associated with the false-positive patch along with the 
#probability of the classification. This approach is called hard-negative mining.

# USAGE
# python sliding_window.py --image images/adrian_florida.jpg 

# import the necessary packages
from imagesearch.helpers import pyramid
from imagesearch.helpers import sliding_window
from task2_2_step_3 import lin_svc
from config import negative_training_1, negative_training_2, negative_training_3, negative_training_4
from config import road9, road10
from skimage.feature import hog
from PIL import Image
from skimage import color, exposure
from sklearn.svm.libsvm import decision_function

import argparse
import time
import cv2
import numpy as np

# load the image and define the window width and height
image = cv2.imread(road10)
(winW, winH) = (225, 225)
false_positives = []
# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			# print window.shape[0], window.shape[1]
			continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW

		hog = cv2.HOGDescriptor()
		h = hog.compute(window)

		print len(h)

		# prediction = lin_svc.predict(hist_of_gradient(window))
		# if false_pos, then append to false_positives

		prediciton = lin_svc.predict(h.reshape(1,-1))

		print prediciton
		# if prediciton[0] == 1:
		# 	proba = decision_function(h)
		# 	false_positives.append(proba)



		# # since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(12)