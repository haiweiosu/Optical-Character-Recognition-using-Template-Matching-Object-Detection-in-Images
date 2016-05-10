#Implement the method and apply it to the 3 provided datasets. To evaluate the quality of the
#results draw on each query image green bounding-boxes of the detected templates.

# USAGE
# python sliding_window.py --image images/adrian_florida.jpg 

# import the necessary packages
from imagesearch.helpers import pyramid
from imagesearch.helpers import sliding_window
from config import roadtemp, road1, road2
import argparse
import time
import cv2

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# load the image and define the window width and height
image = cv2.imread(road1)
(winW, winH) = (128, 128)

image2 = image.copy()

imagew, imageh = image2[::-1]


#load the template image
template = cv2.imread(roadtemp, 0)




# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW

		# since we do not have a classifier, we'll just draw the window
		resizedclone = resized.copy()

		#Apply template Matching
		res = cv2.matchTemplate(resizedclone,template,cv2.TM_CCOEFF_NORMED)

		#set threshold for target classifier
		threshold = 0.70
		loc = np.where(res >= threshold)
		for pt in zip(*(loc[::-1]))
			cv2.rectangle(resizedclone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		

		cv2.imshow("Window", resizedclone)
		# cv2.waitKey(1)
		# time.sleep(0.025)