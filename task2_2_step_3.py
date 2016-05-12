#Step 3

#Train a Linear Support Vector Machine on your positive and negative samples.

from sklearn import svm
from task2_2_step_1 import hog_image_list, test_hog_image_list
from task2_2_step_2 import neg_image_list
import numpy as np

# C = 1.0  # SVM regularization parameter

X = []
for element in hog_image_list:
	X.append(element.flatten())
for element in neg_image_list:
	X.append(element.flatten())

test_X = []
for elment in test_X:
	test_X.append(element.flatten())

Y = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0]

# print np.array(X).shape
# print X[0]


lin_svc = svm.LinearSVC(random_state = 0).fit(X, Y)

# print lin_svc