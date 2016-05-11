#Implement the method and apply it to the 3 provided datasets. To evaluate the quality of the
#results draw on each query image green bounding-boxes of the detected templates.

#Step 1: Sample P positive samples from your training data of 
#the object(s) you want to detect and extract HOG descriptors from these samples.

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from config import roadtemp
from PIL import Image
import numpy as np


im = Image.open(roadtemp)
pix = np.array(im)
image = color.rgb2gray(pix)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()