import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread(r"/home/kinannotfound/vscode_projects/computer_vision/homework1/knuth_300x300_color.jpg", cv.IMREAD_GRAYSCALE)
assert image is not None, "Image not found"
ret, thresh1 = cv.threshold(image, 100, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(image, 100, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(image, 100, 255, cv.THRESH_TOZERO)
ret, thresh4 = cv.threshold(image, 100, 255, cv.THRESH_TOZERO_INV)
ret, thresh5 = cv.threshold(image, 100, 255, cv.THRESH_TRUNC)
thresh6 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
thresh7 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
histogram = cv.calcHist([image], [0], None, [256], [0, 256])
thresh_images = [image, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7]
thresh_titles = ['original', 'binary thresh', 'binary inv thresh', 'to zero thresh', 'to zero inv thresh', 'trunc thresh', 'mean thresh', 'gaussian thresh']

# show histogram
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of the original Image')
plt.plot(histogram, color='red')
plt.show()

# show thresh holds
for i in range(len(thresh_images)):
	plt.subplot(1, len(thresh_images), i+1)
	plt.imshow(thresh_images[i], 'gray')
	plt.title(thresh_titles[i])
	plt.xticks([]), plt.yticks([])
plt.show()