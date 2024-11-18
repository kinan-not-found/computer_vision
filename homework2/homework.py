import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread(r"/home/kinannotfound/vscode_projects/computer_vision/homework2/image_2024-11-15_23-06-07.png")
assert image is not None, "Image not found"
ret, thresh = cv.threshold(image, 150, 255, cv.THRESH_BINARY)
str_element = np.ones((4,4), np.uint8)
erosion = cv.erode(thresh, str_element, iterations=1)
dilation = cv.dilate(erosion, str_element, iterations=1)
closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, str_element)
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, str_element)
images = [image, thresh, erosion, dilation, closing, opening]
titles = ['original', 'binary thresh', 'erosion', 'dilation', 'closing', 'opening']
for i in range(len(images)):
	plt.subplot(1, len(images), i+1)
	plt.imshow(images[i])
	plt.title(titles[i])
	plt.xticks([]), plt.yticks([])
plt.show()