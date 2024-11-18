import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread(r"/home/kinannotfound/vscode_projects/computer_vision/homework3/_92867943_pewdiepie.jpg", cv.IMREAD_GRAYSCALE)
assert image is not None, "Image not found"
histogram = cv.calcHist([image], [0], None, [256], [0, 256])
equalized_image = cv.equalizeHist(image)
equalized_histogram = cv.calcHist([equalized_image], [0], None, [256], [0, 256])

# show original histogram
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of the original Image')
plt.plot(histogram, color='blue')
plt.show()

# show equalized image histogram
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of the equalized Image')
plt.plot(equalized_histogram, color='blue')
plt.show()

# original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('original image')
plt.xticks([]), plt.yticks([])

# equalized image
plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('equalized image')
plt.xticks([]), plt.yticks([])

plt.show()