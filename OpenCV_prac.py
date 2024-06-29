import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


'''
OpenCV, short for Open Source Computer Vision Library
Huge open-source library for computer vision, machine learning, and image processing.
Originally developed by Intel, it is now maintained by a community of developers under the OpenCV Foundation.
'''

img = cv2.imread('Roronoa_Zoro.jpg')
print(img.shape)
print(type(img))

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************
# Rotation
ht,wd = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((ht/2, wd/2), 90, 0.5)
rotated_img = cv2.warpAffine(img,rotation_matrix,(wd,ht))
# cv2.imshow('img', rotated_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************

# Cropping
''' Note: you have to cast the starting and ending values to integers 
because when mapping, the indexes are always integers.'''

start_row = int(ht*0.05)
start_col = int(wd*0.15)
end_row = int(ht*0.8)
end_col = int(wd*0.8)

cropped_img = img[start_row:end_row, start_col:end_col]
# cv2.imshow('img', cropped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************

# Resizing
img_1 = cv2.imread('japanese-girl-8274847_1280.jpg')
# cv2.imshow('img_1', img_1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

resize_img_1 = cv2.resize(img_1, (0,0), fx=0.4, fy=0.4)  # values of x and y axis
# cv2.imshow('img', resize_img_1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# using row&clmns to resize

resize_img_1_rc = cv2.resize(img_1, (550,400))
# cv2.imshow('img', resize_img_1_rc)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************

# Color channels

B,G,R = cv2.split(img)
# cv2.imshow("original", img)
# cv2.waitKey(0)
#
# cv2.imshow("blue", B)
# cv2.waitKey(0)
#
# cv2.imshow("Green", G)
# cv2.waitKey(0)
#
# cv2.imshow("red", R)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# ***************************************************************************

# Contrast
'''
new_img = a * original_img + b

Here a is alpha which defines the contrast of the image. If a is greater than 1, there will be higher contrast.
If the value of a is between 0 and 1, there would be lower contrast. 
If a is 1, there will be no contrast effect on the image.
b stands for beta. The values of b vary from -127 to +127.

To implement this equation in Python OpenCV, you can use the addWeighted() method.
We use The addWeighted() method as it generates the output in the range of 0 and 255 for a 24-bit color image.
'''
contrast_img = cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, 0)
# cv2.imshow('Original Image', img)
# cv2.imshow('Contrast Image', contrast_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************

# Blurring
'''
The GaussianBlur() uses the Gaussian kernel. 
The height and width of the kernel should be a positive and an odd number.
Then you have to specify the X and Y direction that is sigmaX and sigmaY respectively.
If only one is specified, both are considered the same.
'''
blur_img_1 = cv2.GaussianBlur(resize_img_1, (7,7), 0)
# cv2.imshow('Original Image', resize_img_1)
# cv2.imshow('Gaussian Blur', blur_img_1)

'''
In median blurring, the median of all the pixels of the image is calculated inside the kernel area.
The central value is then replaced with the resultant median value.
Median blurring is used when there are salt and pepper noise in the image.
'''

med_blur_img = cv2.medianBlur(resize_img_1, 5)
# cv2.imshow('Median Blur', med_blur_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************

# Detect Edges
'''
To detect the edges in an image, you can use the Canny() method of cv2 which implements the Canny edge detector.
The Canny edge detector is also known as the optimal detector.
'''
edge_img = cv2.Canny(img, 100, 200)
# cv2.imshow('edges', edge_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************

# Grayscale
gray_img = cv2.cvtColor(resize_img_1, cv2.COLOR_BGR2GRAY)
# cv2.imshow('BW', gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************

# Image Shearing in X-axis
M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
sheared_img = cv2.warpPerspective(img, M, (int(ht*1.5), int(wd*1.5)))
# cv2.imshow('img', sheared_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************

# Find Contours

gray_img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, thresh = cv2.threshold(gray_img_0, 127, 255, 0)
img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, img_contours, -1, (0, 255, 0))
# cv2.imshow('Image Contours', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ***************************************************************************
