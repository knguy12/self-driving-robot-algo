import cv2
import numpy as np

#Converts to grayscale to help with computation
#Gaussian blur to smooth image and reduce noise (can cause false edges)
#Canny to perform derivative on fuction to find large gradients and rapid change in brightness
def canny(image):
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grayScale, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

image = cv2.imread('road.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)

cv2.imshow('result', canny)
cv2.waitKey(0)