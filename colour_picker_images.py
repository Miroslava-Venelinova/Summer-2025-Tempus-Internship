import cv2
import numpy as np

img = cv2.imread('img_1.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2

mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

red_result = cv2.bitwise_and(img, img, mask=mask_red)

blue_result = cv2.bitwise_and(img, img, mask=mask_blue)

cv2.imshow('Original', img)
cv2.imshow('Red regions', red_result)
cv2.imshow('Blue regions', blue_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
