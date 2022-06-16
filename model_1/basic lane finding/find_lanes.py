import cv2
import numpy as np

ori_img = cv2.imread('test_images\solidWhiteRight.jpg')
gray_img = cv2.cvtColor(ori_img,cv2.COLOR_RGB2GRAY)
blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
cv2.imshow('gray scale image',gray_img)
cv2.waitKey(0)
cv2.imshow('blur',blur_img)
cv2.waitKey(0)
