import cv2
from cv2 import blur
import numpy as np

ori_img = cv2.imread('test_images\solidWhiteRight.jpg')
gray_img = cv2.cvtColor(ori_img,cv2.COLOR_RGB2GRAY)
blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
canny = cv2.Canny(blur_img,50,150)
cv2.imshow('canny',canny)
cv2.waitKey(0)
