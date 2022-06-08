import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from collections import deque
from lane_detection import color_frame_pipeline

ori_img = cv2.imread('solidWhiteRight.jpg')
cv2.imshow('original img',ori_img)
cv2.waitKey(0)
bgr_img = cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
cv2.imshow('test rgb2bgr',bgr_img)
cv2.waitKey(0)
out_image = color_frame_pipeline([bgr_img], solid_lines=True)
#cv2.imshow(cv2.cvtColor(out_image,cv2.COLOR_RGB2BGR))
plt.imshow(out_image)
plt.waitforbuttonpress()
cv2.waitKey(0)