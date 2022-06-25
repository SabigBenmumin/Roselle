from tkinter.messagebox import NO
import cv2
import numpy as np
import matplotlib.pyplot as plt


# function สำหรับการเปลี่ยนรูป rgb เป็นรูปขาวดำ โดยที่ในฟังชั่นมีการเบลอรูปเพื่อลบ object ที่ไม่ต้องการออก
def image2canny(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    canny = cv2.Canny(blur_img,50,150)
    return canny
    
def display_lines(img,lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1 , y1, x2,y2 = line.reshape(4)
            cv2.line(line_img, (x1,y1),(x2,y2), (255,0,0),10)
    return line_img


# function นี้มีไว้สำหรับสร้างรูปสามเหลี่ยมที่คาดว่าจะมีเลนในพื้นที่นี้
def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([[(125,height),(850,height),(482,302)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img,mask) 
    return masked_img

img = cv2.imread('test_images\solidWhiteRight.jpg')
lane_img = np.copy(img)
canny = image2canny(lane_img)
cropped_img = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
line_img = display_lines(lane_img,lines)
combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)

plt.imshow(canny)
plt.show()
cv2.imshow("cropped image",cropped_img)
cv2.waitKey(0)
cv2.imshow("combo",combo_img)
cv2.waitKey(0)
