import cv2
import numpy as np
import matplotlib.pyplot as plt


# function สำหรับการเปลี่ยนรูป rgb เป็นรูปขาวดำ โดยที่ในฟังชั่นมีการเบลอรูปเพื่อลบ object ที่ไม่ต้องการออก
def image2canny(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    canny = cv2.Canny(blur_img,50,150)
    return canny

# function นี้มีไว้สำหรับสร้างรูปสามเหลี่ยมที่คาดว่าจะมีเลนในพื้นที่นี้
def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([[(125,height),(850,height),(482,302)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img,mask) 
    return masked_img

img = cv2.imread('test_images\solidWhiteRight.jpg')
lane_image = np.copy(img)
canny = image2canny(lane_image)
cropped_img = region_of_interest(canny)

plt.imshow(canny)
plt.show()
cv2.imshow("cropped image",cropped_img)
cv2.waitKey(0)