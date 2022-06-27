import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(frame, line_parameters):
    slope, intercept = line_parameters
    y1 = frame.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope) 
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1 ,x2 ,y2])

def average_slope_intercept(frame,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1 ,y1 , x2 , y2  =line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(frame, left_fit_average)
    right_line = make_coordinates(frame, right_fit_average)
    return np.array([left_line, right_line])


# function สำหรับการเปลี่ยนรูป rgb เป็นรูปขาวดำ โดยที่ในฟังชั่นมีการเบลอรูปเพื่อลบ object ที่ไม่ต้องการออก
def image2canny(frame):
    gray_img = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    canny = cv2.Canny(blur_img,50,150)
    return canny
    
def display_lines(frame,lines):
    line_img = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1 ,y1),(x2, y2), (255, 0, 0), 10)  
    return line_img


# function นี้มีไว้สำหรับสร้างรูปสามเหลี่ยมที่คาดว่าจะมีเลนในพื้นที่นี้
def region_of_interest(frame):
    height = frame.shape[0]
    polygons = np.array([[(125,height),(850,height),(482,302)]])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(frame,mask) 
    return masked_img

cap = cv2.VideoCapture('solidWhiteRight.mp4')

while(cap.read()):
    ref,frame = cap.read()
    roi = frame[:1080,0:1920]

    
    canny_img = image2canny(frame)
    cropped_img = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
    averaged_line = average_slope_intercept(frame,lines)
    line_img = display_lines(frame,averaged_line)
    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    
    cv2.imshow("combo",roi)

    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()