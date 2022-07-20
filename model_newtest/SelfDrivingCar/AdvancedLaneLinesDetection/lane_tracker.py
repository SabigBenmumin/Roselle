import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import pickle
import re
import re
import math


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)',s)]

def sort_nicely(l):
    l.sort(key=alphanum_key)

def plot_images(data, layout='row', cols=2, figsize=(20, 12)):
    rows = math.ceil(len(data) / cols)
    f ,ax = plt.subplots(figsize=figsize)
    if layout == 'row':
        for idx, d in enumerate(data):
            img, title = d

            plt.subplot(rows, cols, idx+1)
            plt.title(title, fontsize=20)
            plt.axis('off')
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')

            elif len(img.shape) == 3:
                plt.imshow(img)

    elif layout == 'col':
        counter = 0
        for r in range(rows):
            for c in range(cols):
                img, title = data[r + rows*c]
                nb_channels = len(img.shape)

                plt.subplot(rows, cols, counter+1)
                plt.title(title, fontsize=20)
                plt.axis('off')
                if len(img.shape) == 2:
                    plt.imshow(img, cmap='gray')

                elif len(img.shape) == 3:
                    plt.imshow(img)

                counter += 1

    return ax

def capture_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)

    print('Starting fram capture...')

    count = 0
    success = True
    while success:
        success, frame = cap.read()
        cv2.imwrite(frames_dir + 'fram{:02}.jpg'.format(count), frame)
        count += 1
    print('Completed!')

test_img_paths = glob.glob('test_images/test*.jpg')
sort_nicely(test_img_paths)

video1 = glob.glob('video_frames/frame*.jpg')
sort_nicely(video1)

video2 = glob.glob('video_frames_1/frame*.jpg')
sort_nicely(video2)


plot_demo = [1, 2, 3, 4, 5, 6, 7, 8]

def calibrate_camera():
    imgpaths = glob.glob('camera_cal/calibration*.jpg')
    sort_nicely(img_paths)

    image = cv2.imread(imgpaths[0])
    imshape = image.shape[:2]

    plt.imshow(image)
    plt.show()
    print('Image shape: {}'.format(image.shape))

    print()
    print('Calibrating the camera...')
    print()

    objpoints = []
    imgpoints = []

    nx = 9
    ny = 6

    objp = np.zeros([ny*nx, 3], dtype=np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    for idx, imgpath in enumerate(imgpaths):
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ret, corners = cv2.drawShessboardCorners(img, (nx, ny), corners, ret)

        imgpoints.append(corners)
        objpoints.append(objp)

        cv2.imshow('img',img)
        cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrate_camera(objpoints, imgpoints , imshape[::-1], None, None)
    print('Calibration complete!')
    cv2.destroyAllWindows()
    return mtx, dist

if os.path.exists('camera_calib.p'):
    with open('camera_calib.p', mode = 'rb') as f:
        data = pickle.load(f)
        mtx, dist = data['mtx'],data['dist']
        print('Loaded the saved camera calibration matrix & dist coefficients!')





