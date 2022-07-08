#!/usr/bin/env python

import os
import datetime
from cv2 import aruco
import cv2
import urllib 
import numpy as np
import glob
import requests

# dispw = 

url = 'http://192.168.1.120:8000/stream.mjpg'
stream = requests.get(url, stream=True)
raw_data = b''

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('*.jpg')

images = []
counter = 0
markers = np.zeros((int(1080/2),int(1920/2),3), np.uint8)
while True:
    raw_data += stream.raw.read(1024)
    a = raw_data.find(b'\xff\xd8')
    b = raw_data.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = raw_data[a:b+2]
        raw_data = raw_data[b+2:]
        counter +=1
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        original = i
        i = cv2.resize(i, (int(1920/2),int(1080/2)), interpolation = cv2.INTER_AREA)
        if cv2.waitKey(1) == 27:
            break

        gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        if  counter%5 == 0:



            print(len(corners))
            if len(corners)>10:
                markers = aruco.drawDetectedMarkers(markers, corners, ids)
                images.append(original)
        i = aruco.drawDetectedMarkers(i, corners, ids)



        cv2.imshow('img',cv2.rotate(i,cv2.ROTATE_90_CLOCKWISE))
        cv2.imshow('markers',cv2.rotate(markers,cv2.ROTATE_90_CLOCKWISE))

dir_name = "camera_calibration_images_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(dir_name)
for im, i in zip(images, range(len(images))):
    print ('writing image %d' % i)
    cv2.imwrite(dir_name+'/img_%d.png' % i, im)

cv2.destroyAllWindows()





