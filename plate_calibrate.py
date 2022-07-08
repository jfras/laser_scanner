#!/usr/bin/env python

import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math

from laser_scanner_utils import *

w  = 300

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
checkboard_len = 89.0 
board = aruco.CharucoBoard_create(7, 7, checkboard_len/7, .8*checkboard_len/7, aruco_dict)

camera_matrix = np.load('calibration_mtx.npy')
dist_coeffs = np.load('calibration_dist.npy')

# f = camera_matrix[0,0]


# def spass(arg):
#     pass

# cv2.namedWindow('mask')
# cv2.createTrackbar('hl','mask', 117, 255, spass)
# cv2.createTrackbar('sl','mask', 70, 255,  spass)
# cv2.createTrackbar('vl','mask', 110, 255, spass)
# cv2.createTrackbar('hh','mask', 203, 255, spass)
# cv2.createTrackbar('sh','mask', 255, 255, spass)
# cv2.createTrackbar('vh','mask', 255, 255, spass)




def calibrate_plate(video):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    laser_points = []
    cap = cv2.VideoCapture(video)
    axes = np.zeros((1080,1920, 3),dtype=np.uint8)
    points = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frame = simplest_cb(frame, 10)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        # frame = cv2.flip(frame,0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(res2[1], res2[2], board, camera_matrix, dist_coeffs, None, None )
            if not retval:
                print('no marker detected')
                continue
            # target rotation matrix
            rot_mat = cv2.Rodrigues(rvec)[0]

            # target corners in 3d
            pt0_3d = tvec

            points.append(tvec[0:3].T[0])

            # targed corners in camera coordinates
            pt0_3d_camera = camera_matrix.dot(pt0_3d).T[0]

            # target corners on the image
            pt0 = pt0_3d_camera[0:2]/pt0_3d_camera[2]

            cv2.circle(axes, (int(pt0[0]), int(pt0[1])),2,(0,255,0),2)

            cv2.imshow('frame',frame)
            cv2.imshow('trace', axes)


            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 27:
                exit()

    cp, n = fit_circle_3d(points)
    return cp, n

if __name__ == '__main__':

    video = "plate_calibrate.avi"

    point, normal = calibrate_plate(video)
    plate_calibration = np.array([point, normal])
    print('final calibration result:')
    print(plate_calibration)
    np.save('plate_calibration', plate_calibration)
    print("data_saved, press any key to exit")
    cv2.waitKey(0)
