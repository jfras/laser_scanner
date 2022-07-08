#!/usr/bin/env python

import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
import copy

from laser_scanner_utils import *

w  = 300

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
checkboard_len = 89.0 
board = aruco.CharucoBoard_create(7, 7, checkboard_len/7, .8*checkboard_len/7, aruco_dict)

camera_matrix = np.load('calibration_mtx.npy')
dist_coeffs = np.load('calibration_dist.npy')

# f = camera_matrix[0,0]

def spass(arg):
    pass

cv2.namedWindow('mask')
cv2.createTrackbar('h','mask', 120, 255, spass)
cv2.createTrackbar('threshold','mask', 98, 255,  spass)
# cv2.createTrackbar('vl','mask', 157, 255, spass)
# cv2.createTrackbar('hh','mask', 144, 255, spass)
# cv2.createTrackbar('sh','mask', 228, 255, spass)
# cv2.createTrackbar('vh','mask', 255, 255, spass)

def calibrate_laser_plane(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    laser_points = []
    for number, im in enumerate(images):
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        corrected_frame = cv2.undistort(cv2.imread(im),camera_matrix, dist_coeffs)
        # cv2.imshow('frame', frame)
        # cv2.imshow('corrected_frame', corrected_frame)
        # cv2.waitKey(0)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # # cv2.imshow('board_image_fixed',board_colors_fixed)
        # cv2.imshow('board_image_h',frame[:,:,0])
        # cv2.imshow('board_image_s',frame[:,:,1])
        # cv2.imshow('board_image_v',frame[:,:,2])
        # cv2.imshow('board_image_h-s-v',1/255/255/255*frame[:,:,0]*frame[:,:,1]*frame[:,:,2])
        
        # while True:
        #     gt = cv2.getTrackbarPos
        #     h,s,v,H,S,V = gt('hl', 'mask'),gt('sl', 'mask'),gt('vl', 'mask'),gt('hh', 'mask'),gt('sh', 'mask'),gt('vh', 'mask')
        #     print(h,s,v,H,S,V)
        #     laser_mask = cv2.inRange(frame, np.array([h,s,v]), np.array([H,S,V])) | cv2.inRange(frame, np.array([h,s,v]), np.array([H,S,V]))
        #     cv2.imshow('mask', laser_mask)
        #     cv2.waitKey(1)

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
                continue 
            # print (rvec)
            # target rotation matrix
            rot_mat = cv2.Rodrigues(rvec)[0]

            # target corners in 3d
            pt0_3d = tvec
            pt1_3d = tvec + rot_mat.dot(np.array([[0],[89],[0]]))
            pt2_3d = tvec + rot_mat.dot(np.array([[89],[89],[0]]))
            pt3_3d = tvec + rot_mat.dot(np.array([[89],[0],[0]]))

            # targed corners in camera coordinates
            pt0_3d_camera = camera_matrix.dot(pt0_3d).T[0]
            pt1_3d_camera = camera_matrix.dot(pt1_3d).T[0]
            pt2_3d_camera = camera_matrix.dot(pt2_3d).T[0]
            pt3_3d_camera = camera_matrix.dot(pt3_3d).T[0]

            # target corners on the image
            pt0 = pt0_3d_camera[0:2]/pt0_3d_camera[2]
            pt1 = pt1_3d_camera[0:2]/pt1_3d_camera[2]
            pt2 = pt2_3d_camera[0:2]/pt2_3d_camera[2]
            pt3 = pt3_3d_camera[0:2]/pt3_3d_camera[2]

            # points for perspective correction
            pts1 = np.float32([pt0, pt1, pt2, pt3]) 
            pts2 = np.float32([[0, 0], [w, 0], [w, w], [0, w]]) 
            
            # perspective correction
            matrix = cv2.getPerspectiveTransform(pts1, pts2) 
            m_inv = cv2.getPerspectiveTransform(pts2, pts1)

            # corrected image
            board_image = cv2.warpPerspective(corrected_frame, matrix, (w, w) ) 

            # board_image = cv2.GaussianBlur(board_image, (15,15), 0,0)
            # board_image = simplest_cb(board_image, 1)
            
            board_hsv = cv2.cvtColor(board_image, cv2.COLOR_RGB2HSV)

            

            # h,s,v,H,S,V =  117, 70, 110, 203, 255, 255
            # h,s,v,H,S,V =  91, 134, 91, 120, 189, 222
            gt = cv2.getTrackbarPos
            # h,s,v,H,S,V = gt('hl', 'mask'),gt('sl', 'mask'),gt('vl', 'mask'),gt('hh', 'mask'),gt('sh', 'mask'),gt('vh', 'mask')
            
            # hsv filtering of the laser beam on the target board
            while True:
                gt = cv2.getTrackbarPos
                h, thres = gt('h', 'mask'),gt('threshold', 'mask')
                print(h,thres)
                # laser_mask = cv2.inRange(board_hsv, np.array([h,s,v]), np.array([H,S,V])) 
                # cv2.imshow('mask', laser_mask)
                color = np.maximum(0, 1 - 4*np.minimum((abs(board_hsv[:,:,0] - h)),abs(255 - abs(board_hsv[:,:,0] - h))).astype(float)/255.0 )*board_hsv[:,:,2]/255.0
                ret ,laser_mask = cv2.threshold(color, thres/255.0, 255, cv2.THRESH_BINARY)
                cv2.imshow('color',color)
                # cv2.imshow('mask22',color2)
                # cv2.imshow('h',board_hsv[:,:,0])
                # cv2.imshow('s',board_hsv[:,:,1])
                # cv2.imshow('v',board_hsv[:,:,2])
                

                # laser_mask = cv2.inRange(board_hsv, np.array([h,s,v]), np.array([H,S,V]))
                cv2.imshow('mask', laser_mask)
                # line fitting
                laser_pixels = np.array(laser_mask.nonzero()).T
                if len(laser_pixels)> 10:
                    laser_pixels[:, 0], laser_pixels[:, 1] = laser_pixels[:, 1], laser_pixels[:, 0].copy()
                    vx, vy, cx, cy = cv2.fitLine(np.float32(laser_pixels), cv2.DIST_HUBER, 0, 0.01, 0.01)
                    line_pt1 = (int(cx-vx*w/2), int(cy-vy*w/2))
                    line_pt2 = (int(cx+vx*w/2), int(cy+vy*w/2))


                    # line perspective wrap
                    tpt1 = m_inv.dot(np.array([[line_pt1[0]],[line_pt1[1]],[1]])).T[0]
                    tpt2 = m_inv.dot(np.array([[line_pt2[0]],[line_pt2[1]],[1]])).T[0]

                    tpt1 = tpt1[0:2]/ tpt1[2]
                    tpt2 = tpt2[0:2]/ tpt2[2]

                    # projection of the line ends on the target plane
                    laser_point_3d_1 = line_plane_intersection_points([pt0_3d.T[0], pt1_3d.T[0], pt2_3d.T[0]], tpt1, camera_matrix)
                    laser_point_3d_2 = line_plane_intersection_points([pt0_3d.T[0], pt1_3d.T[0], pt2_3d.T[0]], tpt2, camera_matrix)

                    # laser_point_check_1 = camera_matrix.dot(laser_point_3d_1)
                    # laser_point_check_1 = laser_point_check_1[0:2]/laser_point_check_1[2]
                    
                    # laser_point_check_2 = camera_matrix.dot(laser_point_3d_2)
                    # laser_point_check_2 = laser_point_check_2[0:2]/laser_point_check_2[2]
                    
                    # cv2.line(frame, (int(laser_point_check_1[0]),int(laser_point_check_1[1])), (int(laser_point_check_2[0]),int(laser_point_check_2[1])) , (0, 255, 255), 3)
                    # # cv2.circle(frame, (int(laser_point_check_1[0]), int(laser_point_check_1[1])),int(laser_point_3d[2]%100),(0,255,0),5)
                    # cv2.imshow('name',cv2.resize(frame, (int(1920/2), int(1080/2))))
                    # cv2.imshow('board_image',board_image)
                    # print(laser_points)

                    frame_w_laser = copy.copy(corrected_frame)
                    cv2.line(frame_w_laser, (int(tpt1[0]),int(tpt1[1])), (int(tpt2[0]),int(tpt2[1])) , (0, 255, 255), 4)
                    cv2.imshow('frame',frame_w_laser)
                    cv2.imshow('board_image',board_image)

                key = cv2.waitKey(1)
                if key == ord(' '):
                    break
            
            
            laser_points.append(laser_point_3d_1.tolist())
            laser_points.append(laser_point_3d_2.tolist())

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 27:
                exit()
    
    # fitting plane to the points and returining
    return fit_plane(laser_points)

    # import matplotlib.pyplot as plt
    # print("plotting")
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # # Data for a three-dimensional line
    # # zline = np.linspace(0, 15, 1000)
    # # xline = np.sin(zline)
    # # yline = np.cos(zline)
    # # ax.plot3D(xline, yline, zline, 'gray')

    # # Data for three-dimensional scattered points
    # print (np.array(laser_points))
    # xdata = np.array(laser_points)[:,0]
    # ydata = np.array(laser_points)[:,1]
    # zdata = np.array(laser_points)[:,2]
    # print(xdata)
    # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    # plt.show()
    # print("plotting done")
    # key = cv2.waitKey(0)
        
    # # if key == 27:
    # #     exit()



if __name__ == '__main__':

    datadir = "laser_calibrate/"
    images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png") ])
    order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    images = images[order]

    point, normal = calibrate_laser_plane(images)

    np.save('laser_plane_point',  point)
    np.save('laser_plane_normal', normal)

    print("data_saved, press any key to exit")
    cv2.waitKey(0)
