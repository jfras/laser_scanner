#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from laser_scanner_utils import *
from scipy.optimize import curve_fit
from threading import Thread
import sys
import copy

import open3d as o3d


threshold = 90
threshold = 20
radius = 50
rot_step = 0.0623/2
rot_step = 0.031415927*1.04
rot_step = 0.031415927
rot_step = 0.031415927*0.8
rot_step = -2*np.pi/250.0/4
rot_step = -2*np.pi/(160*4)
rot_step = 4*np.pi/(160*4)

frames_per_scan_step = 1
refine = False
refine = True
movie_name='miarka.avi'
print ("asdasdasdasdas")
if len(sys.argv)>1:
    movie_name = sys.argv[1]

def fit_square(y, idx, results, i):

    # print ('starting square fit')
    n = len(y)          
    results[i] = -1
    if n <=3:
        return
        # print ('asd')
    x = range(idx, idx+n)
    enfit = np.polyfit(x, y, 2)
    # print(enfit)
    results[i] = -enfit[1]/enfit[0] /2
    # print (idx, results[i])
    # results[]


def fit_gauss(y, idx, results, i):

    n = len(y)          
    results[i] = -1
    if n <=3:
        return
    x = range(idx, idx+n)

    def _1gaussian(x, amp1,cen1,sigma1):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))
    try:
        popt_gauss, pcov_gauss = scipy.optimize.curve_fit(_1gaussian, x, y, p0 = [600, idx+n/2, 4])
        # print(popt_gauss)
        # print(pcov_gauss)
        results[i] = popt_gauss[1]
    except Exception as e:
        print(e)
    # return popt_gauss[1]


w  = 300

camera_matrix = np.load('calibration_mtx.npy')
dist_coeffs = np.load('calibration_dist.npy')
plate_calib = np.load('plate_calibration.npy')
plate_point = plate_calib[0,:]
plate_normal = plate_calib[1,:]

# f = camera_matrix[0,0]


def spass(arg):
    pass

cv2.namedWindow('mask')
cv2.createTrackbar('threshold','mask', threshold, 250, spass)
# cv2.createTrackbar('hl','mask', 117, 255, spass)
# cv2.createTrackbar('sl','mask', 70, 255,  spass)
# cv2.createTrackbar('vl','mask', 110, 255, spass)
# cv2.createTrackbar('hh','mask', 203, 255, spass)
# cv2.createTrackbar('sh','mask', 255, 255, spass)
# cv2.createTrackbar('vh','mask', 255, 255, spass)

laser_point = np.load('laser_plane_point.npy')
laser_normal = np.load('laser_plane_normal.npy')

def imshow(window_name, frame):
    # cv2.imshow(window_name, cv2.resize(frame,(int(1920/2), int(1080/2))))
    cv2.imshow(window_name, frame)

class Scanner():

    def __init__(self):
        self.cap = cv2.VideoCapture(movie_name)
        # ret1, frame1 = self.cap.read()
        # ret1, frame1 = self.cap.read()
        
    def get_two_frames(self):
        if not self.cap.isOpened():
            print("video problem")
            exit(-1)

        # ret1, frame1 = self.cap.read()
        ret1, frame1 = self.cap.read()
        ret2, frame2 = self.cap.read()
        # imshow('f1', frame1)
        # imshow('f2', frame2)
        # cv2.waitKey(0)

        if not (ret1 and ret2):
            print("all over again")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return ret1 and ret2, frame1, frame2


    def fridi(self):
        mpl.use('tkagg')

        fig = plt.figure()
        ax = p3.Axes3D(fig)
        threedeplot = ax.scatter([0],[0],[0], s=1)
        ax.set_xlim3d([-50.0, 50.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-50.0, 50.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0, 50.0])
        ax.set_zlabel('Z')
        ax.set_title('3D Test')      

        # plt.ion()
        # plt.show()
        points_3d = []
        colors = []
        angle = -rot_step
        open(movie_name.split('.')[0]+".obj", 'w')
        while True:
            for i in range(frames_per_scan_step):
                ret, f1, f2 = self.get_two_frames()
                if not ret:
                    break
                angle+=rot_step

            if not ret:
                print("no frame received")
                break
            frame = cv2.absdiff(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY))
            # frame = cv2.undistort(frame,camera_matrix, dist_coeffs)
            if refine:
                frame = cv2.blur(frame, (1,2))  
            h, w = frame.shape

            max_values = frame.max(axis = 0)
            max_indexes = np.argmax(frame, axis = 0)
            
            # refine = False
            if refine:
                gauss_w = 3
                threads = [] 
                max_indexes_refined = [-1] * w
                for i in range(w):
                    if max_values[i] > cv2.getTrackbarPos('threshold','mask'):
                        threads.append(Thread(target=fit_square, args=(frame[max_indexes[i]-gauss_w:max_indexes[i]+gauss_w+1, i], max_indexes[i]-gauss_w, max_indexes_refined, i)))
                        threads[-1].start()
                print(len(threads))
                for i in range(len(threads)):
                    threads[i].join()
                max_indexes = max_indexes_refined
                # max_indexes = [fit_gauss(frame[max_indexes[i]-gauss_w:max_indexes[i]+gauss_w+1, i], max_indexes[i]-gauss_w) if max_values[i] > threshold else -1 for i in range(w) ]

            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            raw_points = []
            colors = []
            # f1 = simplest_cb(f1,10)
            for i in range(w):
                if max_values[i] > cv2.getTrackbarPos('threshold','mask'):
                    if 0 <= int(max_indexes[i]) < h:
                        colors.append(copy.copy(f1[int(max_indexes[i]),i]))
                    # print (colors[-1])
                    # print(f1[max_indexes[i],1])
                    # print()
                        # cv2.circle(f2, (i, int(max_indexes[i])), 0, (255,255,0),  2)
                        # print(w, i, max_indexes[i])
                        # if int(max_indexes[i]) < 1080:
                        f2[ int(max_indexes[i]), i] = (255,255,0)

                    # points_3d.append(line_plane_intersection(laser_point, laser_normal, (i, max_indexes_refined[i]), camera_matrix))
                    raw_points.append(line_plane_intersection(laser_point, laser_normal, (i, max_indexes[i]), camera_matrix))
            rotated_points = put_on_plate(plate_point, plate_normal, raw_points, angle, radius, colors)
            
            # print(raw_points)*frames_per_scan_step
            # print(rotated_points)
            points_3d+=rotated_points
            # imshow('diff',frame)
            # imshow('f1',f1)
            # imshow('f2',f2)
            # print(colors[-1])

            effective_image = np.vstack((f2, f1, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)))
            width = int(effective_image.shape[1] * 0.3)
            height = int(effective_image.shape[0] * 0.3)
            dim = (width, height)
            effective_image = cv2.resize(effective_image, dim, interpolation = cv2.INTER_AREA)
            effective_image = cv2.flip(effective_image,0)
            effective_image = cv2.transpose(effective_image)
            cv2.imshow('window', effective_image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 27:
                exit()


            
        # plt.draw()
        # plt.pause(0.00001)
        
            print("writing file")
            with open(movie_name.split('.')[0]+".obj", 'a') as f:
                for point in rotated_points:
                    # print(color)
                    # print(*list(color))
                    f.write("v %f %f %f %d %d %d\n" % (*list(point),))

        print("writing done")

        pcd = o3d.geometry.PointCloud()
        pcd_pts = np.array(points_3d)
        pcd.points = o3d.utility.Vector3dVector(pcd_pts[:,0:3])
        pcd.colors = o3d.utility.Vector3dVector(pcd_pts[:,3:6]/256.0)
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=5)
        cl = pcd.voxel_down_sample(voxel_size=0.2)

        o3d.visualization.draw_geometries([cl])

        # xdata = np.array(points_3d)[:,0]
        # ydata = np.array(points_3d)[:,1]
        # zdata = np.array(points_3d)[:,2]
        # threedeplot._offsets3d = (xdata, ydata, zdata)
        #     # threedeplot.set_ydata(ydata)
        #     # threedeplot.set_zdata(zdata)

        # plt.show()
        # plt.pause(10)
        # key = cv2.waitKey(0)
            
        # if key == 27:
        #     exit()



if __name__ == '__main__':

    scanner = Scanner()
    scanner.fridi()

    # cv2.waitKey(0)
