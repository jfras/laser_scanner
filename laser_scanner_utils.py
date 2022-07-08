import numpy as np
import scipy.optimize

import functools

from math import *
from matplotlib import pyplot as plt
# from scipy import optimize
# import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pprint as pp
import cv2
import math
## color balance code ##

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        # print ("Lowval: ", low_val)
        # print ("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


def fit_plane(points):
    def plane(x, y, params):
        a = params[0]
        b = params[1]
        c = params[2]
        z = a*x + b*y + c
        return z

    def error(params, points):
        result = 0
        # print (points)
        for (x,y,z) in points:
            plane_z = plane(z, y, params)
            diff = abs(plane_z - z)
            result += diff**2
        return result

    def cross(a, b):
        return [a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]]

    # points = [(1.1,2.1,8.1),
    #         (3.2,4.2,8.0),
    #         (5.3,1.3,8.2),
    #         (3.4,2.4,8.3),
    #         (1.5,4.5,8.0)]

    fun = functools.partial(error, points=points)
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    point  = np.array([0.0, c, 0 ])
    normal = np.array(cross([1,a,0], [0,b,1]))
    
    print(point)
    print(normal)

    xs, ys, zs = zip(*points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)
    d = -point.dot(normal)
    xx, yy = np.meshgrid([-100,100], [-100,100])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2, color=[0,1,0])

    ax.set_xlim(-300,300)
    ax.set_ylim(-300,300)
    ax.set_zlim(  0,600)

    plt.show()
    return point, normal



def line_plane_intersection_points(ppts, pix, camera_matrix):
                    
    pp0, pp1, pp2 = ppts
    # print(ppts)
    #plane
    plane_point  = pp0
    plane_normal = np.cross(pp1-pp0, pp2-pp0)
    return line_plane_intersection(plane_point, plane_normal, pix, camera_matrix)

def line_plane_intersection(plane_point, plane_normal, pix, camera_matrix):
    cx, cy = camera_matrix[0:2,2]
    line_vec = np.array([pix[0]-cx, pix[1] - cy, camera_matrix[0,0]])
    line_point = np.array([0,0,0])

    # print(plane_point, line_point, plane_normal, line_vec)

    d = (plane_point - line_point).dot(plane_normal)/line_vec.dot(plane_normal)
    return line_point + d* line_vec




def rotate_about_axis(line_point, line_normal, points, angle):
    
    line_normal/=np.linalg.norm(line_normal)
    rotated = []
    for point in points:
        x = line_point+((point-line_point).dot(line_normal))*line_normal  # vector from line to point
        point_vec = point - x
        dist = np.linalg.norm(point_vec)
        point_vec /= dist
        rArr=np.cross(point_vec,np.array(line_normal)) #inplane vector

        rotated_point = -line_point + x + dist*(point_vec*cos(angle)+rArr*sin(angle))
        rotated.append(rotated_point)
    
    return rotated




def put_on_plate(plate_point, plate_normal, points, angle, radius, colors):
    # print (plate_point)
    plate_normal/=np.linalg.norm(plate_normal)
    rotated = []
    for point, color in zip(points, colors):
        closest_pt_on_axis = plate_point+((point-plate_point).dot(plate_normal))*plate_normal  # vector from line to point
        point_vec = closest_pt_on_axis - point
        x_dist = np.linalg.norm(point_vec)*np.sign(point_vec[2])
        # print (x_dist)
        if abs(x_dist)<=radius:
        # if True:
            z_vec = plate_point - closest_pt_on_axis 
            z_dist = np.linalg.norm(z_vec)*np.sign(z_vec[2])
            # print(z_dist)
            if z_dist>= 0:
                
            # if True:
                rotated_point = np.array([x_dist*cos(angle), x_dist*sin(angle), z_dist])

                rotated.append(list(rotated_point) + list(reversed(color)))
    
    return rotated






# https://stackoverflow.com/questions/15481242/python-optimize-leastsq-fitting-a-circle-to-3d-set-of-points


# Fitting a plane first
# let the affine plane be defined by two vectors, 
# the zero point P0 and the plane normal n0
# a point p is member of the plane if (p-p0).n0 = 0 

def distanceToPlane(p0,n0,p):
    return np.dot(np.array(n0),np.array(p)-np.array(p0))    

def residualsPlane(parameters,dataPoint):
    px,py,pz,theta,phi = parameters
    nx,ny,nz =sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)
    distances = [distanceToPlane([px,py,pz],[nx,ny,nz],[p[0],p[1],p[2]]) for p in dataPoint]
    return distances

def fit_circle_3d(points):
    # print (list(xs),list(ys),list(zs)) 
    # dataTupel=zip(xs,ys,zs) 
    xs = [i[0] for i in points]
    ys = [i[1] for i in points]
    zs = [i[2] for i in points]
    estimate = [np.mean(xs), np.mean(ys), np.mean(zs),0,0] # px,py,pz and zeta, phi
    #you may automize this by using the center of mass data
    # note that the normal vector is given in polar coordinates
    bestPlaneFitValues, ier = scipy.optimize.leastsq(residualsPlane, estimate, args=points)
    xF,yF,zF,tF,pF = bestPlaneFitValues
    normal = [sin(tF)*cos(pF),sin(tF)*sin(pF),cos(tF)]

    #projection of the estimate:

    line_vec = np.array(normal)
    line_point = np.array(estimate[0:3])
    plane_point = np.array([xF,yF,zF])

    # print(plane_point, line_point, plane_normal, line_vec)

    d = (plane_point - line_point).dot(line_vec)/line_vec.dot(line_vec)
    est_proj = line_point + d* line_vec
    print(est_proj)

    
    # est_dist = np.linalg.norm(np.array(estimate[0:3]) - np.array([xF,yF,zF]))
    # est_proj = np.array([estimate[0:3]]) - est_dist*np.array(normal)
    # point  = [xF,yF,zF]
    print("estimate, point")
    print(estimate, bestPlaneFitValues)
    # point  = [xF,yF,zF]
    # point  = [est_proj[0,0],est_proj[0,1],est_proj[0,2]]
    point  = est_proj
    print(point)

    # Fitting a circle inside the plane
    #creating two inplane vectors
    sArr=np.cross(np.array([1,0,0]),np.array(normal))#assuming that normal not parallel x!
    sArr=sArr/np.linalg.norm(sArr)
    rArr=np.cross(sArr,np.array(normal))
    rArr=rArr/np.linalg.norm(rArr)#should be normalized already, but anyhow


    def residualsCircle(parameters,dataPoint):
        r,s,Ri = parameters
        # Ri = min(abs(Ri), 150)
        planePointArr = s*sArr + r*rArr + np.array(point)
        print(s,r, Ri)
        distance = [ np.linalg.norm( planePointArr-np.array([x,y,z])) for x,y,z in dataPoint]
        res = [(Ri-dist) for dist in distance]
        ##############################################################
        # centerPointArr=sF*sArr + rF*rArr + np.array(point)
        synthetic=[list(planePointArr+ Ri*cos(phi)*rArr+Ri*sin(phi)*sArr) for phi in np.linspace(0, 2*pi,250)]
        [cxTupel,cyTupel,czTupel]=[ x for x in zip(*synthetic)]

        ### Plotting
        # d = -np.dot(np.array(point),np.array(normal))# dot product
        # # create x,y mesh
        # xx, yy = np.meshgrid(np.linspace(estimate[0]-100,estimate[0]+100,10), np.linspace(estimate[1]-100,estimate[1]+100,10))
        # # calculate corresponding z
        # # Note: does not work if normal vector is without z-component
        # z = (-normal[0]*xx - normal[1]*yy - d)/normal[2]

        # # plot the surface, data, and synthetic circle
        # fig = plt.figure()
        # ax = fig.add_subplot(211, projection='3d')
        # ax.scatter(xs, ys, zs, c='b', marker='o')
        # ax.plot_wireframe(xx,yy,z)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_xlim(estimate[0]-100,estimate[0]+100)
        # ax.set_ylim(estimate[1]-100,estimate[1]+100)
        # ax.set_zlim(estimate[2]-100,estimate[2]+100)
        # # bx = fig.add_subplot(212, projection='3d')
        # # bx.scatter(xs, ys, zs, c='b', marker='o')
        # ax.scatter(cxTupel,cyTupel,czTupel, c='r', marker='.')
        # # bx.set_xlabel('X Label')
        # # bx.set_ylabel('Y Label')
        # # bx.set_zlabel('Z Label')
        # plt.show()
        ##############################################################

        # print ()/
        # print ("circle approximation")
        # print (res)
        return res

    estimateCircle = [0, 0, 0] # px,py,pz and zeta, phi
    bestCircleFitValues, ier = scipy.optimize.leastsq(residualsCircle, estimateCircle, args=points)

    rF,sF,RiF = bestCircleFitValues
    print("bestCircleFitValues")
    print(bestCircleFitValues)

    # Synthetic Data
    centerPointArr=sF*sArr + rF*rArr + np.array(point)
    synthetic=[list(centerPointArr+ RiF*cos(phi)*rArr+RiF*sin(phi)*sArr) for phi in np.linspace(0, 2*pi,250)]
    [cxTupel,cyTupel,czTupel]=[ x for x in zip(*synthetic)]

    ### Plotting
    d = -np.dot(np.array(point),np.array(normal))# dot product
    # create x,y mesh
    xx, yy = np.meshgrid(np.linspace(estimate[0]-100,estimate[0]+100,10), np.linspace(estimate[1]-100,estimate[1]+100,10))
    # calculate corresponding z
    # Note: does not work if normal vector is without z-component
    z = (-normal[0]*xx - normal[1]*yy - d)/normal[2]

    print("centerPointArr, normal")
    print(centerPointArr, normal)
    # plot the surface, data, and synthetic circle
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(xs, ys, zs, c='b', marker='o')
    ax.plot_wireframe(xx,yy,z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(estimate[0]-100,estimate[0]+100)
    ax.set_ylim(estimate[1]-100,estimate[1]+100)
    ax.set_zlim(estimate[2]-100,estimate[2]+100)
    # bx = fig.add_subplot(212, projection='3d')
    # bx.scatter(xs, ys, zs, c='b', marker='o')
    ax.scatter(cxTupel,cyTupel,czTupel, c='r', marker='.')
    # bx.set_xlabel('X Label')
    # bx.set_ylabel('Y Label')
    # bx.set_zlabel('Z Label')
    plt.show()
    return centerPointArr, normal