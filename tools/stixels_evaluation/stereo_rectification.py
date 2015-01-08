#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os, os.path
from collections import namedtuple

from calibration_pb2 import StereoCameraCalibration
#CameraCalibration, \
#RotationMatrix, TranslationVector, Pose, \
#CameraInternalParameters
#RadialDistortion, TangentialDistortion, \

from google import protobuf

import numpy
from numpy.linalg import inv as inverse
from numpy.linalg import norm

try:
    import Image #, ImageDraw
except ImportError, exc:
    raise SystemExit("PIL must be installed to run this application")
    
        
Point = namedtuple("Point", "x y")

def rotation_to_array(R):
    
    return numpy.matrix([[R.r11, R.r12, R.r13], \
                          [R.r21, R.r22, R.r23], \
                            [R.r31, R.r32, R.r33]])

def translation_to_array(T):
    return numpy.matrix([T.t1, T.t2, T.t3])

def camera_internal_parameters_to_array(K):
    return  numpy.matrix([[K.k11, K.k12, K.k13], \
                          [K.k21, K.k22, K.k23], \
                            [K.k31, K.k32, K.k33]])
    
def get_focal_point(pose):
    """
    Return focal point of camera
    """
    pose_t = translation_to_array(pose.translation)
    pose_R = rotation_to_array(pose.rotation)
    #print("pose_R.transpose().shape() == ", pose_R.transpose().shape)
    #print("pose_t == ", pose_t)
    #print("pose_t.transpose().shape() == ", pose_t.transpose().shape)
    #return -pose_R.transpose() * pose_t
    return numpy.dot(-pose_R.transpose(), pose_t.transpose())

def normalize(vector):
    return vector/norm(vector)
    
cross = numpy.cross

def gridify(quad, steps):
    """
    Divides a box (`quad`) into multiple boxes (``steps x steps``).
    
    >>> list(griddify((0, 0, 500, 500), 2))
    [(0, 0, 250, 250), (250, 0, 500, 250), (0, 250, 250, 500), (250, 250, 500, 500)]
    
    From: https://bitbucket.org/olt/mapproxy/src/142/mapproxy/core/image.py#cl-553
    """
    w = quad[2]-quad[0]
    h = quad[3]-quad[1]
    x_step = w / float(steps)
    y_step = h / float(steps)
    
    y = quad[1]
    for _ in range(steps):
        x = quad[0]
        for _ in range(steps):
            yield (int(x), int(y), int(x+x_step), int(y+y_step))
            x += x_step
        y += y_step
       
    return
        
       
class StereoRectification:
    """
    This class is based on doppia::CpuPreprocessor
    We assume that input images are already undistorted
    """
    
    
    def __init__(self, calibration_filename):

        if not os.path.exists(calibration_filename):
            raise ValueError("Could not find the indicated calibration file %s" % calibration_filename)

        self.stereo_calibration = StereoCameraCalibration()
        
        with open(calibration_filename, "r") as calibration_file:
            #self.stereo_calibration.ParseFromString(calibration_file.read())
            protobuf.text_format.Merge(calibration_file.read(), self.stereo_calibration)
        
        self._compute_rectification_homographies()
        return
        
    
    def _compute_rectification_homographies(self):
        """
        This method a C++ to Python translation of 
        void doppia::CpuPreprocessor::compute_rectification_homographies(...)
        """

        assert self.stereo_calibration.IsInitialized()    
        
        # rectification: http://profs.sci.univr.it/~fusiello/demo/rect/

        # in this function 1 refers to left and 2 refers to right
        left_camera_calibration = self.stereo_calibration.left_camera
        right_camera_calibration = self.stereo_calibration.right_camera
        
        K1 = left_camera_calibration.internal_parameters
        K1 = camera_internal_parameters_to_array(K1)
        R1 = left_camera_calibration.pose.rotation
        R1 = rotation_to_array(R1)

        K2 = right_camera_calibration.internal_parameters
        K2 = camera_internal_parameters_to_array(K2)                    
        R2 = right_camera_calibration.pose.rotation
        R2 = rotation_to_array(R2)

        # optical centers (unchanged)
        # focal_center1 and focal_center2 are Vector3f
        focal_center1 = get_focal_point(left_camera_calibration.pose)
        focal_center2 = get_focal_point(right_camera_calibration.pose)

        # Q1 and Q2 are Matrix3f 
        Q1 = inverse(K1 * R1)
        Q2 = inverse(K2 * R2)
    
    
        # new x axis (direction of the baseline)
        v1 = normalize(focal_center2 - focal_center1) # Normalise by dividing through by the magnitude.
        v1 = v1.transpose()
        
        # new y axes (old y)
        #v2 = (R1.row(1) + R2.row(1))*0.5
        v2 = (R1[1,:] +  R2[1,:])*0.5
        v2 = normalize(v2)
        # new z axes (orthogonal to baseline and y)        
        v3 = cross(v1,v2)
        v3 = normalize(v3)
        
        v2 = cross(v3,v1)
        v2 = normalize(v2)
    
        # new extrinsic parameters
        R1[0, :] = v1 # R1.row(0)
        R1[1, :] = v2
        R1[2, :] = v3
        
        K1 = (K1 + K2) * 0.5
        K2 = K1
    
        # Q1new and Q2new are Matrix3f
        Q1new = K1 * R1
        Q2new = K1 * R1
    
        Q1new = inverse(Q1new * Q1)
        Q2new = inverse(Q2new * Q2)
        

        # do centering
        # FIXME why 5 iterations ?
        for i in range(5):
            # new intrinsic parameters (arbitrary)
            K1 = (K1 + K2) * 0.5
            K2 = K1
    
            K1[0,2] += (Q1new[0, 2] + Q2new[0, 2]) / 4
            K2[0,2] += (Q1new[0, 2] + Q2new[0, 2]) / 4
    
            Q1new = K1 * R1
            Q2new = K2 * R1
    
            Q1new = inverse(Q1new * Q1)
            Q2new = inverse(Q2new * Q2)
      
      
        use_terrible_hack = True
        #use_terrible_hack = False
        if use_terrible_hack:
            print("!"*10 + "USING 30 pixels offset in _compute_rectification_homographies" + "!"*50)
            x_offset = 30
            Q1new[0, 2] += x_offset # [pixels]
            Q2new[0, 2] -= x_offset # [pixels]

      
        # left, right rectified MetricCamera(K, radial_distortion_parameters, R, t)
        #metric_camera1_rectified.set(K1, zeros< double >(3), R1, -R1.transpose() * focal_center1)
        #metric_camera2_rectified.set(K2, zeros< double >(3), R1, -R1.transpose() * focal_center2)
    
        self.left_rectification_homography = Q1new
        self.right_rectification_homography = Q2new
    
        self.left_rectification_inverse_homography = inverse(self.left_rectification_homography)
        self.right_rectification_inverse_homography = inverse(self.right_rectification_homography)
    
    
        self.left_K_rectified = K1
        self.right_K_rectified = K2
                
        return
        
        
    def rectify_left_image(self, left_image):
        """
        Given an undistorted left image, 
        will provide the rectified image
        """
        
        image_size = left_image.size

        # we will use Image.transform to warp the image
        # http://www.pythonware.com/library/pil/handbook/image.htm

        data = []
        image_box = (0,0, image_size[0], image_size[1])
        steps = 50 #100
        H = self.left_rectification_homography
        for quad in gridify(image_box, steps):
            top_left = self.rectify_point_xy(H, quad[0], quad[1])
            bottom_left = self.rectify_point_xy(H, quad[0], quad[3])
            bottom_right = self.rectify_point_xy(H, quad[2], quad[3])
            top_right = self.rectify_point_xy(H, quad[2], quad[1])
            dst_quad = quad
            src_quad = tuple(
            list(top_left) + list(bottom_left) + 
            list(bottom_right) + list(top_right) )
            data.append( (dst_quad, src_quad) )
            
        rectified_image = left_image.transform(image_size, Image.MESH, data, Image.BILINEAR)
                
        return rectified_image
    
    def rectify_point(self, H, point):
        """
        H is an homography matrix
        Given a point in the left image, 
        will provide the corresponding rectified point.
        point is expected to have .x and .y fields
        """
        
        rectified_point_x = H[0,0] * point.x + H[0,1] * point.y + H[0,2]
        rectified_point_y = H[1,0] * point.x + H[1,1] * point.y + H[1,2]
        w = H[2,0] * point.x + H[2,1] * point.y + H[2,2]
        rectified_point_x /= w
        rectified_point_y /= w
    
        rectified_point = Point(rectified_point_x, rectified_point_y)
        return rectified_point


    def rectify_point_xy(self, H, point_x, point_y):
        return self.rectify_point(H, Point(point_x, point_y))
                
    def rectify_left_point(self, point):
        """
        Given a point in the left image, 
        will provide the corresponding rectified point.
        point is expected to have .x and .y fields
        """
        
        H = self.left_rectification_inverse_homography
        return self.rectify_point(H, point)
        
    def rectify_left_bounding_box(self, bounding_box):
        """
        Bounding box is expected to be a tuple with four elements defining
        the top left and bottom left corners 
        Example: bounding_box = (221, 183, 261, 289)
        """
        
        top_left = Point(bounding_box[0], bounding_box[1])
        bottom_right = Point(bounding_box[2], bounding_box[3])
         
        top_right = Point(bottom_right.x, top_left.y)
        bottom_left = Point(top_left.x, bottom_right.y)
       
        top_left = self.rectify_left_point(top_left)
        top_right = self.rectify_left_point(top_right)
        bottom_right = self.rectify_left_point(bottom_right)
        bottom_left = self.rectify_left_point(bottom_left)

        # we compensate a little bit the rectification distortion by averaging
        # the height of the corners,
        # the horizontal dimension is not modified         
        top_left_y = (top_left.y + top_right.y) / 2
        bottom_right_y = (bottom_right.y + bottom_left.y) / 2
        
        return (top_left.x, top_left_y, bottom_right.x, bottom_right_y)
    
    
    