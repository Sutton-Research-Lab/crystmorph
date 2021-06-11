#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from math import atan2
import numpy as np

sin = np.sin
cos = np.cos


###############################################
# 3D transformation (homogeneous coordinates) #
###############################################

def translation3D(tx=0, ty=0, tz=0):
    """ Translation matrix in 3D.
    """

    mat = np.array([[1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0,  1]])

    return mat


def scaling3D(sx, sy, sz):
    """ Scaling matrix in 3D.
    """

    mat = np.array([[sz, 0, 0, 0],
                    [0, sy, 0, 0],
                    [0, 0, sz, 0],
                    [0, 0, 0,  1]])

    return mat


def rotxh(rx, radian=False):
    """ Rotation matrix about x axis.
    """

    if not radian:
        rx = np.radians(rx)
    
    mat = np.array([[1,       0,        0, 0],
                    [0, cos(rx), -sin(rx), 0],
                    [0, sin(rx),  cos(rx), 0],
                    [0,       0,        0, 1]])
    
    return mat


def rotyh(ry, radian=False):
    """ Rotation matrix about y axis.
    """

    if not radian:
        ry = np.radians(ry)

    mat = np.array([[cos(ry),  0,  sin(ry), 0],
                    [0,        1,        0, 0],
                    [-sin(ry), 0,  cos(ry), 0],
                    [0,        0,        0, 1]])

    return mat


def rotzh(rz, radian=False):
    """ Rotation matrix about z axis.
    """

    if not radian:
        rz = np.radians(rz)

    mat = np.array([[cos(rz), -sin(rz), 0, 0],
                    [sin(rz),  cos(rz), 0, 0],
                    [0,              0, 1, 0],
                    [0,              0, 0, 1]])

    return mat


#############################################
# 3D transformation (Cartesian coordinates) #
#############################################

def Glazer_deform(a, b, c):
    """ Deformation matrix formulation of Glazer tilt system (single/double perovskites).
    """

    mat = np.array([[0, -c, b],
                    [-c, 0, a],
                    [b, -a, 0]])

    return mat


def rotx(rx, radian=False):
    """ Rotation matrix about x axis.
    """

    if not radian:
        rx = np.radians(rx)
    
    mat = np.array([[1,       0,        0],
                    [0, cos(rx), -sin(rx)],
                    [0, sin(rx),  cos(rx)]])
    
    return mat


def roty(ry, radian=False):
    """ Rotation matrix about y axis.
    """

    if not radian:
        ry = np.radians(ry)

    mat = np.array([[cos(ry),  0,  sin(ry)],
                    [0,        1,        0],
                    [-sin(ry), 0,  cos(ry)]])

    return mat


def rotz(rz, radian=False):
    """ Rotation matrix about z axis.
    """

    if not radian:
        rz = np.radians(rz)

    mat = np.array([[cos(rz), -sin(rz), 0],
                    [sin(rz),  cos(rz), 0],
                    [0,              0, 1]])

    return mat


def kabsch(A, B):
    """ Calculates the least-squares best-fit transform that maps corresponding points
    A to B in m spatial dimensions using the Kabsch algorithm.
    
    **Parameters**\n
    A, B: numpy.ndarray, numpy.ndarray
          Coordinates of the corresponding points (Nxm)
    
    **Returns**\n
    R: numpy.ndarray
        mxm rotation matrix
    t: numpy.ndarray
        mx1 translation vector
    """
    
    if not A.shape == B.shape:
        raise ValueError('The shapes of A and B should be the same.')
    else:
        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[m-1,:] *= -1
           R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        return R, t


def rot2euler(R, radian=False):
    """ Decompose arbitrary 3D rotation matrix into rotations around x, y, and z
    (applied in the order Rz.Ry.Rx) axis respectively and return the Euler angles.
    """
 
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else:
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
        
    angles = np.array([x, y, z])
    
    if not radian:
        return np.degrees(angles)
    else:
        return angles


def cart2homo(coords_cart):
    """ Transform from Cartesian to homogeneous coordinates.
    """

    ones = np.ones((len(coords_cart), 1))
    coords_homo = np.squeeze(np.concatenate((coords_cart, ones), axis=1))

    return coords_homo


def homo2cart(coords_homo):
    """ Transform from homogeneous to Cartesian coordinates.
    """

    try:
        coords_cart = np.squeeze(coords_homo)[:,:2]
    except:
        coords_cart = np.squeeze(coords_homo)[:2]

    return coords_cart


###################################
# Crystallographic transformation #
###################################

def cart2frac(cart_coords, cellAxes):
    """
    Conversion from Cartesian to fractional coordinates
    """

    invAxes = np.linalg.inv(cellAxes)
    xyzFrac = np.dot(cart_coords, invAxes)

    return xyzFrac


def frac2cart(xyzFrac, axes):
    """
    Conversion from fractional to Cartesian coordinates
    """

    xyzCart = np.dot(xyzFrac, axes)

    return xyzCart