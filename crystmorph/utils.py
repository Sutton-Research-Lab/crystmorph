#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

import numpy as np
from operator import attrgetter


def multiretrieve(info_list, iterable):
    """ Retrieve multiple pieces of information at different levels of iterable object.
    """

    info = list(map(lambda s: attrgetter(*info_list)(s), iterable))
    info = np.asarray(info).T
    info_dict = {name: value for name, value in zip(info_list, info)}

    return info_dict


def multidict_merge(composites):
    """ Merge multiple dictionaries with the same keys.
    """
    
    if len(composites) in [0, 1]:
        return composites
    else:
        # Initialize an empty dictionary with only keys
        merged = dict.fromkeys(list(composites[0].keys()), [])
        for k in merged.keys():
            merged[k] = list(d[k] for d in composites)
    
    return merged


def pointset_order(pset, center=None, direction='ccw', ret='order'):
    """
    Order a point set around a center in a clockwise or counterclockwise way.

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        center : list/tuple/1D array | None
            Pixel coordinates of the putative shape center.
        direction : str | 'ccw'
            Direction of the ordering ('cw' or 'ccw').

    :Return:
        pset_ordered : 2D array
            Sorted pixel coordinates of the point set.
    """

    dirdict = {'cw':1, 'ccw':-1}

    # Calculate the coordinates of the
    if center is None:
        pmean = np.mean(pset, axis=0)
        pshifted = pset - pmean
    else:
        pshifted = pset - center

    pangle = np.arctan2(pshifted[:, 1], pshifted[:, 0]) * 180/np.pi
    # Sorting order
    order = np.argsort(pangle)[::dirdict[direction]]
    pset_ordered = pset[order]
    
    if ret == 'order':
        return order
    elif ret == 'points':
        return pset_ordered
    elif ret == 'all':
        return order, pset_ordered


def csm(coords, model):
    """ Continuous symmetry measure for ordered polyhedral vertices.
    """
    
    numerator = np.sum(np.linalg.norm(coords - model, axis=1))
    center = coords.mean(axis=0)
    denominator = np.sum(np.linalg.norm(coords - center, axis=1))
    
    metric = numerator / denominator
    
    return metric