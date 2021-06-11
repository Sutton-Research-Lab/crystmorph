#! /usr/bin/env python
# -*- coding: utf-8 -*-


from . import polyhedron, transformation
try:
    from . import structure
except:
    pass

import warnings as w
w.filterwarnings("ignore")


__version__ = '0.2.0'
__author__ = 'R. Patrick Xian'