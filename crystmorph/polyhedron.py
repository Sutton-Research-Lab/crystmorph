#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from . import transformation as trans
import numpy as np
import vg
import itertools as it
from scipy.spatial import ConvexHull


class ConvexPolyhedron(object):
    """ Root class for convex polyhedra.
    """
    
    def __init__(self, center=np.array([0, 0, 0]), n_vertex=None, n_edge=None):
        
        self.center = center
        self.n_vertex = n_vertex
        self.n_edge = n_edge
    
    @property
    def n_face(self):
        """ Number of faces (calculated with Euler's formula).
        """
        
        try:
            n_face = self.n_vertex + self.n_edge - 2
            return n_face
        except:
            return None
    
    @property
    def is_proper_shape(self):
        """ Check if the shape is proper.
        """
        
        try:
            if (self.n_vertex > 0) and (self.n_edge > 0) and (self.n_face > 0):
                return True
        except:
            return False


def is_corner_sharing(ph1, ph2, **kwargs):
    """ Determine if two polyhedra are sharing a corner.
    """

    is_sharing = []
    
    if ph1.n_vertex >= ph2.n_vertex:
        for v in ph2.vertices:
            is_sharing.append(np.allclose(ph1.vertices, v, **kwargs))
    else:
        for v in ph1.vertices:
            is_sharing.append(np.allclose(ph2.vertices, v, **kwargs))
    nclose = np.sum(is_sharing)
    
    if nclose == 1:
        return True # It's counted as corner sharing iff one vertex is common
    else:
        return False


class Face(object):
    """ Coplanar polygon data structure featuring an ordered vertex list.
    """
    
    def __init__(self, vertices, order='cw'):
        
        self.vertices = vertices
        self.n_vertex = len(vertices)
        if self.n_vertex < 3:
            raise ValueError('Number of vertices should be more than 3!')
        else:
            # Calculate basic properties of the polygon
            self.calculate_center()
            self.vertex_order(order=order)
            self.calculate_edges()
            self.calculate_normal()
        
    def vertex_order(self, order='cw'):
        """ Vertex list ordering ('cw' as clockwise or 'ccw' as counterclockwise).
        """
        
        # For coplanar polygon the point
        keys = list(range(1, self.n_vertex+1))
        self.verts_dict = dict(zip(keys, self.vertices))

    def calculate_center(self):
        """ Calculate the center coordinate of the face polygon.
        """

        self.center = self.vertices.mean(axis=0)
        
    def calculate_normal(self):
        """ Calculate the unit normal vector of the face in 3D.
        """
        
        normal = np.cross(self.edges[1], self.edges[2])
        self.unit_normal = normal / np.linalg.norm(normal)
    
    def calculate_edges(self):
        """ Calculate the edge properties of vertices.
        """
        
        self.edges = np.diff(self.vertices, axis=0, append=self.vertices[:1])
        self.edgelengths = np.linalg.norm(self.edges, axis=1)
        
    def calculate_area(self):
        """ Area of the face in 3D (polygon area).
        """
        
        total = np.array([0, 0, 0])
        for i in range(self.n_vertex):
            vi1 = self.vertices[i]
            if i == self.n_vertex-1:
                vi2 = self.vertices[0]
            else:
                vi2 = self.vertices[i+1]
            prod = np.cross(vi1, vi2)
            total += prod
        
        self.area = abs(np.dot(total, self.unit_normal) / 2)


class Ordering(object):
    """ Class for ordering of a point set.
    """
    
    def __init__(self, *n):
        
        self.n = n
        self.n_parts = len(n)
        self.indices = [list(range(ni)) for ni in n]
        
    def cartesian_product_ordering(self):
        """ Calculate the Cartesian product ordering of the set.
        """
        
        ordered_indices = it.product(*[self.indices*self.n])
        
        return list(ordered_indices)
    
    def cwr_ordering(self):
        """ Calculate the set ordering by combination with replacement (CWR).
        """
        
        if self.n_parts == 1:
            ordered_indices = it.combinations_with_replacement(self.indices, self.n)
        
        return list(ordered_indices)
    
    def permute(self, shift, keep=True):
        """ Permute the indices.
        """
        
        rolled = np.roll(self.indices, shift=shift)
        if keep:
            self.indices = rolled.tolist()
        
        
class DoubletOrdering(Ordering):
    """ Class for ordering of two things.
    """
    
    def __init__(self, type='cwr'):
        
        super().__init__(2)
        if type == 'cartesian':
            self.ordered_indices = self.cartesian_product_ordering()
        elif type == 'cwr':
            self.ordered_indices = self.cwr_ordering()


class TripletOrdering(Ordering):
    """ Class for ordering of three things.
    """
    
    def __init__(self, type='cwr'):
        
        super().__init__(3)
        if type == 'cartesian':
            self.ordered_indices = self.cartesian_product_ordering()
        elif type == 'cwr':
            self.ordered_indices = self.cwr_ordering()


class Octahedron(ConvexPolyhedron):
    """ Octahedral object with coordinate transformation of its vertices and faces.
    """
    
    def __init__(self, n_vertex=6, n_edge=12, vertices=None, **kwargs):
        
        if vertices is None:
            center = kwargs.pop('center', np.array([0, 0, 0]))
        else:
            center = np.mean(vertices, axis=0)
        super().__init__(center=center, n_vertex=n_vertex, n_edge=n_edge)
        self.vertices = vertices

    @property
    def convex_hull(self):
        """ Convex hull of the vertices.
        """
        try:
            return ConvexHull(self.vertices)
        except:
            return None

    @property
    def volume(self):
        """ Volume of the octahedron.
        """

        try:
            return self.convex_hull.volume
        except:
            return None
        
    def generate_vertices(self, radius, poly_type='regular', angles=None, alpha=0, beta=0, gamma=0):
        """ Generate the vertex coordinates of the octahedron.
        """
        
        self.poly_type = poly_type
        # Generate the coordinates of vertices
        if self.poly_type == 'regular': # regular octahedron
            a = radius
            self.vertices = np.array([[a, 0, 0], [-a, 0, 0], 
                                      [0, a, 0], [0, -a, 0],
                                      [0, 0, a], [0, 0, -a]])
        elif self.poly_type == 'rhom_bipyramid': # rhombic bipyramid
            a, b, c = radius
            self.vertices = np.array([[a, 0, 0], [-a, 0, 0], 
                                      [0, b, 0], [0, -b, 0],
                                      [0, 0, c], [0, 0, -c]])
        elif self.poly_type == 'asym_bipyramid': # asymmetric bipyramid
            self.vertices = np.array([[radius[0], 0, 0], [radius[1], 0, 0], 
                                      [0, radius[2], 0], [0, radius[3], 0],
                                      [0, 0, radius[4]], [0, 0, radius[5]]])
            
        # Rotate according to the pose angles
        if angles is not None:
            alpha, beta, gamma = angles
        
        ctr = self.center
        transform_list = [trans.translation3D(*ctr), trans.rotzh(gamma),
                          trans.rotyh(beta), trans.rotxh(alpha),
                          trans.translation3D(*-ctr)]
        
        transform_matrix = np.matmul(*transform_list)
        self.vertices = np.dot(transform_matrix, self.vertices)
        
        # Create vertex list
        keys = range(1, len(self.n_vertex)+1)
        self.verts_dict = dict(keys, self.vertices)

    @property
    def apical_vector(self):
        """ Apical vector of the octahedron.

        self.vertices[0] is the vertex on the far side,
        self.vertices[-1] is the vertex on the near side.
        """

        return self.vertices[0] - self.vertices[-1]


    def vector_orientation(self, vector, refs=None):
        """ Orientation angles of a vector.
        """

        if refs is None:
            xvec = np.array([1, 0, 0])
            yvec = np.array([0, 1, 0])
            zvec = np.array([0, 0, 1])
            refs = [xvec, yvec, zvec]

        angles = [vg.angle(vector, ref) for ref in refs]

        return angles




