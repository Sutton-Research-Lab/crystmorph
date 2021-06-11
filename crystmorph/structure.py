#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from . import polyhedron as ph, utils as u
import numpy as np
import itertools as it
from math import sqrt
# Geometry packages
import vg
from pyrr import plane

import json
from copy import deepcopy
import operator as op
from functools import reduce
try:
    import pymatgen
    from pymatgen.core.structure import Structure
    from pymatgen.core.sites import PeriodicSite
    from pymatgen.symmetry import analyzer as sym
except:
    pass

try:
    from CifFile import ReadCif
except:
    pass

import warnings as w
w.filterwarnings("ignore")


class CIFParser(object):
    """ Parser class for crystallographic information files (CIFs).
    """

    def __init__(self, filepath, backend='pymatgen'):

        if backend == 'pymatgen':
            self.struct = Structure.from_file(filepath)
        elif backend == 'pycifrw':
            self.struct = ReadCif(filepath)


class JSONParser(object):
    """ Parser class for crystallographic information in json format.
    """

    def __init__(self, filepath, **kwargs):
        
        with open(filepath) as f:
            self.struct = json.load(f, **kwargs)


class CIFExporter(object):
    """ Exporter class for crystallographic information files (CIFs).
    """

    def __init__(self, obj, filepath, backend='pymatgen'):

        if backend == 'pymatgen':
            obj.to(filepath)
        elif backend == 'pycifrw':
            with open(filepath) as f:
                f.write(obj.WriteOut())


class JSONExporter(object):
    """ Exporter class for crystallographic information in json format.
    """

    def __init__(self, obj, filepath, **kwargs):

        with open(filepath, 'w') as f:
            json.dump(obj, f, **kwargs)


class StructureParser(object):
    """ Parser class for crystal structures. The class funnels the atomistic information
    of a crystal structure into a searchable format using a combination of new methods and
    those present in ``pymatgen.core.structure.Structure`` class.
    """

    def __init__(self, filepath, form='cif', parser_kwargs={}, **kwargs):

        if form == 'cif':
            parser = CIFParser(filepath=filepath, **parser_kwargs)
        elif form == 'json':
            parser = JSONParser(filepath=filepath, **parser_kwargs)
        self.struct = parser.struct

        # Retrieve essential chemical information (default is the atom name, number, Cartesian and fractional coordinates)
        self.atoms_info_list = kwargs.pop('atoms_info_list', ['specie.name', 'specie.number', 'coords', 'frac_coords'])
        self.atoms_dict = u.multiretrieve(self.atoms_info_list, self.struct.sites)
        indices = np.array(list(range(self.n_atoms)))
        self.atoms_dict['atom_id'] = indices
        
        self.unique_elements = [s.name for s in self.struct.types_of_specie]
        self.composition = self.struct.composition.as_dict()

    @property
    def n_atoms(self):
        """ Number of atoms in the structure.
        """

        try:
            return self.struct.num_sites
        except:
            return 0

    def filter_elements(self, keep=[], remove=[], update_structure=False):
        """ Remove atoms according to the element type.
        """

        if len(keep) > 0:
            pass
        elif len(remove) > 0:
            pass

        if update_structure:
            self.struct

        return 

    def symmetry_analyze(self, **kwargs):
        """ Produce symmetry analysis.
        """

        self.symmetry = sym.SpacegroupAnalyzer(self.struct, **kwargs)
        self.crystal_system = self.symmetry.get_crystal_system()

    def find_atoms(self, names=None):
        """ Locate the atomic species according to specification.
        """

        atom_names = self.atoms_dict['specie.name']
        if type(names) == str:
            names = [names]
        for name in names:
            loc = np.where(atom_names == name)
            selected_atoms_dict = {ai_name: ai_value[loc] for ai_name, ai_value in self.atoms_dict.items()}

        return selected_atoms_dict

    def find_neighbors(self, radius, idx=None, include_center_atom=False):
        """ Locate the neighboring atom species according to specification.
        """

        # Retrieve the atomic site instance
        site = self.struct.sites[idx]
        neighbors_sites = self.struct.get_neighbors(site, radius)
        neighbors_dict = u.multiretrieve(self.atoms_info_list, neighbors_sites)
        if include_center_atom:
            return site.as_dict(), neighbors_dict
        else:
            return neighbors_dict

    def find_local_environment(self, r_cutoff, a_cutoff=0.3, excluded_atoms=[]):
        """ Determination of local coordination environment.
        """

        from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
        from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
        from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

        lgf = LocalGeometryFinder()
        lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=False)
        lgf.setup_structure(structure=self.struct)
        self.struct_env = lgf.compute_structure_environments(excluded_atoms=excluded_atoms, maximum_distance_factor=1.41)
        strategy = SimplestChemenvStrategy(distance_cutoff=r_cutoff, angle_cutoff=a_cutoff)
        self.lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=self.struct_env)

    def perturb(self, distance, min_distance=None, keep=False, ret=True):
        """ Apply random perturbation to the atomic coordinates.
        """

        struct = deepcopy(self.struct)
        struct_perturbed = self.struct.perturb(distance, min_distance)
        if not keep:
            self.struct = struct

        if ret:
            return struct_perturbed

    def calculate_distances(self, atom1=[], atom2=[], atom_pairs=None):
        """ Calculate distances between specified coordinates of atom pairs.
        """

        if atom_pairs is not None:
            atom1, atom2 = atom_pairs

        d = vg.euclidean_distance(atom1, atom2)

        return d        

    def calculate_angles(self, atom1=[], atom2=[], atom3=[], atom_trios=None):
        """ Calculate angles between specified coordinates of atom trios.
        """

        if atom_trios is not None:
            atom1, atom2, atom3 = atom_trios
        
        ag = vg.angle(atom2-atom1, atom2-atom3)

        return ag


class Neighborhood(object):
    """ Class for ordering neighborhood atomic sites.
    """
    
    def __init__(self, atom_sites, site_indices=None, site_coords=None):
        
        # self.isites = tuple(atom_sites)
        if atom_sites is None:
            self.sites = []
            for sid, sc in zip(site_indices, site_coords):
                site = {'index':sid, 'coords':sc}
                self.sites.append(site)
        else:
            self.sites = list(atom_sites)
        self.ordering = []
        
        if site_indices is None:
            self.site_indices = list(s['index'] for s in self.sites)
        else:
            self.site_indices = site_indices
            
        if site_coords is None:
            self.site_coords = list(s['site'].coords for s in self.sites)
        else:
            self.site_coords = site_coords
        
        try:
            cds = list(s['site'].coords for s in self.sites)
        except:
            cds = list(s['coords'] for s in self.sites)
        self.center = np.mean(self.site_coords, axis=0)
        self.cvdist = [np.linalg.norm(cd - self.center) for cd in cds]
        self.mean_cvdist = np.mean(self.cvdist)
        
    @property
    def nsite(self):
        """ Number of atomic sites.
        """
        
        return len(self.sites)
        
    def _get_pairs(self, components, exclude=[]):
        """ Generate a list of every two pairs of nonequivalent components.
        """
        
        if len(exclude) != 0:
            excluded_components = op.itemgetter(*exclude)(components)
            comps = [i for i in components if i not in excluded_components]
        else:
            comps = components
        
        return [list(cb) for cb in it.combinations(comps, 2)]
        
    def _get_trios(self, components, exclude=[]):
        """ Generate a list of every three nonequivalent components.
        """
        
        if len(exclude) != 0:
            excluded_components = op.itemgetter(*exclude)(components)
            comps = [i for i in components if i not in excluded_components]
        else:
            comps = components
        
        return [list(cb) for cb in it.combinations(comps, 3)]
    
    def get_atom_pairs(self, exclude=[]):
        """ Generate every two pairs of nonequivalent atoms.
        """
        
        return self._get_pairs(self.sites, exclude=exclude)
    
    def get_atom_trios(self, exclude=[]):
        """ Generate every three nonequivalent atoms.
        """
        
        return self._get_trios(self.sites, exclude=exclude)
    
    def get_pair_geometries(self, exclude_atoms=[], as_dict=False):
        """ Obtain pairwise direction vectors of atoms. Atom_pairs_merged list has the following structure,
        
        atom_pairs_merged = [pair_1, pair_2, ...],
        pair_1 = {'coords': [...], 'direction': array([...]),
                  'index': [...], 'length': ...},
        
        with 'coords' containing the Cartesian coordinates of atom pair, 'direction' containing
        the directional vector connecting the atom pair, 'index' containing the atomic indices
        of the pair within the crystal structure, 'length' containing the distance between the atoms.
        
        The atom_pairs_dict returns with additional integer-valued dictionary keys. The integer is assigned
        according to the sequence of the atom pair.
        """
        
        atom_pairs = self._get_pairs(components=self.sites, exclude=exclude_atoms)
        atom_pairs_merged = list(map(u.multidict_merge, atom_pairs))
        for p in atom_pairs_merged:
            p['coords'] = []
            p['coords'].extend([s.coords for s in p['site']])
            del p['site']
        
            p['direction'] = op.sub(*p['coords'])
            p['length'] = sqrt(sum(p['direction']**2))
        
        if not as_dict:
            return atom_pairs_merged
        else:
            npairs = len(atom_pairs_merged)
            atom_pairs_dict = dict(zip(range(npairs), atom_pairs_merged))
            return atom_pairs_dict
    
    def directional_filter(self, direction_vector=np.array([0, 0, 1]), direction='ccw'):
        """ Filter the atomic coordinates along a specified direction.
        """
        
        projections = [sc.dot(direction_vector) for sc in self.site_coords]
        direction_order = np.argsort(projections) # Sort projection in ascending order
        
        # Order coplanar atoms according to their coordinates
        in_plane_order = direction_order[1:-1]
        in_plane_coords = [self.site_coords[i] for i in in_plane_order]
        temp_order = u.pointset_order(np.array(in_plane_coords), direction=direction, ret='order')
        in_plane_order = [in_plane_order[i] for i in temp_order]
        
        basal_indices = [self.site_indices[i] for i in in_plane_order]
        lower = [self.site_indices[direction_order[0]]]
        upper = [self.site_indices[direction_order[-1]]]
        
        ordered_indices = upper + basal_indices + lower
        ordering = [self.site_indices.index(oi) for oi in ordered_indices]
        
        return ordering

    def get_coplanar_atom_ids(self, exclude_atoms=[], parallel_tol=1e-8, length_filter=True, **kwargs):
        """ Obtain sets of approximately coplanar vertices by filtering geometric features.
        
        1. Impose length filter on individual atom pairs (distance should be within a range).
        2. Impose approximate collinearity filter on pairs of internuclear vectors (retrieve the vectors that
        would be the base edges of the coordination polyhedron). This amounts to a coplanarity test on atom positions.
        
        Returns unordered indices of coplanar atoms.
        """
        
        all_atom_pairs = self.get_pair_geometries(exclude_atoms=exclude_atoms, as_dict=False)
        # Impose length filter
        if length_filter:
            length_range = kwargs.pop('length_range', [1.0, 1.8])
            length_range = np.array(length_range) * self.mean_cvdist
            lmin, lmax = min(length_range), max(length_range)
            
            all_distances = [p['length'] for p in all_atom_pairs]
            length_filtered_indices = np.argwhere((all_distances < lmax) & (all_distances > lmin)).ravel()
            
            all_atom_pairs = op.itemgetter(*length_filtered_indices)(all_atom_pairs)
            
        vector_pairs = self._get_pairs(all_atom_pairs)
        # Impose approximate collinearity filter on internuclear vectors
        collinearity = np.array([vg.almost_collinear(vp[0]['direction'], vp[1]['direction']) for vp in vector_pairs])
        collinearity_indices = np.argwhere(collinearity == True).ravel()
        collinear_vectors = op.itemgetter(*collinearity_indices)(vector_pairs)
        
        # Break down the collinear vector pairs into coplanar atoms and retrieve the unique indices
        coplanar_atoms = reduce(op.add, collinear_vectors)
        paired_atom_indices = [p['index'] for p in coplanar_atoms]
        # Take only the unique atomic indices
        paired_atom_indices = set(reduce(op.add, paired_atom_indices))
        coplanar_atom_indices = list(paired_atom_indices)
        
        return coplanar_atom_indices
    
    @property
    def ordered_index(self):
        """ Ordered atomic indices.
        """
        
        try:
            indices = [self.site_indices[i] for i in self.ordering]
            return indices
        except:
            return []
        
    @property
    def ordered_coords(self):
        """ Ordered atomic coordinates.
        """
        
        try:
            coords = [self.site_coords[i] for i in self.ordering]
            return coords
        except:
            return []
        
    def get_ordered_vertices(self, indices=None, direction='ccw', ordering=[], site_index=True):
        """ Obtain vertices after imposing ordering, referred to as counterclockwise or clockwise spiral.
        The apical atoms are ordered by the relationship to midplane.
        """
        
        if len(ordering) != 0:
            self.ordering = ordering
        else:
            in_plane_coords = [a['site'].coords for a in self.sites if a['index'] in indices]
            oo_plane_indices = [a['index'] for a in self.sites if a['index'] not in indices]
            oo_plane_coords = [a['site'].coords for a in self.sites if a['index'] not in indices]

            # Order coplanar atoms according to their coordinates
            in_plane_order = u.pointset_order(np.array(in_plane_coords), direction=direction, ret='order')
            in_plane_indices = [indices[i] for i in in_plane_order]
            
            # Order apical atoms according to their position below or above the midplane
            midplane = plane.create_from_points(*in_plane_coords[:3])
            upper, lower = [], []
            for opc in oo_plane_coords:
                if opc.dot(midplane[:3]) > 0:
                    upper.append(opc)
                else:
                    lower.append(opc)
                    
            ordered_indices = upper + in_plane_indices + lower
            self.ordering = [self.site_indices.index(oi) for oi in ordered_indices]
        
        # Impose ordering on the site atoms
        ordered_vertices = {}
        ordered_vertices['order'] = list(range(len(self.site_indices)))
        ordered_vertices['coords'] = self.ordered_coords
        
        if site_index:
            ordered_vertices['index'] = self.ordered_index
        
        return ordered_vertices


class PerovskiteParser(StructureParser):
    """ Parser class for perovskite structures. The single perovskite chemical formula
    follows ABX3 (A, B are cations, X is the anion). The double perovskite chemical formula
    follows AA'B2X6, A2BB'X6, or AA'BB'X6 (A, A', B, B' are cations).
    """

    def __init__(self, filepath, form='cif', category='single', parser_kwargs={}):

        super().__init__(filepath=filepath, form=form, parser_kwargs=parser_kwargs)
        self.category = category
        self.A_cation = None
        self.B_cation = None

    def count(self, propname):
        """ Count the number of things.
        """

        propval = getattr(self, propname)
        if propval is None:
            return 0
        else:
            return len(propval)

    @property
    def nA(self):
        """ Number of A cations.
        """

        return self.count('A_cation')

    @property
    def nB(self):
        """ Number of B cations.
        """

        return self.count('B_cation')

    def find_cation(self, label, names=None):
        """ Find the cation and their basic information from the known structure.
        """

        if label == 'A':
            self.A_cation = self.find_atoms(names=names)
        elif label == 'B':
            self.B_cation = self.find_atoms(names=names)

    def find_octahedron(self, radius=3.5, B_idx=0):
        """ Determine the octahedral coordination of a B cation.

        **Parameters**:
        radius: numeric
            Radial cutoff for finding the atoms within the octahedra.
        """

        if self.B_cation is None:
            raise ValueError('B-site cation information is missing.')
        else:
            octahedron_dict = self.find_neighbors(radius=radius, idx=B_idx)
            octahedron = ph.Octahedron(n_vertex=6, n_edge=12)

            return octahedron

    def get_octahedral_components(self, rcutoff=3.5):
        """ Retrieve all symmetry inequivalent octahedral components in the crystal structure.
        """

        pass

    def get_organic_components(self, method='filter_elements'):
        """ Retrive all symmetry inequivalent organic components in the crystal structure.
        The current algorithm includes filtering out the elemenets or filtering specific structural
        components (e.g. octahedra).
        """

        if self.category == 'hybrid':
            self.struct.remove_species

        pass

    def adjust_A_cation(self, idx=None, shifts=None):
        """ Adjust individual or all A-site cations (either type of atom or the position).
        """

        site_subsitute = PeriodicSite
        if idx is None: # Replace all A cations
            self.struct.specie[0] = []
        elif type(idx) in (list, tuple):
            pass
            
        else:
            raise ValueError('Index should either be None or list or tuple.')

    def adjust_B_cation(self, idx=None, shifts=None):
        """ Adjust individual or all B-site cations (either type of atom or the position).
        """

        pass

    def adjust_X_anion(self, idx=None, shifts=None):
        """ Adjust individual or all X anions.
        """

        pass

    def estimate_tilt_angles(self, axis=['a', 'b', 'c'], method='glazer'):
        """ Calculate the tilting angles defined by Glazer.
        """

        if method == 'glazer':
            pass
        elif method == 'kabsch':
            pass

    def adjust_octahedral_tilting(self):
        """ Adjust the octahedral tilting angles.
        """

        pass

    def save_structure(self, struct, filepath, form='json', **kwargs):
        """ Save the adjusted crystal structure.
        """

        if form == 'json':
            if (type(struct) == Structure) or (type(struct) == PeriodicSite):
                dic = struct.as_dict()
            JSONExporter(obj=dic, filepath=filepath)
        elif form == 'cif':
            CIFExporter(obj=struct, filepath=filepath, **kwargs)

