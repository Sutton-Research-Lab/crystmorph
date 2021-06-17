#! /usr/bin/env python

# The script calculates the tilting angles of coordinated octohedra.

import crystmorph as cmor
import parsetta as ps
import numpy as np

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments


excluded = ['H', 'C', 'O']
ce_octahedron = 'O:6'
xvec = np.array([1, 0, 0])
yvec = np.array([0, 1, 0])
zvec = np.array([0, 0, 1])

# Gather structure files
fdir = r'../data'
flist = ps.utils.findFiles(fdir, fstring='/*', ftype='cif')

for ife, fe in enumerate(flist[:]):
    # Parse structure
    sp = cmor.structure.PerovskiteParser(fe)

    # Parse coordination environment (details see https://doi.org/10.1107/S2052520620007994)
    lgf = LocalGeometryFinder()
    lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=False)
    lgf.setup_structure(structure=sp.struct)
    se = lgf.compute_structure_environments(excluded_atoms=excluded, maximum_distance_factor=1.41, only_cations=True)
    strategy = SimplestChemenvStrategy(distance_cutoff=3.4, angle_cutoff=0.3)
    lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
    
    # Enumerate over structure-parsed neighborhoods
    for ice, ce in enumerate(lse.coordination_environments):
        # Retrieve octahedral coordination environments
        if (ce is not None) and (ce[0]['ce_symbol'] == ce_octahedron):
            nbset = lse.neighbors_sets[ice][0]

            # Order atoms within neighborhood
            nh = cmor.structure.Neighborhood(atom_sites=nbset.neighb_sites_and_indices, site_coords=nbset.neighb_coords)
            coplanar_indices = nh.get_coplanar_atom_ids()
            ordered_octahedron_vertices = nh.get_ordered_vertices(indices=coplanar_indices, direction='ccw')

            ags = [] # Tilting angles
            for ivec, vec in enumerate([xvec, yvec, zvec]):
                ordering = nh.directional_filter(direction_vector=vec, direction='ccw')
                ordered_octahedron_vertices = nh.get_ordered_vertices(ordering=ordering)

                # Computer geometric parameters related to the coordination octahedron
                octa = cmor.polyhedron.Octahedron(vertices=ordered_octahedron_vertices['coords'])

                # Compute geometric parameters (using Glazer definition)
                angle = octa.vector_orientation(octa.apical_vector, refs=[vec])
                ags.extend(angle)
            
            print('Tilting angles along a, b, c axes are {}, {}, {} degrees'.format(*ags))

                # TODO: estimate the angles using optimal rotations
                # Computer geometric parameters (using Kabsch algorithm)