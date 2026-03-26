#!/usr/bin/env python3
#
# Script to insert a Molecule saved in .xyz into a new POSCAR with the cell of POSCAR,
# including centering in cell, sorting elements, constraining atoms, and rotating to xy-plane
# by Jakob Schramm
# 2026/03/26
#
import numpy as np
from ase import Atom, Atoms
from ase.io import read, write
from ase.build.tools import sort
from ase.constraints import FixAtoms
from scipy.optimize import leastsq
import argparse

def distance_to_plane(params,X):
    distance = ((params[0:3]*X.T).sum(axis=1) + params[3])/np.linalg.norm(params[0:3])
    return distance

def main(xyz, poscar, out, rot, cen, sor, const):
    mol = read(xyz)
    cell_info = read(poscar)
    mol.set_pbc(True)
    mol.set_cell(cell_info.cell)
    print("rotated to xy plane?", rot)
    print("centered?", cen)
    print("sorted?", sor)
    print("constrained?", const)
    if rot:
        mol_coords = mol.get_positions()
        plane_guess = [0.1, 0.1, 0.1, 0.1]
        solution = leastsq(distance_to_plane, plane_guess, args=(mol_coords.T), maxfev=100000)[0]
        a = solution[0]/max(solution)
        b = solution[1]/max(solution)
        c = solution[2]/max(solution)
        d = solution[3]/max(solution)
        cos_angle = c/np.sqrt(a**2+b**2+c**2)
        sin_angle = np.sqrt((a**2+b**2)/(a**2+b**2+c**2))
        u1 = b/np.sqrt(a**2+b**2)
        u2 = -a/np.sqrt(a**2+b**2)
        R_matrix = np.array([[cos_angle+(u1**2)*(1-cos_angle), u1*u2*(1-cos_angle),           u2*sin_angle],
                             [u1*u2*(1-cos_angle),           cos_angle+(u2**2)*(1-cos_angle), -u1*sin_angle],
                             [-u2*sin_angle,                 u1*sin_angle,                  cos_angle]])
        for index,atom in enumerate(mol):
            atom.position = R_matrix.dot(mol_coords[index])
    if cen: mol.center()
    if sor: mol = sort(mol)
    if const: mol.set_constraint(FixAtoms(indices=[atom.index for atom in mol]))
    mol.write(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a Molecule saved in xyz into a new POSCAR with the Cell of POSCAR')
    parser.add_argument('xyz_file', type=str, help='Coordinates from xyz File')
    parser.add_argument('cell_from_POSCAR', type=str, help='Cell from POSCAR File')
    parser.add_argument('--outfile', help='Name of new POSCAR', default='POSCAR_new')
    parser.add_argument('--no_rotation_to_xy', help='DON\'T Molacular Plane into XY Plane', action='store_false')
    parser.add_argument('--no_center', help='DON\'T Center Atoms in Cell', action='store_false')
    parser.add_argument('--no_sort', help='DON\'T Sort Atom Labels', action='store_false')
    parser.add_argument('--constrain', help='Fix all Atom Positions', action='store_true')
    args = parser.parse_args()
    main(args.xyz_file, args.cell_from_POSCAR, args.outfile, args.no_rotation_to_xy, args.no_center, args.no_sort, args.constrain)
