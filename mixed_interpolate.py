#!/usr/bin/python

# Uses geodesic interpolation for the molecule and idpp interpolation for the surface of a molecule

from ase.io import read, write
from ase import Atoms
import re
import os
import sys
from ase.mep import NEB
from ase.calculators.lj import LennardJones as LJ
from ase.io import read,Trajectory
from ase.visualize import view
from pathlib import Path
import csv
from decimal import Decimal
import ase.io
from geodesic_interpolate.interpolation import redistribute
from geodesic_interpolate.geodesic import Geodesic

######## Functions as copied from geodesic_wrapper.py ########
def ase_geodesic_interpolate(initial_mol,final_mol, n_images = 20, friction = 0.01, dist_cutoff = 3, scaling = 1.7, sweep = None, tol = 0.002, maxiter = 15, microiter = 20):
    atom_string = initial_mol.symbols
    
    atoms = list(atom_string)

    initial_pos = [initial_mol.positions]
    final_pos = [final_mol.positions]

    total_pos = initial_pos + final_pos

    # First redistribute number of images.  Perform interpolation if too few and subsampling if too many
    # images are given
    raw = redistribute(atoms, total_pos, n_images, tol=tol * 5)

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = Geodesic(atoms, raw, scaling, threshold=dist_cutoff, friction=friction)

    if sweep is None:
        sweep = len(atoms) > 35
    try:
        if sweep:
            smoother.sweep(tol=tol, max_iter=maxiter, micro_iter=microiter)
        else:
            smoother.smooth(tol=tol, max_iter=maxiter)
    finally:

        all_mols = []
        
        for pos in smoother.path:
            mol = Atoms(atom_string, pos)
            all_mols.append(mol)
        
        return all_mols

########## Functions copied from geodesic_wrapper.py ##########
def interpolate_traj(initial,final,LEN,method,calculator=None):
    interpolated_length=LEN
    initial_1=initial.copy()
    initial_1.calc=calculator #attach calculator to images between start and end
    RETURN=initial_1.copy()
    images = [initial]
    images += [initial_1.copy() for i in range(interpolated_length)] #create LEN instances
    images += [final]
    nebts = NEB(images)
    nebts.interpolate(mic=True,method=method,apply_constraint=True)
    new_traj=Trajectory(f'{LEN}_interpol.traj',mode='w')
    for im in images:
        new_traj.write(im)
def create_trajs(START,END,LEN,method):
    calc = LJ()
    TRAJ=interpolate_traj(START,END,LEN,method,calculator=calc)

def vprint(words):
    if args.verbose:
        print(words)


######### Combination ######### @FThiemann



def main():
    import argparse

    parser = argparse.ArgumentParser(description='Interpolate Mixing the geodesic and idpp/direct interpolation methods.')
    parser.add_argument('start', type=str, help='POSCAR file of initial molecule')
    parser.add_argument('end', type=str, help='POSCAR file of final molecule')
    parser.add_argument('Images', type=int, help='Number of NEB Images to generate')
    parser.add_argument('SurfaceCutoff', type=int, help='Length of the molecule')
    parser.add_argument('method', type=str,choices=['idpp','direct'] , help='Interpolation method')
    parser.add_argument('-c','--check', action='store_true', help='Check the cutoff trajectory')
    parser.add_argument('-s','--show', action='store_true', help='view trajectory')
    parser.add_argument('-v','--verbose', action='store_true', help='verbosity of output')
    global args
    args = parser.parse_args()

    initial_mol = read(args.start)
    final_mol = read(args.end)
    LEN = args.Images
    moleculeStart = -1 * args.SurfaceCutoff
    method = args.method

    create_trajs(initial_mol, final_mol, LEN, method) #create the interpolated trajectory using the idpp/direct method
    trajectory_pbc=read(f'{LEN}_interpol.traj',index=':') #read the interpolated trajectory
    vprint(f"Interpolated trajectory created with {len(trajectory_pbc)} images using {method} method")
    trajectory_geodesic = ase_geodesic_interpolate(initial_mol,final_mol, n_images= LEN+1) #create the geodesic interpolated trajectory
    vprint(f"Geodesic interpolated trajectory created with {len(trajectory_geodesic)} images")
    difference=initial_mol[0].position-trajectory_geodesic[0][0].position 
    for image in trajectory_geodesic: #fix the the position of the shifted molecule since the geodesic interpolation does not use pbc
        image.set_cell(initial_mol.cell)
        for at in image:
            at.position=at.position+difference 
    newTrajectory = trajectory_pbc.copy()
    trajectory_geodesic.append(final_mol)
    new_traj=Trajectory(f'{LEN}_interpol.traj',mode='w')
    for nr, image in enumerate(trajectory_pbc):
        atoms_idpp = trajectory_pbc[nr]
        atoms_geodesic = trajectory_geodesic[nr]
        molecule = atoms_geodesic[moleculeStart:]
        surface = atoms_idpp[:moleculeStart]
        if nr == 0 and args.check == True: #check the cutoff
            view(surface)
            view(molecule)
        merged = Atoms(surface + molecule) #merge the molecule and the surface
        newTrajectory[nr] = merged
        new_traj.write(merged)
        N=f'{nr:02d}'
        vprint(f'Writing {N}/POSCAR')
        Path(N).mkdir(parents=True,exist_ok=True)
        newTrajectory[nr].write(f'{N}/POSCAR',format='vasp')
    
    if args.show == True:
        view(newTrajectory)
main()
