#!/usr/bin/python

# Script that mixes geodesic interpolation for the molecule and idpp/direct interpolation for the surface
# by F. Thiemann & J. Schramm

from ase.io import read
from ase import Atoms
from ase.mep import NEB
from ase.calculators.lj import LennardJones as LJ
from ase.io import Trajectory
from ase.visualize import view
from pathlib import Path
from geodesic_interpolate.interpolation import redistribute
from geodesic_interpolate.geodesic import Geodesic
from tools4vasp.split_surf_and_mol import detect_surf
import numpy as np
from scipy.optimize import leastsq

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
    # RETURN=initial_1.copy()
    images = [initial]
    images += [initial_1.copy() for i in range(interpolated_length)] #create LEN instances
    images += [final]
    nebts = NEB(images)
    nebts.interpolate(mic=True,method=method,apply_constraint=True)
    new_traj=Trajectory(f'{LEN}_{method}_interpol.traj',mode='w')
    for im in images:
        new_traj.write(im)
def create_trajs(START,END,LEN,method):
    calc = LJ()
    TRAJ = interpolate_traj(START,END,LEN,method,calculator=calc)
    return TRAJ

def vprint(words):
    if args.verbose:
        print(words)

def remove_periodic_shifted_atoms(initial_mol, final_mol, surf_indices):
    #Function that removes surface atoms that are shifted between initial mol and final mol due to pbc, because geodesic interpolation cannot handle this and they fly through the air, by J. Schramm
    a_length = np.linalg.norm(initial_mol.cell[0])
    b_length = np.linalg.norm(initial_mol.cell[1])
    removed_idx_list = []
    for atom_idx in surf_indices:
        distance_a = np.abs(initial_mol[atom_idx].position[0]-final_mol[atom_idx].position[0])
        distance_b = np.abs(initial_mol[atom_idx].position[1]-final_mol[atom_idx].position[1])
        if distance_a > a_length - 1 or distance_b > b_length - 1:
            removed_idx_list.append(atom_idx)
    mask = np.ones(len(initial_mol), dtype=bool)
    mask[removed_idx_list] = False
    new_initial_mol = initial_mol[mask]
    new_final_mol = final_mol[mask]
    return new_initial_mol, new_final_mol, removed_idx_list

def distance_to_plane(params,X):
    #Function that calculates distance to a plane, which is needed in rotate_to_XYplane, by J. Schramm
    distance = ((params[0:3]*X.T).sum(axis=1) + params[3])/np.linalg.norm(params[0:3])
    return distance

def rotate_to_XYplane(image,surf_indices):
    #Function that fits a plane to the surface atoms and returns a rotation matrix to the XY plane, by J. Schramm
    surf_coords = image[surf_indices].get_positions()
    plane_guess = [0.1, 0.1, 0.1, 0.1]
    solution = leastsq(distance_to_plane, plane_guess, args=(surf_coords.T,), maxfev=100000)[0]
    a = solution[0]/max(solution)
    b = solution[1]/max(solution)
    norm_val = solution[np.argmax(np.abs(solution))]
    a = solution[0]/norm_val
    b = solution[1]/norm_val
    c = solution[2]/norm_val
    #d = solution[3]/max(solution)
    axis_norm = np.sqrt(a**2+b**2)
    if np.isclose(axis_norm, 0.0):
        return np.eye(3)
    cos_angle = c/np.sqrt(a**2+b**2+c**2)
    sin_angle = np.sqrt((a**2+b**2)/(a**2+b**2+c**2))
    u1 = b/axis_norm
    u2 = -a/axis_norm
    R_matrix = np.array([[cos_angle+(u1**2)*(1-cos_angle), u1*u2*(1-cos_angle),           u2*sin_angle],
                            [u1*u2*(1-cos_angle),           cos_angle+(u2**2)*(1-cos_angle), -u1*sin_angle],
                            [-u2*sin_angle,                 u1*sin_angle,                  cos_angle]])
    return R_matrix

def create_both(initial_mol, final_mol, LEN, method, surf_indices, removepbc=False, alignXY=False, write_geo=False):
    _ = create_trajs(initial_mol, final_mol, LEN, method) #create the interpolated trajectory using the idpp/direct method
    trajectory_pbc=read(f'{LEN}_{method}_interpol.traj',index=':') #read the interpolated trajectory
    vprint(f"Interpolated trajectory created with {len(trajectory_pbc)} images using {method} method")
    if removepbc:
        new_initial_mol, new_final_mol, removed_idx_list = remove_periodic_shifted_atoms(initial_mol, final_mol, surf_indices)
    else:
        new_initial_mol, new_final_mol, removed_idx_list = initial_mol, final_mol, []
    trajectory_geodesic = ase_geodesic_interpolate(new_initial_mol, new_final_mol, n_images= LEN+2) #create the geodesic interpolated trajectory
    vprint(f"Geodesic interpolated trajectory created with {len(trajectory_geodesic)} images")
    trajectory_geodesic = trajectory_geodesic[:-1] #remove last image since it is replaced by original final image
    new_surf_indices = [] #removed indices in geodesic interpolation need to be taken care of when aligning surface in geodesic 
    for idx in surf_indices:
        shift = sum(r < idx for r in removed_idx_list)
        if idx not in removed_idx_list:
            new_surf_indices.append(idx-shift)
    difference = initial_mol[new_surf_indices[0]].position - trajectory_geodesic[0][new_surf_indices[0]].position #if not rotated to shift position
    for nrim, image in enumerate(trajectory_geodesic): #fix the the position of the shifted molecule since the geodesic interpolation does not use pbc, if wanted rotate to xy plane
        image.set_cell(initial_mol.cell)
        if nrim != 0 and alignXY: # do not rotate first image so that it aligns exactly to original initial structure
            R_matrix = rotate_to_XYplane(image,new_surf_indices)
        else:
            R_matrix = np.eye(3) #identity matrix to not rotate
        for index,atom in enumerate(image):
            atom.position = R_matrix.dot(atom.position)
            if index == 0 and alignXY: #if rotateted, each image needs to be shifted
                difference = initial_mol[new_surf_indices[0]].position - image[new_surf_indices[0]].position
            atom.position += difference
    trajectory_geodesic.append(new_final_mol)
    if write_geo:
        new_traj=Trajectory(f'{LEN}_geodesic_interpol.traj',mode='w')
        for image in trajectory_geodesic:
            new_traj.write(image)
    return trajectory_pbc, trajectory_geodesic, removed_idx_list
######### Combination ######### @FThiemann



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Mixing the geodesic (molecule) and idpp/direct (surface) interpolation methods.')
    parser.add_argument('start', type=str, help='POSCAR file of initial molecule.')
    parser.add_argument('end', type=str, help='POSCAR file of final molecule.')
    parser.add_argument('Images', type=int, help='Number of NEB Images to generate in between.')
    parser.add_argument('method', type=str,choices=['idpp','direct'] , help='Interpolation method')
    parser.add_argument('-m','--molind', nargs='+', type=int, help='Manual molecule indices: Single value is used as length of molecule at end of file. Multiple values are used directly as indices.')
    parser.add_argument('-r','--removepbc', action='store_true', help='Remove surface atoms that are shifted due to pbc in geodesic interpolation.')
    parser.add_argument('-a','--alignXY', action='store_true', help='Align geodesic images to XY plane by rotating surface.')
    parser.add_argument('-w','--writegeo', action='store_true', help='Write geodesic interpolation of surface and molecule, mainly for debugging.')
    parser.add_argument('-c','--check', action='store_true', help='Check the separation into surface and molecule.')
    parser.add_argument('-s','--show', action='store_true', help='Show the final merged trajectory.')
    parser.add_argument('-v','--verbose', action='store_true', help='Verbosity of output.')
    parser.add_argument('-i','--intermediate',help='Use a guess for the transition state, will be interpolated from start to I and from I to end')
    global args
    args = parser.parse_args()

    initial_mol = read(args.start)
    final_mol = read(args.end)
    original_constraints = initial_mol.constraints
    LEN = args.Images
    if args.molind is None:
        surf, mol = detect_surf(initial_mol)
        print(f"Surface ({surf.get_chemical_formula()}) and Molecule ({mol.get_chemical_formula()}) were automatically determined.")
        surf_indices = surf.get_tags()
        mol_indices = mol.get_tags()
    elif len(args.molind) == 1:
        mol_indices = list(range(len(initial_mol) - args.molind[0], len(initial_mol)))
        surf_indices = list(range(0, len(initial_mol) - args.molind[0]))
    else:
        mol_indices = args.molind
        surf_indices = [i for i in range(len(initial_mol)) if i not in mol_indices]
    method = args.method

    if args.intermediate:
        in_read = read(args.intermediate)
        vprint(f"Using intermediate. Length of first segment: {(LEN)//2}, second segment: {(LEN-1)//2}") #if LEN is odd, TS is in the middle, if LEN is even, TS is shifted one image to right
        trajectory_pbc1, trajectory_geodesic1, removed_idx_list1 = create_both(initial_mol, in_read, (LEN)//2, method, surf_indices, args.removepbc, args.alignXY, args.writegeo)
        trajectory_pbc2, trajectory_geodesic2, removed_idx_list2 = create_both(in_read , final_mol, (LEN-1)//2, method, surf_indices, args.removepbc, args.alignXY, args.writegeo)
        trajectory_pbc = trajectory_pbc1.copy()
        trajectory_pbc += trajectory_pbc2[1:] #intermediate is already last image of copy, so we can skip it
        trajectory_geodesic = trajectory_geodesic1.copy()
        trajectory_geodesic += trajectory_geodesic2[1:] #intermediate is already last image of copy, so we can skip it
        vprint(f"Total length of geodesic: {len(trajectory_geodesic)}, pbc: {len(trajectory_pbc)}")
    else:
        trajectory_pbc, trajectory_geodesic, removed_idx_list = create_both(initial_mol, final_mol, LEN, method, surf_indices, args.removepbc, args.alignXY, args.writegeo)
    newTrajectory = trajectory_pbc.copy()
    new_traj=Trajectory(f'{LEN}_interpol.traj',mode='w')
    for nr in range(len(trajectory_pbc)):
        atoms_idpp = trajectory_pbc[nr]
        atoms_geodesic = trajectory_geodesic[nr]
        if args.intermediate: #removed indices in geodesic interpolation need to be taken care of when assigning mol indices for geodesic
            if nr <= (LEN)//2+1:
                removed_indices = removed_idx_list1
            else:
                removed_indices = removed_idx_list2
        else:
            removed_indices = removed_idx_list
        modified_mol_indices = []
        for idx in mol_indices: 
            no_removed_before = sum(r < idx for r in removed_indices)
            modified_mol_indices.append(idx - no_removed_before)
        molecule = atoms_geodesic[modified_mol_indices]
        surface = atoms_idpp[surf_indices]
        if nr == 0 and args.check is True: #check the cutoff
            view(surface)
            view(molecule)
        atom_list = [None] * (len(mol_indices) + len(surf_indices)) #merge the molecule and the surface in old ordering of indices
        for current_idx, old_idx in enumerate(mol_indices):
            atom_list[old_idx] = molecule[current_idx]
        for current_idx, old_idx in enumerate(surf_indices):
            atom_list[old_idx] = surface[current_idx]
        merged = Atoms(atom_list, cell=surface.cell, pbc=surface.pbc)
        merged.set_constraint(original_constraints) #set constraints ase they got lost by combination
        newTrajectory[nr] = merged
        new_traj.write(merged)
        N=f'{nr:02d}'
        vprint(f'Writing {N}/POSCAR')
        Path(N).mkdir(parents=True,exist_ok=True)
        newTrajectory[nr].write(f'{N}/POSCAR',format='vasp')
    
    if args.show is True:
        view(newTrajectory)

if __name__ == '__main__':
    main()
