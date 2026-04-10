#!/usr/bin/python
#
# Script to split an adsorbate-surface-complex into surface and molecule. 
# by Jakob Schramm
# 2026/04/09
#
from ase import Atoms
from ase.io import read
from ase.visualize import view
from ase.constraints import FixAtoms
import numpy as np
import matplotlib.pyplot as plt

def count_z_voxels_using_window(z_cell,step,window,z_coords):
    # Function that screens z_coordinates and counts occurenes using a sliding window
    points = list(np.arange(0.0, z_cell, step)) + [z_cell]
    counts = []
    for pos in range(len(points)):
        count = sum(1 for x in z_coords if points[pos] <= x < points[pos]+window)
        counts.append(count)
    return points, counts

def find_plateaus(data):
    # Function that identifies plateaus in counts
    plateaus = []
    plat_height = []
    i = 0
    n = len(data)
    while i < n:
        start = i
        while i + 1 < n and data[i] == data[i + 1]:
            i += 1
        end = i
        if end > start and data[start] > 0:
            plateaus.append((start, end))
            plat_height.append(data[i])
        i += 1
    return plateaus, plat_height

def add_low_peaks(counts, plateaus, plat_height):
    # Function that adds peaks (all atoms at same height and below dirst step of 0.05 so that they are not captured by sliding window) that are at the bottom of slab
    for i,height in enumerate(counts[:10]): ### to include peaks at the bottom of the slab (< 0.5 Å)
        if height > 0 and not any(start <= i <= end for start, end in plateaus):
            plateaus.insert(0,(i,i))
            plat_height.insert(0,height)
    return plateaus, plat_height

def get_no_atoms_in_layer(plat_height, no_atoms):
    # Function that determines the number of atoms in a layer by checking the most occuring and heighest plateau height 
    occurence = {}
    for x in plat_height:
        occurence[x] = occurence.get(x, 0) + 1
    best_value = 0
    best_count = -1
    for value, count in occurence.items():
        if value >= max(9,no_atoms/20): ### to only count layers with at least 9 atoms or one twentieth of all atoms - arbitrary values that yielded good results in tests
            if count > best_count or (count == best_count and value > best_value):
                best_value = value
                best_count = count
    return best_value

def get_layers(points,window,step,plateaus,plat_height,no_atoms_in_layer):
    # Function that determines the layer and its position from the plateaus 
    layers = []
    if no_atoms_in_layer >= 6:
        for pos,plat in enumerate(plateaus):
            if plat_height[pos] == no_atoms_in_layer:
                if plat[1] >= 10: ### peaks at the bottom of the slab (< 0.5 Å) need to be treated differently
                    layer = points[plat[0]]+window
                else:
                    layer = points[plat[1]]+step
                layers.append(layer)
    return layers

def separate_surf_and_mol(symbol, initial_mol, layers, surf, mol):
    # Function that separates the surface and molecule in two Atoms objects while maintaining the constraints and adding old indices to tag information
    for atom in initial_mol:
        if atom.symbol == symbol:
            is_frozen = any(isinstance(c, FixAtoms) and atom.index in c.index for c in initial_mol.constraints)
            if len(layers) > 0 and max(layers) <= 0.5 and atom.position[2] <= max(layers): ### to include non-empty layers that are on the bottom (< 0.5 Å) of the slab like hydrogen saturation layers
                atom.tag = atom.index
                surf += atom
                if is_frozen:
                    new_index = len(surf) - 1
                    surf.constraints.append(FixAtoms(indices=[new_index]))
            elif len(layers) > 1 and atom.position[2] <= max(layers): ### to not include single layers like flat-laying polycyclic hydrocarbons 
                atom.tag = atom.index
                surf += atom
                if is_frozen:
                    new_index = len(surf) - 1
                    surf.constraints.append(FixAtoms(indices=[new_index]))
            else:
                atom.tag = atom.index
                mol += atom
                if is_frozen:
                    new_index = len(mol) - 1
                    mol.constraints.append(FixAtoms(indices=[new_index]))
    return surf, mol

def detect_surf(initial_mol,plot=False):
    """
    Main function for detecting the surface and the molecule.
    Makes use of layered structure of surface and tolerates minor distortions,
    includes saturation layer at the bottom of slab (between 0 and 0.5 Å), 
    but otherwise assumes more then one layer and at least 9 atoms (3x3) or one twentieth of all atoms to be in one layer.
    
    Returns:
        Two ase.Atoms objects - surf that contains surface and mol that contains molecule.
        Constraints are kept and old indices are written to tags so when combining surf and mol the old ordering can be restored.
    """
    z_cell = initial_mol.get_cell()[2][2]
    step = 0.05 ### arbitrary parameter for sweeping resolution - yielded good results in tests 
    window = 1.1 ### arbitrary parameter for sweeping window width to even detect slightly distorted layers - yielded good results in tests
    symbols = list(dict.fromkeys(initial_mol.get_chemical_symbols()))
    no_atoms = len(initial_mol)
    surf = Atoms(cell=initial_mol.cell.copy(), pbc=initial_mol.pbc)
    mol = Atoms(cell=initial_mol.cell.copy(), pbc=initial_mol.pbc)
    for symbol in symbols:
        z_coords = sorted([atom.position[2] for atom in initial_mol if atom.symbol == symbol])
        points, counts = count_z_voxels_using_window(z_cell,step,window,z_coords)
        plateaus, plat_height = find_plateaus(counts)
        plateaus, plat_height = add_low_peaks(counts, plateaus, plat_height)
        no_atoms_in_layer = get_no_atoms_in_layer(plat_height, no_atoms)
        layers = get_layers(points,window,step,plateaus,plat_height,no_atoms_in_layer)
        surf, mol = separate_surf_and_mol(symbol, initial_mol, layers, surf, mol)
        #print(symbol, plateaus, plat_height, no_atoms_in_layer, layers) ### for debugging
        if plot:
            plt.plot(points,counts,label=symbol)
    if plot:
        plt.plot([atom.position[2] for atom in surf],[-1 for i in range(len(surf))],"o",color="red",label="surf")
        plt.plot([atom.position[2] for atom in mol],[-1 for i in range(len(mol))],"o",color="blue",label="mol")
        plt.ylabel("Counts")
        plt.xlabel("z-Coordinate [Å]")
        plt.legend()
        plt.show()
    return surf, mol

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split adsorbate-surface-complex into surface and molecule.')
    parser.add_argument('POSCAR_asc', type=str, help='POSCAR file of adsorbate-surface-complex')
    parser.add_argument('-p','--plot', action='store_true', help='Plot z-count distribution')
    parser.add_argument('-v','--view', action='store_true', help='View surface and molecule')
    global args
    args = parser.parse_args()
    
    initial_mol = read(args.POSCAR_asc)
    surf, mol = detect_surf(initial_mol,args.plot)
    if args.view: 
        view(surf)
        view(mol)
    surf.write("POSCAR_surf")
    mol.write("POSCAR_mol")

if __name__ == '__main__':
    main()
