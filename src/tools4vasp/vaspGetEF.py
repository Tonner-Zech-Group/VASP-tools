#!/usr/bin/env python3
"""
Combine VASP xml files of all numerical subdirectories and the parent directory and plot the forces and energies.
Useful for comparing the convergence of forces and energies in VASP calculations.
"""

from natsort import natsorted
import glob
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from ase.io.vasp import read_vasp_xml
from typing import Tuple, List, Dict


print_format = "{:5d} {:20.8f} {:20.6f} {:20.6g}"

def get_max_f(atoms) -> float:
    """Get the maximum force from an ASE atoms object."""
    return np.sqrt((atoms.get_forces()**2).sum(axis=1).max())

def read_xml(path, verbose=False, write_fe=False) -> Tuple[List[float]]:
    """Reads the xml file from a VASP calculation and returns the max forces and energies.

    Input Parameters
    ----------------
    path : str
        Path of the VASP xml file
    verbose : bool
        Print the values of the forces and energies
    write_fe : bool
        Write the forces and energies to fe.dat


    Returns
    -------
    Tuple(List(float), List(float))
        List of max forces and energies
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("No such file: {}".format(path))

    print(path)
    traj = read_vasp_xml(path, index=slice(0,None))

    if write_fe:
        fe_path = os.path.join(os.path.dirname(os.path.abspath(path)), 'fe.dat')
        fe = open(fe_path, 'w')

    forces = []
    energies = []
    for i, atoms in enumerate(traj):
        energies.append(atoms.get_potential_energy())
        forces.append(get_max_f(atoms))
        if verbose:
            print(print_format.format(i, forces[-1], energies[-1], energies[-1]-energies[0]))
        if write_fe:
            fe.write(print_format.format(i, forces[-1], energies[-1], energies[-1]-energies[0])+'\n')

    if write_fe:
        fe.close()
    return forces, energies

def get_all_xmls(path, vervose=False) -> List[str]:
    """Get all xml files in the path and numerical subdirectories.

    Input Parameters
    ----------------
    path : str
        Path to the VASP calculation

    Returns
    -------
    List(str)
        List of all vasprun.xml files
    """
    files = natsorted(glob.glob(os.path.join(path, '*/vasprun.xml'), recursive=True))
    if os.path.isfile(os.path.join(path, 'vasprun.xml')):
        files.append(os.path.join(path, 'vasprun.xml'))
    return files

def process_all_xmls(path, verbose=False, write_json=False) -> Dict[str, List[float]]:
    """Process all xml files in the path and numerical subdirectories.

    Input Parameters
    ----------------
    path : str
        Path to the VASP calculation
    verbose : bool
        Print the values of the forces and energies
    write_json : bool
        Write the forces and energies to fe-combined.json


    Returns
    -------
    Dict: {'force': List(float), 'energy': List(float)}
        Dictionary of forces and energies

    """
    files = get_all_xmls(path, verbose)

    data = []
    for f in files:
        folder = os.path.dirname(os.path.abspath(f))
        if (not os.path.basename(folder).isdigit()) and (not os.path.realpath(folder) == os.path.realpath(os.getcwd())):
            if verbose:
                print("Not using {}".format(folder))
            continue
        if os.path.isfile(os.path.join(folder,'fe.dat')):
            if verbose:
                print("Adding {}".format(folder))
        else:
            print("Generating fe.dat in {}".format(folder))
            f, e = read_xml(f, verbose=verbose, write_fe=True)
            assert os.path.isfile(os.path.join(folder,'fe.dat')), "Problem generating fe.dat in {:}".format(folder)
        to_add = np.loadtxt(os.path.join(folder,'fe.dat'))
        # check if only one entry
        if to_add.shape == (4,):
            to_add = to_add.reshape(1,4)
        assert to_add.shape[1] == 4, "Problem with the shape of the data in {:}".format(folder)
        data.append(to_add)
        if verbose:
            print("Found {} values".format(len(data[-1])))
    combined = {}
    combined['force'] = []
    combined['energy'] = []
    for d in data:
        # filter one entry runs
        if d.shape == (4,):
            combined['force'].append(d[1])
            combined['energy'].append(d[2])
        else:
            combined['force'].extend(d[:,1].tolist())
            combined['energy'].extend(d[:,2].tolist())
    if write_json:
        with open(os.path.join(path,'fe-combined.json'),'w') as f:
            json.dump(combined, f)
    return combined


def plot_fe(combined, filename, lw=2, show=False) -> None:
    """Plot the forces and energies."""
    nItems = len(combined['force'])
    xAxis = list(range(1, nItems+1))
    fig, ax1 = plt.subplots()
    plt.xlabel('Step #')# [Å]
    color = 'black'
    ax1.set_ylabel(r'$\Delta E$ [eV]', color=color)
    #plt.ylim([-10**exp,10**exp])
    #plt.yscale('symlog')
    #plt.gca().yaxis.grid(True)
    ax1.plot(xAxis, combined['energy'], color=color, ls='-', lw=lw)
    color = 'grey'
    ax2 = ax1.twinx()
    ax2.grid(None)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'max($F$) [eV Å$^{-1}$]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(xAxis, combined['force'], color=color, ls='-', lw=lw)
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd()
    combined = process_all_xmls(path, verbose=True, write_json=True)
    plot_fe(combined, 'fe-combined.png')
    print('...Done!')

