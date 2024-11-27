#!/usr/bin/env python3
"""
A script to split a VASP frequency calculation into individual parts and recombine the results.

No dipoles supported yet, therefore no IR!

Can be used as a command line tool or as a python module.

Usage as a command line tool
-----
>>> split_vasp_freq.py split 10
        Split the VASP frequency calculation into parts with 10 atoms each.

>>> split_vasp_freq.py combine
        Combine the results of a split VASP frequency calculation.

Usage as a python module
-----
>>> from split_vasp_freq import split, combine
>>> split('POSCAR', 10) # Split the VASP frequency calculation into parts with 10 atoms each.
>>> vib = combine('POSCAR', return_vibrations=True, sanity_checks=[ function1, function2, ...]) # Recombine the results
>>> zpe = vib.get_zero_point_energy()

"""

import numpy as np
from math import ceil
from ase import io
from ase.constraints import FixAtoms
from ase.vibrations.data import VibrationsData
from ase.vibrations import Vibrations
from ase.geometry import find_mic
import os
import glob
from typing import Union

def log(msg, verbose=True):
    if verbose:
        print(msg)

def read_input_structure(input_file, verbose=True, use_only_indices=None) -> dict:
    """
    Read the input structure and return atoms, constraints and indices of free atoms.

    Parameters
    ----------
    input_file : str
        Input file with structure, e.g. POSCAR.

    verbose : bool
        Print some information about the structure.

    use_only_indices : list of int
        List of indices of atoms to be moved. If None, the constraints from the input_file are used. Defaults to None. Can be used to only move a subset of atoms, even if the calculations moved more atoms. Beware to not give indices that were restrained in the original calculation, otherwise building the Vibrations object will fail.

    Returns
    -------
    dict
        Dictionary with keys 'atoms', 'constraints', 'indices', 'n_free_atoms'
    """
    res = {}
    assert os.path.isfile(input_file), "File {:} does not exist".format(input_file)
    log("Reading {}".format(input_file), verbose=verbose)
    res['atoms'] = io.read(input_file)
    n_atoms = len(res['atoms'])
    log("Number of atoms: {}".format(n_atoms), verbose=verbose)
    if use_only_indices:
        log("Setting new additional constraint based on use_only_indices", verbose=verbose)
        new_constraint = FixAtoms(np.delete(np.arange(n_atoms), use_only_indices))
        res['atoms'].set_constraint(new_constraint)
    res['constraints'] = res['atoms'].constraints
    res['indices'] = np.arange(n_atoms)
    res['n_free_atoms'] = n_atoms
    if res['constraints']:
        log("Found {} constraints!".format(len(res['constraints'])), verbose=verbose)
        for const in res['constraints']:
            if not isinstance(const, FixAtoms):
                raise RuntimeError("Only FixAtoms constraints are supported!")
            res['indices'] = np.delete(res['indices'], const.index)
        res['n_free_atoms'] = len(res['indices'])
    log("Remaining unconstrained atoms: {}".format(res['n_free_atoms']), verbose=verbose)
    return res

def split(input_file, n_atoms_per_calc, cwd=".", verbose=True) -> None:
    """Split a VASP frequency calculation into individual parts.
    
    Parameters
    ----------    
    input_file : str
        Input file with structure, e.g. POSCAR

    n_atoms_per_calc : int
        Number of atoms to be moved in each partial calculation

    cwd : str
        Working directory, defaults to '.'

    verbose : bool
        Print information to stdout?
    """
    info = read_input_structure(input_file, verbose=verbose)
    log("Number of atoms per calculation: {}".format(n_atoms_per_calc), verbose=verbose)
    n_calcs = ceil(info['n_free_atoms'] / n_atoms_per_calc)
    log("Number of calculations: {}".format(n_calcs), verbose=verbose)
    log("Will now create {} subfolders with corresponding POSCARs".format(n_calcs), verbose=verbose)
    for i_calc in range(n_calcs):
        folder_name = "freq_{:03d}".format(i_calc+1)
        folder_path = os.path.join(cwd, folder_name)
        if os.path.isdir(folder_path):
            raise RuntimeError("Folder {} already exists!".format(folder_path))
        os.mkdir(folder_path)
        tmp_atoms = info['atoms'].copy()
        tmp_constraints = info['constraints'].copy()
        move_indices = info['indices'][i_calc*n_atoms_per_calc:(i_calc+1)*n_atoms_per_calc]
        fix_indices = np.setdiff1d(info['indices'], move_indices)
        add_const = FixAtoms(fix_indices)
        tmp_constraints.append(add_const)
        tmp_atoms.set_constraint(tmp_constraints)
        tmp_atoms.write(os.path.join(folder_path, "POSCAR"))


def get_nfree_delta(incar_path, verbose=True) -> int:
    """
    Read the NFREE parameter from an INCAR file.
    """
    assert os.path.isfile(incar_path), "INCAR file does not exist!"
    n_displacements = delta = None
    with open(incar_path, 'r') as f:
        # get n_displacements from INCAR
        for line in f:
            if "nfree" in line.lower():
                n_displacements = int(line.split('=')[1].strip()[0])
                if n_displacements not in [2, 4]:
                    raise ValueError("NFREE must be 2 or 4, only those are supported by ASE!")
            elif "potim" in line.lower():
                delta = float(line.split('=')[1].split('#')[0].strip())
            if n_displacements and delta:
                break
    if n_displacements == 2:
        log("Central Differences (NFREE=2) detected.", verbose=verbose)
    elif n_displacements == 4:
        log("Four Displacements (NFREE=4) detected.", verbose=verbose)
    else:
        raise RuntimeError("Could not find NFREE in INCAR!")
    if delta:
        log("Found a delta value (POTIM) of {} Angstrom.".format(delta), verbose=verbose)
    else:
        raise RuntimeError("Could not find POTIM in INCAR!")
    return n_displacements, delta

def combine(input_file, cwd=".", verbose=True, return_vibrations=False, sanity_checks=[], use_only_indices=None) -> Union[None, Vibrations]:
    """
    Combine the results of a split VASP frequency calculation.

    No dipoles supported yet, therefore no IR!

    Parameters
    ----------
    input_file : str
        Input file with structure, e.g. POSCAR (INCAR needs to be in the same folder)

    cwd : str
        Working directory, defaults to '.'

    verbose : bool
        Print information to stdout?
    
    return_vibrations : bool
        Return the ASE Vibrations object

    sanity_checks : list of functions
        List of functions to perform custom sanity checks on all jobs. Each function must take a single argument, which is the path to the folder of the job. It's return value should be True if the job is ok and False otherwise.

    use_only_indices : list of int
        List of indices of atoms to be moved. If None, the constraints from the input_file are used. Defaults to None. Can be used to only move a subset of atoms, even if the calculations moved more atoms. Beware to not give indices that were restrained in the original calculation, otherwise building the Vibrations object will fail.

    Returns
    -------
    None or ase.vibrations.Vibrations object
    """
    name = os.path.join(cwd, "freq")
    info = read_input_structure(input_file, verbose=verbose, use_only_indices=use_only_indices)
    n_free, delta = get_nfree_delta(os.path.join(os.path.dirname(input_file), "INCAR"), verbose=verbose)
    dirs = [ os.path.normpath(d) for d in glob.glob(os.path.join(cwd, "{}_*".format("freq"))) if os.path.isdir(d) ]
    dirs.sort()
    n_calcs = len(dirs)
    log("Found {} subfolders with frequency job parts.".format(n_calcs), verbose=verbose)

    # Setup the ASE Vibrations Object
    vib = Vibrations(info['atoms'], indices=info['indices'], name=name, delta=delta, nfree=n_free)

    ref_pos = info['atoms'].positions
    # Read the data using ase.calculators.vasp.get_vibrations()
    results = { 'eq': {} } # store the equilibrium data in a separate dict
    vibname2dir = {} # map vibration name to directory and index
    for i_dir, dir in enumerate(dirs):
        log("Reading data from {}".format(dir), verbose=verbose)
        results[dir] = read_input_structure(os.path.join(dir, "POSCAR"), verbose=False)
        #load job using ase
        if not os.path.isfile(os.path.join(dir, "vasprun.xml")):
            log("WARNING: No vasprun.xml found in {}, skipping.".format(dir), verbose=verbose)
            continue
        results[dir]['atoms'] = io.read(os.path.join(dir, "vasprun.xml"), format='vasp-xml', index=slice(0, None))
        # basic check if job succeeded
        assert results[dir]['atoms'][0].calc is not None, "Could not read VASP calculation from {}".format(dir)
        # perform custom sanity checks
        for func in sanity_checks:
            log("Running custom sanity check {} on {}".format(func.__name__, dir), verbose=verbose)
            assert func(dir), "Custom sanity check {} failed for {}".format(func.__name__, dir)
        # in the first round save equilibrium forces otherwise
        # xml has originial structure as first entry, so we need to remove it
        # make sure to always use apply_constraint=False, otherwise ASE sets the forces to zero!
        if i_dir == 0:
            vibname2dir['eq'] = ('eq', slice(0,None)) # only one item, all idxs needed
            results['eq']['atoms'] = results[dir]['atoms'][0].copy()
            results['eq']['forces'] = results[dir]['atoms'][0].get_forces(apply_constraint=False)
        else: # check that equilibrium forces are the same
            assert np.allclose(results['eq']['atoms'].positions, results[dir]['atoms'][0].positions, atol=1e-6), "Equilibrium structures are not the same!"
            assert np.allclose(results['eq']['forces'], results[dir]['atoms'][0].get_forces(apply_constraint=False), atol=1e-6), "Equilibrium forces are not the same!"

        del results[dir]['atoms'][0]
        results[dir]['n_displacements'] = len(results[dir]['atoms'])
        #log("Number of displacements: {}".format(results[dir]['n_displacements']))
        #log("Number of free atoms: {}".format(results[dir]['n_free_atoms']))
        assert results[dir]['n_free_atoms'] == len(results[dir]['atoms']) / 3 / n_free, "Number of free atoms does not match!"
        #results[dir]['atoms'].calc = Vasp(restart=True, directory=dir)
        #results[dir]['atoms'].calc._read_xml()
        # get forces for all displacements
        results[dir]['forces'] = [ ats.get_forces(apply_constraint=False) for ats in results[dir]['atoms'] ]
        # Note that indices must be the same as in the original structure! So no resorting needed.
        results[dir]['indices'] = VibrationsData.indices_from_constraints(results[dir]['atoms'][0])
        #log("Indices: {}".format(results[dir]['indices']))
        #results[dir]['vibrations'] = results[dir]['atoms'].calc.get_vibrations()
        # make names for these displacements as <index><x/y/z><++/+/-/-->
        results[dir]['names'] = []
        indices = results[dir]['indices']
        for i in range(len(results[dir]['atoms'])):
            pos_moving = results[dir]['atoms'][i].positions[indices]
            diff_vecs, diffs = find_mic(pos_moving - ref_pos[indices], cell=info['atoms'].cell, pbc=True)
            # take care of central and four displacements
            tmp_delta = delta
            if n_free == 4:
                #check if we find a difference of delta*2
                if np.any(np.abs(diffs - delta*2) < 0.01):
                    tmp_delta = delta*2
            # find the index where position difference is close to delta or delta*2
            # this is the index of the atom that was moved
            at_idx = np.argmin(np.abs(diffs - tmp_delta))
            assert np.abs(diffs[at_idx] - tmp_delta) < 0.01, "Could not find the moved atom!"
            real_at_index = indices[at_idx]
            # get the dimension of the movement
            dim_idx = np.argmin(np.abs(np.abs(diff_vecs[at_idx]) - tmp_delta))
            # get the direction of the movement
            direction = '+' if diff_vecs[at_idx][dim_idx] > 0 else '-'
            # double the direction if we have four displacements and this is a double displacement
            if tmp_delta == delta*2:
                direction += direction
            #log("Atom {} moved in direction {} in dimension {}".format(real_at_index, direction, dim_idx))
            results[dir]['names'].append("{}{}{}".format(real_at_index, 'xyz'[dim_idx], direction))
            vibname2dir[results[dir]['names'][-1]] = (dir, i) # store dir and index
            #log("Name: {}".format(results[dir]['names'][-1]))
        # make sure the names are unique
        assert len(set(results[dir]['names'])) == len(results[dir]['atoms']), "Names are not unique!"

    # Now we have all the data, we can start to write it into the json files expected by ASE
    log("Writing data to json files.", verbose=verbose)
    #cache = MultiFileJSONCache(name)
    vib.cache.clear() # remove all previous data
    for disp, ats in vib.iterdisplace():
        assert disp.name in vibname2dir, "Could not find displacement {} in results!".format(disp.name)
        # save the forces as <name>.<label>.json
        # content of this file is a dict {'forces': np.ndarray}
        dir, idx = vibname2dir[disp.name]
        dct = {'forces': results[dir]['forces'][idx]}
        with vib.cache.lock(disp.name) as handle:
            if handle is None:
                raise RuntimeError("There seems to be leftovers from {}".format(disp.name))
            handle.save(dct)
    #vib.combine()
    vib.read()
    if verbose:
        vib.summary()

    # return the ASE Vibrations object if requested
    if return_vibrations:
        return vib
    else:
        return None


def load_vibrations(input_file, cwd=".", verbose=True, use_only_indices=None) -> Vibrations:
    """Load an already combined vibrational calculation.
    
    Parameters
    ----------
    input_file : str
        Input file with structure, e.g. POSCAR
    cwd : str
        Working directory, defaults to '.'
    verbose : bool
        Print information to stdout?
    use_only_indices : list of int
        List of indices of atoms to be moved. If None, the constraints from the input_file are used. Defaults to None. Can be used to only move a subset of atoms, even if the calculations moved more atoms. Beware to not give indices that were restrained in the original calculation, otherwise building the Vibrations object will fail.
    Returns
    -------
    None or ase.vibrations.Vibrations object
    """
    name = os.path.join(cwd, "freq")
    info = read_input_structure(input_file, verbose=verbose, use_only_indices=use_only_indices)
    n_free, delta = get_nfree_delta(os.path.join(os.path.dirname(input_file), "INCAR"), verbose=verbose)
    # Setup the ASE Vibrations Object
    vib = Vibrations(info['atoms'], indices=info['indices'], name=name, delta=delta, nfree=n_free)
    return vib

def export_jmol(vib: Vibrations, output_file: str) -> None:
    """Write the vibrational modes to a file that can be read by Jmol."""
    with open(f"{output_file}.xyz", 'w') as f:
        vib._write_jmol(f)

def export_xyz_traj(vib: Vibrations, output_file: str, index: int=None) -> None:
    """Write the vibrational modes to a xyz trajectory file.
    
    Parameters
    ----------
    vib : ase.vibrations.Vibrations
        The vibrational modes object
    output_file : str
        Output file name
    index : integer or None
        Index of the mode to write. If None, all modes are written.
    """
    def _write_mode(vib_data, index, output_file):
        with open("{}_{:03d}.xyz".format(output_file, index+1), 'w') as f:
            for frame in vib_data.iter_animated_mode(index):
                frame.write(f, format='extxyz')

    vib_data = vib.get_vibrations()
    if isinstance(index, int):
        _write_mode(vib_data, index, output_file)
    else:
        nVibs = len(vib.get_energies())
        for i in range(nVibs):
            _write_mode(vib_data, i, output_file)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Script to split a VASP frequency calculation into individual parts and recombine the results.')
    parser.add_argument('task', choices=['split', 'combine', 'write_jmol', 'write_xyz_traj'],
                        help='Task to be performed')
    parser.add_argument('n_atoms', nargs='?', type=int,
                        help='Number of atoms to be moved in each partial calculation', default=10)
    parser.add_argument('-input', nargs='?', type=str,
                        help='Input file with structure, e.g. POSCAR', default='POSCAR')
    parser.add_argument('--output', help='Output file name, will be appended with .xyz', default='vib')
    parser.add_argument('--silent', help="Don't print to stdout", default=False, action='store_true')

    args = parser.parse_args()
    if args.task == 'split':
        split(args.input, args.n_atoms, verbose=not args.silent)
    elif args.task == 'combine':
        combine(args.input, verbose=not args.silent)
    elif args.task == 'write_jmol':
        vib = load_vibrations(args.input, verbose=not args.silent)
        export_jmol(vib, args.output)
    elif args.task == 'write_xyz_traj':
        vib = load_vibrations(args.input, verbose=not args.silent)
        export_xyz_traj(vib, args.output, index=None)