#!/usr/bin/env python3
#
# Script to assert proper occupations in VASP calculation using ASE
# by Patrick Melix
# 2021/06/15
#
from io import TextIOWrapper
from ase.calculators.vasp.vasp import Vasp
from ase import io
import os
import numpy as np
from typing import Optional
import subprocess

def _get_elements_from_outcar(f: TextIOWrapper) -> list:
    """Get elements from OUTCAR file.
    
    Input Parameters
    ----------------

    Returns
    -------
    List of element names
    """
    lines = []
    for line in f:
        if "POSCAR" in line:
            elements_poscar = line.split(':')[1].strip().split()
            break
        elif "POTCAR:" in line:
            lines.append(line)
    if len(lines) == 1:
        elements_potcar = [lines[0].split(':')[-1].strip().split()[1]]
    else:
        assert len(lines) % 2 == 0, "POTCAR: lines are not even"
        elements_potcar = [ line.split(':')[-1].strip().split()[1] for line in lines[0:int(len(lines)/2)] ]
    # clean up element names by removing _* suffixes for PAW potentials
    for i in range(len(elements_potcar)):
        if '_' in elements_potcar[i]:
            elements_potcar[i] = elements_potcar[i].split('_')[0]
        if '_' in elements_poscar[i]:
            elements_poscar[i] = elements_poscar[i].split('_')[0]
    return elements_poscar, elements_potcar

def check_vasp_potcar_order(path) -> Optional[str]:
    """Check VASP calculations for proper POTCAR order.

    Input Parameters
    ----------------
    path : str
        Path to VASP files

    Returns
    -------
    None if everything is good or a string with a message if a problem occurs.
    """
    assert os.path.isdir(path), "Given path is not a directory"
    with open(os.path.join(path, "OUTCAR"), "r") as f:
        elements_poscar, elements_potcar = _get_elements_from_outcar(f)
    if elements_poscar != elements_potcar:
        return "POTCAR order does not match POSCAR order"
    else:
        return None


def check_vasp_occupations(calc) -> Optional[str]:
    """Check VASP calculations for non-integer occupations.
    Input Parameters
    ----------------
    calc : ASE Vasp calculator
        Vasp calculator object

    Returns
    -------
    None if everything is good or a string with a message if a problem occurs.
    """
    xml = calc._read_xml()
    if xml.get_spin_polarized():
        spins = [0, 1]
        electrons = 1.0
    else:
        spins = [0]
        electrons = 2.0

    nkpoints = len(xml.get_ibz_k_points())
    for s in spins:
        for i in range(nkpoints):
            occ = xml.get_occupation_numbers(i, s)
            if occ is None:
                msg = "No occupations found in vasprun.xml for kpoint" +\
                      " #{} and spin {}!"
                return msg.format(i, s)
            test = np.where(np.logical_or(occ == electrons, occ == 0.0), 1, 0)
            if not test.all():
                return "Bad Occupation found"
    return


def check_vasp_electronic_entropy(path, calc, limit=0.001) -> Optional[str]:
    """Check if the electronic entropy is larger than limit.
    
    Input Parameters
    ----------------
    path : str
        Path to VASP files

    calc : ASE Vasp calculator
        Vasp calculator object

    limit : float
        Limit for the electronic entropy in eV/atom

    Returns
    -------
    None if everything is good or a string with a message if a problem occurs.
    """
    ret = check_vasp_occupations(calc)
    # non-integer occupations
    if ret:
        print("Integer occupation check returned: {:}".format(ret))
        outcar = os.path.join(path, "OUTCAR")
        cmd = 'tail -n 200 {} | grep -A 4 "TOTEN"'.format(outcar)
        res = subprocess.check_output([cmd], shell=True).decode('utf-8')
        res = res.split('\n\n')
        toten = float(res[0].split()[-2])
        e_wo_entropy = float(res[1].split()[3])
        entropy = toten - e_wo_entropy
        mol = io.read(os.path.join(path, 'CONTCAR'))
        entropy_per_atom = entropy / len(mol)
        if not entropy_per_atom < limit:
            return "Entropy per atom is {:}eV".format(entropy_per_atom)
    return
        



def main(path):
    assert os.path.isdir(path), "Given path is not a directory"
    calc = Vasp(directory=path)
    ret = check_vasp_electronic_entropy(path, calc)
    if ret:
        print(ret)
        return
    print("Seems like there are no bad occupations (only last step).")
    

    if not calc.read_convergence():
        print("Either SCF or GO did not converge!")
    else:
        print("No convergence issues found (only last step).")
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Check VASP run for proper occupations')
    parser.add_argument(
        'path',
        type=str,
        help='path to VASP files',
        default='./')
    args = parser.parse_args()
    main(args.path)
