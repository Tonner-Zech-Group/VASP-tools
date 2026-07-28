#!/usr/bin/env python3
#
# Script to assert proper occupations in VASP calculation using ASE
# by Patrick Melix
# 2021/06/15
#
from __future__ import annotations

import os
import re
from io import TextIOWrapper

import numpy as np
from ase import io
from ase.calculators.vasp.vasp import Vasp

from tools4vasp._fileutils import iter_lines_reversed

# OUTCAR final-energy block, e.g.:
#   free  energy   TOTEN  =       -68.41063650 eV
#   energy  without entropy=      -68.40903650  energy(sigma->0) = ...
_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*(-?\d+\.\d+)")
_E_WO_ENTROPY_RE = re.compile(r"energy\s+without\s+entropy\s*=\s*(-?\d+\.\d+)")

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

def check_vasp_potcar_order(path) -> str | None:
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


def check_vasp_occupations(calc) -> str | None:
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


def _get_entropy_energies(outcar, chunk_size=64 * 1024) -> tuple:
    """Parse the final TOTEN and energy without entropy from an OUTCAR file.

    Returns the energies of the *last* "free energy TOTEN" line that is
    followed by an "energy without entropy" line within four lines, i.e.
    the final electronic step. The file is read backwards from the end,
    so even multi-GB OUTCARs only have their tail touched. Earlier
    implementations only looked at the last 200 lines (crashing for large
    systems where the block sits further from the end) and took the first
    match instead of the final one (issue #24).

    Input Parameters
    ----------------
    outcar : str
        Path to the OUTCAR file

    chunk_size : int
        Number of bytes per backwards read step, must be positive

    Returns
    -------
    Tuple of (TOTEN, energy without entropy) in eV.
    """
    e_wo_entropy = None
    lines_above_entropy = 0
    with open(outcar, "rb") as f:
        for raw_line in iter_lines_reversed(f, chunk_size=chunk_size):
            line = raw_line.decode(errors="replace")
            if e_wo_entropy is None:
                entropy_match = _E_WO_ENTROPY_RE.search(line)
                if entropy_match:
                    e_wo_entropy = float(entropy_match.group(1))
                    lines_above_entropy = 0
            else:
                lines_above_entropy += 1
                toten_match = _TOTEN_RE.search(line)
                if toten_match:
                    return float(toten_match.group(1)), e_wo_entropy
                if lines_above_entropy >= 4:
                    # unpaired entropy line — keep searching earlier ones
                    e_wo_entropy = None
    raise ValueError(f"Could not parse TOTEN/entropy from {outcar}")


def check_vasp_electronic_entropy(path, calc, limit=0.001) -> str | None:
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
        print(f"Integer occupation check returned: {ret}")
        outcar = os.path.join(path, "OUTCAR")
        toten, e_wo_entropy = _get_entropy_energies(outcar)
        entropy = toten - e_wo_entropy
        mol = io.read(os.path.join(path, 'CONTCAR'))
        entropy_per_atom = entropy / len(mol)
        if not entropy_per_atom < limit:
            return f"Entropy per atom is {entropy_per_atom}eV"
    return
        



def run(path):
    """Check VASP run for proper occupations and convergence."""
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


def main():
    """CLI entry point: parse arguments and call run()."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Assert proper occupations and SCF/geometry convergence in a VASP "
                    "calculation using ASE.",
        epilog="Example: vaspcheck /path/to/vasp/run")
    parser.add_argument("path", type=str, nargs="?", default="./",
                        help="Path to VASP files (default: ./)")
    args = parser.parse_args()
    run(args.path)


if __name__ == "__main__":
    main()
