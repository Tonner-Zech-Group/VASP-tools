#!/usr/bin/env python3
#
# Script to get a KSPACING from a KPOINTS file and a POSCAR
# by Patrick Melix
# 2023/05/22
#


def get_kspacing():
    from ase import io
    import numpy as np
    mol = io.read('POSCAR')
    cellparams = mol.cell.cellpar()
    r_cell = mol.cell.reciprocal()
    print("The cell parameters are |a|={} |b|={} |c|={} alpha={} beta={} gamma={}".format(*[ round(x,2) for x in cellparams]))
    print("The reciprocal cell is: {}".format(r_cell))
    with open('KPOINTS', 'r') as f:
        kpoints = f.readlines()
    if kpoints[1].strip() != "0":
        raise ValueError("KPOINTS file does not use Automatic Scheme, but only this is supported!")
    kgrid = [ int(k) for k in kpoints[3].split() ]
    assert len(kgrid) == 3, "Expected to find 3 integers in line 4 of KPOINTS file!"
    print("The KGRID from KPOINTS is: {} {} {}".format(*kgrid))
    kspacing = [ np.linalg.norm(r_cell[i])*2*np.pi/kgrid[i] for i in range(3) ]
    #kgrid = [ int(max(1, np.ceil(np.linalg.norm(r_cell[i])*2*np.pi/kspacing))) for i in range(3)]
    print("The corresponding KSPACING for KGRID {} {} {} is {}Å^-1, {}Å^-1, {}Å^-1".format(*kgrid, *[ round(k, 3) for k in kspacing ]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Get a KSPACING from current POSCAR and KPOINTS')
    args = parser.parse_args()
    get_kspacing()
