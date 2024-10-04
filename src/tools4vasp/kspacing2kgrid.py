#!/usr/bin/env python3
#
# Script to get a KGrid from a KSPACING value and a POSCAR
# by Patrick Melix
# 2023/05/22
#


def get_kgrid(kspacing):
    from ase import io
    import numpy as np
    mol = io.read('POSCAR')
    cellparams = mol.cell.cellpar()
    r_cell = mol.cell.reciprocal()
    print("The cell parameters are |a|={} |b|={} |c|={} alpha={} beta={} gamma={}".format(*[ round(x,2) for x in cellparams]))
    print("The reciprocal cell is: {}".format(r_cell))
    kgrid = [ int(max(1, np.ceil(np.linalg.norm(r_cell[i])*2*np.pi/kspacing))) for i in range(3)]
    print("The corresponding KGRID for KSPACING {}Å^-1 is: {} {} {}".format(kspacing, *kgrid))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Get a KGrid from current POSCAR and a KSPACING value')
    parser.add_argument('kspacing', type=float, help='KSPACING value')
    args = parser.parse_args()
    get_kgrid(args.kspacing)
