#!/usr/bin/env python3

def addMODECAR():
    """
    Add the displacements from a MODECAR file to the positions in a POSCAR file.
    """
    from ase import io
    import numpy as np

    poscar = io.read('POSCAR')

    add = np.loadtxt('MODECAR')

    poscar.write('poscar+modecar.xyz')

    poscar.positions += add

    poscar.write('poscar+modecar.xyz', append=True)


if __name__ == '__main__':
    addMODECAR()
