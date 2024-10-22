#!/usr/bin/env python3
from ase import io
import numpy as np

def main():
    """
    Add the displacements from a MODECAR file to the positions in a POSCAR file.
    """
    
    poscar = io.read('POSCAR')

    add = np.loadtxt('MODECAR')

    poscar.write('poscar+modecar.xyz')

    poscar.positions += add

    poscar.write('poscar+modecar.xyz', append=True)


if __name__ == '__main__':
    main()
