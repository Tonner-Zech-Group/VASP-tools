#!/usr/bin/env python3
from ase import io
import numpy as np

poscar = io.read('POSCAR')

add = np.loadtxt('MODECAR')

poscar.write('poscar+modecar.xyz')

poscar.positions += add

poscar.write('poscar+modecar.xyz', append=True)
