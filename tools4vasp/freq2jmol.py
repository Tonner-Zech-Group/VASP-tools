#!/usr/bin/env python3
from ase.calculators.vasp import Vasp
import os
os.environ['VASP_PP_PATH'] = "/home/patrickm/lib/vasp/ase"

calc = Vasp(restart=True, directory='./')
vibs = calc.get_vibrations()
vibs.write_jmol()

energies = vibs.get_energies()
nVibs = len(energies)
for i in range(nVibs):
    with open("vib-{:03d}.xyz".format(i+1), 'w') as f:
        for frame in vibs.iter_animated_mode(i):
            frame.write(f, format='extxyz')
    print("Frequency #{:03d}: {:10.6f} cm-1".format(i+1, energies[i]))
