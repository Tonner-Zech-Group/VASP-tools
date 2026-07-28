#!/usr/bin/env python3
#
# Script to convert ELFCAR files to cube files
# by Patrick Melix
# 2022/04/04
#
# You can import the module and then call .main() or use it as a script
import os

import numpy as np
from ase.io.cube import write_cube
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Elfcar


def elf2cube(inFiles, outFiles, verbose=True, return_integrals=False, return_spin_integrals=False):
    assert len(inFiles) == len(outFiles), "Number of input and output files must be equal!"
    integrals = []
    spin_integrals = []
    for iFile, inFile in enumerate(inFiles):
        if not os.path.isfile(inFile):
            raise ValueError(f'File {inFile} does not exist')

        # if output exists mv to .bak
        if os.path.isfile(outFiles[iFile]):
            if verbose:
                print(f'ATTENTION: {outFiles[iFile]} exists, moving to *.bak')
            os.rename(outFiles[iFile], outFiles[iFile]+'.bak')

        if verbose:
            print(f"Reading {inFile}")
        full_elfcar = Elfcar.from_file(inFile)
        spinpol = 'diff' in full_elfcar.data
        #pymatgen: "total" key refers to Spin.up, and "diff" refers to Spin.down.
        if return_spin_integrals and not spinpol:
            raise ValueError(f"File {inFile} is not spinpolarized!")
        shape = full_elfcar.data['total'].shape
        n_data = np.prod(shape)

        if spinpol:
            full_data = full_elfcar.data['total'] + full_elfcar.data['diff']
        else:
            full_data = full_elfcar.data['total']

        if return_integrals:
            integrals.append(np.sum(np.abs(full_data)))
        if return_spin_integrals:
            spin_integrals.append((np.sum(np.abs(full_elfcar.data['total'])), np.sum(np.abs(full_elfcar.data['diff']))))
        if verbose:
            print(f"Shape of data: {shape}")
            print(f"Total number of datapoints: {n_data}")
            if return_integrals:
                integral = integrals[-1]
            else:
                integral = np.sum(np.abs(full_data))
            print(f"Integral of total data is {integral}")
            if spinpol:
                if return_spin_integrals:
                    spin_integral = spin_integrals[-1]
                else:
                    spin_integral = (np.sum(np.abs(full_elfcar.data['total'])), np.sum(np.abs(full_elfcar.data['diff'])))
                print("Integral of spin data is up: {}, down: {}".format(*spin_integral))

        origin = np.zeros(3)
        atoms = AseAtomsAdaptor.get_atoms(full_elfcar.structure)

        # write cubes
        if spinpol:
            filename = f"{outFiles[iFile]}_up.cube"
            if verbose:
                print(f"Writing {filename}")
            with open(filename, 'w') as f:
                write_cube(f, atoms, data=full_elfcar.data['total'], origin=origin)
            filename = f"{outFiles[iFile]}_down.cube"
            if verbose:
                print(f"Writing {filename}")
            with open(filename, 'w') as f:
                write_cube(f, atoms, data=full_elfcar.data['diff'], origin=origin)
            filename = f"{outFiles[iFile]}_diff.cube"
            if verbose:
                print(f"Writing {filename}")
            with open(filename, 'w') as f:
                write_cube(f, atoms, data=full_elfcar.data['total']-full_elfcar.data['diff'], origin=origin)
        else:
            filename = f"{outFiles[iFile]}.cube"
            if verbose:
                print(f"Writing {filename}")
            with open(filename, 'w') as f:
                write_cube(f, atoms, data=full_data, origin=origin)

    if return_integrals:
        if len(integrals) == 1:
            if return_spin_integrals:
                return integrals[0], spin_integrals[0]
            else:
                return integrals[0]
        else:
            if return_spin_integrals:
                return integrals, spin_integrals
            else:
                return integrals
    else:
        return


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert one or many ELFCAR files to cube format.')
    parser.add_argument('input', type=str, nargs='+', help='Input Files')
    parser.add_argument('--output', type=str, nargs='+', help='Output file names (no extension, .cube will be appended)')
    parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
    args = parser.parse_args()
    elf2cube(args.input, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
