#!/usr/bin/env python3
#
# Script to convert CHGCAR files to cube files and convert to e-/Ang^3
# by Patrick Melix
# 2022/04/04
#
# You can import the module and then call .main() or use it as a script
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.cube import write_cube
import numpy as np
import os


def chgcar2cube(inFiles, outFiles, verbose=True, return_integrals=False, return_spin_integrals=False, mult_volume=False):
    assert len(inFiles) == len(outFiles), "Number of input and output files must be equal!"
    integrals = []
    spin_integrals = []
    for iFile,inFile in enumerate(inFiles):
        if not os.path.isfile(inFile):
            raise ValueError('File {:} does not exist'.format(inFile))

        #if output exists mv to .bak
        if os.path.isfile(outFiles[iFile]):
            if verbose:
                print('ATTENTION: {:} exists, moving to *.bak'.format(outFiles[iFile]))
            os.rename(outFiles[iFile], outFiles[iFile]+'.bak')

        if verbose:
            print("Reading {}".format(inFile))
        full_chgcar = Chgcar.from_file(inFile)
        spinpol = 'diff' in full_chgcar.data.keys()
        if return_spin_integrals and not spinpol:
            raise ValueError("File {} is not spinpolarized!".format(inFile))
        shape = full_chgcar.data['total'].shape
        n_data = np.prod(shape)
        
        if return_integrals:
            integrals.append(np.sum(np.abs(full_chgcar.data['total'])))
            integrals[-1] /= n_data
        if return_spin_integrals:
            spin_integrals.append(np.sum(np.abs(full_chgcar.data['diff'])))
            spin_integrals[-1] /= n_data
        if verbose:
            print("Shape of data: {}".format(shape))
            print("Total number of datapoints: {}".format(n_data))
            if return_integrals:
                integral = integrals[-1]
            else:
                integral = np.sum(np.abs(full_chgcar.data['total']))
                integral /= n_data
            print("Integral of total data is {}".format(integral))
            if spinpol:
                if return_spin_integrals:
                    spin_integral = spin_integrals[-1]
                else:
                    spin_integral = np.sum(np.abs(full_chgcar.data['diff']))
                    spin_integral /= n_data
                print("Integral of diff data is {}".format(spin_integral))

            origin = np.zeros(3)
            atoms = AseAtomsAdaptor.get_atoms(full_chgcar.structure)

            #Contrary to VASP Wiki, the CHGCAR is not rho*V, but rho*n_data.
            #So in order to have the integral over space = nelectrons, we need to divide by n_data.
            #Since this would result in super small numbers, we can transform to rho*V
            factor = n_data
            if mult_volume:
                factor /= atoms.get_volume()
            full_chgcar.data['total'] /= factor
            if spinpol:
                full_chgcar.data['diff'] /= factor
            #write cube
            filename = "{}.cube".format(outFiles[iFile])
            if verbose:
                print("Writing {}".format(filename))
            with open(filename, 'w') as f:
                write_cube(f, atoms, data=full_chgcar.data['total'], origin=origin)
            if spinpol:
                filename = "{}_mag.cube".format(outFiles[iFile])
                if verbose:
                    print("Writing {}".format(filename))
                with open(filename, 'w') as f:
                    write_cube(f, atoms, data=full_chgcar.data['diff'], origin=origin)
                
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
    parser = argparse.ArgumentParser(description='Convert one or many CHGCAR-like files to cube format.')
    parser.add_argument('input', type=str, nargs='+', help='Input Files')
    parser.add_argument('-output', type=str, nargs='+', help='Output File Names (no extension)')
    parser.add_argument('-v', help='Verbose', action='store_true')
    parser.add_argument('--integral', help='Print Integrals', action='store_true')
    parser.add_argument('--volume', help='Multiply the Density with the Cell Volume', action='store_true')
    args = parser.parse_args()
    chgcar2cube(args.input, args.output, verbose=args.v, return_integrals=args.integral, mult_volume=args.volume)



if __name__ == "__main__":
    main()