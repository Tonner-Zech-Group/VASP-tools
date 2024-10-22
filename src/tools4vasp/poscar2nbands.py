#!/usr/bin/env python3
#
# Script to get NBANDS value for LOBSTER calculation
# by Patrick Melix
# 2022/03/10
#


def get_nbands():
    from pymatgen.core.structure import Structure
    from pymatgen.io.lobster import Lobsterin
    lobsterin = Lobsterin.standard_calculations_from_vasp_files("POSCAR", "INCAR", "POTCAR", option='standard')
    print(lobsterin._get_nbands(Structure.from_file("POSCAR")))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Get NBANDS for LOBSTER from current POSCAR, POTCAR and INCAR')
    _ = parser.parse_args()
    get_nbands()

if __name__ == "__main__":
    main()