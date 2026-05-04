#!/usr/bin/env python3
# Script to set vacuum in VASP POSCAR files using ASE

from ase.io import read, write
import numpy as np
import os.path
import argparse
import glob


def set_vacuum(atoms_obj, vac, bottom_space=1, direction=2):
    """Set vacuum of an ase Atoms object."""
    translation_vector = [0, 0, 0]
    translation_vector[direction] -= np.min(
        atoms_obj.positions[:, direction] - bottom_space
    )
    atoms_obj.translate(translation_vector)
    new_cell_vector = [0, 0, 0]
    new_cell_vector[direction] += np.max(atoms_obj.positions[:, direction]) + (
        vac - bottom_space
    )
    atoms_obj.cell[direction] = new_cell_vector
    return atoms_obj


def set_potcar_vacuum(
    poscar_path, vac, bottom_space=1, direction=2, overwrite=True, verbose=False
):
    """Set vacuum of VASP POSCAR FILES"""
    structure = read(poscar_path, format="vasp")
    if overwrite:
        write(poscar_path + "_old", structure, format="vasp")
    new_structure = set_vacuum(
        structure, vac=vac, bottom_space=bottom_space, direction=direction
    )
    if overwrite:
        write(poscar_path, new_structure, format="vasp")
    else:
        write(poscar_path + f"_vac{vac}", new_structure, format="vasp")
    if verbose:
        print(
            f"Set vacuum to {vac} Angstroms in {poscar_path} (bottom space: {bottom_space} Angstroms, direction: {direction})"
        )


def main():
    parser = argparse.ArgumentParser(
        prog="VacuumSetter", description="Tool for adjusting the vacuum of POSCAR files"
    )
    parser.add_argument(
        "vacuum_size", type=float, help="total cell length in the vacuum direction"
    )
    parser.add_argument("-f", "--file", default="POSCAR", help="filename to process")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="search subdirectories"
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="replace the input file (default false)",
    )
    parser.add_argument(
        "-d",
        "--direction",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="cell axis (0,1,2) to modify",
    )
    parser.add_argument(
        "-b",
        "--bottom_space",
        type=float,
        default=1,
        help="vacuum below the lowest atom",
    )

    args = parser.parse_args()
    vac = float(args.vacuum_size)
    file = args.file
    recursive = args.recursive
    overwrite = args.overwrite
    direction = int(args.direction)
    bottom_space = float(args.bottom_space)

    current_dir = os.getcwd()
    if recursive:
        files = glob.glob(current_dir + "/**/" + file, recursive=True)
    else:
        files = glob.glob(current_dir + "/" + file)

    if not files:
        print(f"No files named '{file}' found in the current directory.")
        return

    for file in files:
        set_potcar_vacuum(
            poscar_path=file,
            vac=vac,
            bottom_space=bottom_space,
            direction=direction,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    main()
