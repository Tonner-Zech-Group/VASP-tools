#!/usr/bin/env python3
"""Set the vacuum size in a VASP POSCAR file using ASE."""

import argparse
import glob
import os
import os.path

import numpy as np
from ase.io import read, write


def set_vacuum(atoms_obj, vac, bottom_space=1.0, direction=2):
    """Translate atoms and resize the cell so the empty space along ``direction``
    equals ``vac``.

    After the call, the lowest atom along ``direction`` sits at ``bottom_space``,
    and the lattice vector along ``direction`` is grown so that
    ``cell_length - slab_thickness == vac`` (i.e. ``vac`` is the total vacuum:
    space below the lowest atom + space above the highest atom).

    Only cells whose lattice vector along ``direction`` is axis-aligned are
    supported; otherwise a ``ValueError`` is raised.
    """
    direction = int(direction)
    cell_row = np.asarray(atoms_obj.cell[direction], dtype=float)
    other_axes = [i for i in range(3) if i != direction]
    parallel = abs(cell_row[direction])
    tolerance = 1e-6 * max(parallel, 1.0)
    if np.any(np.abs(cell_row[other_axes]) > tolerance):
        raise ValueError(
            f"Lattice vector along direction {direction} is not axis-aligned "
            f"({cell_row.tolist()}); non-orthorhombic cells are not supported."
        )

    translation = [0.0, 0.0, 0.0]
    translation[direction] = bottom_space - float(
        np.min(atoms_obj.positions[:, direction])
    )
    atoms_obj.translate(translation)

    new_length = float(
        np.max(atoms_obj.positions[:, direction]) + (vac - bottom_space)
    )
    atoms_obj.cell[direction, direction] = new_length
    return atoms_obj


def run(
    poscar_path,
    vac,
    bottom_space=1.0,
    direction=2,
    overwrite=False,
    verbose=False,
):
    """Apply :func:`set_vacuum` to a POSCAR on disk.

    If ``overwrite`` is True, the input file is replaced and a backup is
    written to ``<poscar_path>_old`` (raising ``FileExistsError`` if such a
    backup already exists). Otherwise the result is written to
    ``<poscar_path>_vac<vac>`` and the input is left untouched.
    """
    structure = read(poscar_path, format="vasp")
    if overwrite:
        backup = poscar_path + "_old"
        if os.path.exists(backup):
            raise FileExistsError(
                f"Refusing to overwrite existing backup file: {backup}"
            )
        write(backup, structure, format="vasp")
        out_path = poscar_path
    else:
        out_path = f"{poscar_path}_vac{vac}"

    new_structure = set_vacuum(
        structure, vac=vac, bottom_space=bottom_space, direction=direction
    )
    write(out_path, new_structure, format="vasp")
    if verbose:
        print(
            f"Set vacuum to {vac} Å in {poscar_path} "
            f"(bottom_space={bottom_space} Å, direction={direction}) -> {out_path}"
        )
    return out_path


def main():
    """CLI entry point registered in pyproject.toml [project.scripts]."""
    parser = argparse.ArgumentParser(
        prog="set_vacuum",
        description="Set the vacuum size in a VASP POSCAR file.",
        epilog="Example: set_vacuum 15.0 -f POSCAR -o",
    )
    parser.add_argument(
        "vacuum_size",
        type=float,
        help=(
            "total vacuum size in Å (space below the lowest atom + space "
            "above the highest atom along DIRECTION)"
        ),
    )
    parser.add_argument(
        "-f", "--file", default="POSCAR", help="filename to process"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="search subdirectories"
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="replace the input file (a backup is written to <file>_old)",
    )
    parser.add_argument(
        "-d",
        "--direction",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="cell axis (0,1,2) to expand",
    )
    parser.add_argument(
        "-b",
        "--bottom_space",
        type=float,
        default=1.0,
        help="vacuum below the lowest atom in Å (default: 1.0)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print per-file progress"
    )
    args = parser.parse_args()

    current_dir = os.getcwd()
    if args.recursive:
        pattern = os.path.join(current_dir, "**", args.file)
        files = glob.glob(pattern, recursive=True)
    else:
        candidate = os.path.join(current_dir, args.file)
        files = [candidate] if os.path.exists(candidate) else []

    if not files:
        print(f"No files named '{args.file}' found.")
        return

    for path in files:
        run(
            poscar_path=path,
            vac=args.vacuum_size,
            bottom_space=args.bottom_space,
            direction=args.direction,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
