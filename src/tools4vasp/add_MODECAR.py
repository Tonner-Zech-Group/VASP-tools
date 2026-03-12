#!/usr/bin/env python3
import argparse
from ase import io
import numpy as np


def main():
    """Add the displacements from a MODECAR file to the positions in a POSCAR file."""
    parser = argparse.ArgumentParser(
        description="Add MODECAR displacement vectors to POSCAR positions and write a "
                    "two-frame xyz animation (original + displaced structure).",
        epilog="Example: add-MODECAR --poscar POSCAR --modecar MODECAR")
    parser.add_argument("--poscar", type=str, default="POSCAR",
                        help="Input POSCAR file (default: POSCAR)")
    parser.add_argument("--modecar", type=str, default="MODECAR",
                        help="Input MODECAR displacement file (default: MODECAR)")
    parser.add_argument("--output", type=str, default="poscar+modecar.xyz",
                        help="Output xyz file (default: poscar+modecar.xyz)")
    args = parser.parse_args()

    poscar = io.read(args.poscar)
    add = np.loadtxt(args.modecar)
    poscar.write(args.output)
    poscar.positions += add
    poscar.write(args.output, append=True)


if __name__ == "__main__":
    main()
