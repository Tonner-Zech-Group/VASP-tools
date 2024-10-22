#!/usr/bin/env python3
import argparse
import ase.io
import ase
import numpy as np
from ase.visualize import view


def read_modecar(file):
    mode = []
    with open(file, "r") as f:
        modecar = f.readlines()
    for line in modecar:
        ex = line.split()
        mode.append([float(ex[0]), float(ex[1]), float(ex[2])])
    re_mode = np.array(mode, dtype=np.float64)
    return re_mode


def increment_positions(atoms, mode, factor):
    n_atoms = atoms.copy()
    n_atoms.positions = atoms.positions + mode * factor
    return n_atoms


def make_animation(atoms, mode, frames, scale):
    anim = []
    anim.append(atoms)  # First image should be original POSCAR
    frames = frames - 1
    while frames % 4:   # Ensure the number of frames is divisible by four
        frames += 1
    for frame in range(int(frames / 4)):
        atoms = increment_positions(atoms, mode, scale)
        anim.append(atoms)
    for frame in range(int(frames / 2)):
        atoms = increment_positions(atoms, mode, -scale)
        anim.append(atoms)
    for frame in range(int(frames / 4)):
        atoms = increment_positions(atoms, mode, scale)
        anim.append(atoms)
    # Last Frame should be the original POSCAR
    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='viewMode',
        description='shows a preview of a modecar file in ASE GUI',
        epilog='')
    # Add parser arguments
    parser.add_argument("-p", "--poscar", type=str, default='POSCAR',
                        help="path to poscar file that corresponds to the modecar file")
    parser.add_argument("-m", "--modecar", type=str, default='MODECAR', help="modecar file that should be previewed")
    parser.add_argument("-f", "--frames", type=int, default=60, help="number of frames in the animation")
    parser.add_argument("-s", "--scale", type=float, default=1,
                        help="factor by which the modecar will be scaled each frame")
    args = parser.parse_args()
    mode = read_modecar(args.modecar)
    atoms = ase.io.read(args.poscar)
    anim = make_animation(atoms, mode, args.frames, args.scale * 0.05)
    view(anim)
