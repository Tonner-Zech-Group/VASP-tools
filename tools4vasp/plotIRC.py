#!/usr/bin/env python3
import argparse
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import ase
import ase.io
from ase.units import _Nav as Nav, _e as e

ev_to_kjMol = (Nav * e)/1000


def calc_rms(pos1, pos2):
    step = pos1 - pos2
    rms = np.sqrt(np.mean(np.square(step)))
    return rms


def read_sp(directory, allow_freq):
    path = os.path.join(directory, "vasprun.xml")
    if os.path.isfile(path):
        file = ase.io.read(path, index=":")
        # Check if the file is a single point calculation:
        if not (len(file) == 1) and not allow_freq:
            print("Supplied transition state is not a single point calculation")
            exit(1)
        energy = file[0].calc.results["energy"]
        energy *= ev_to_kjMol
        positions = file[0].positions
    else:
        print("vasprun.xml file not found")
        exit(1)
    return positions, energy


def read_irc(directory):
    path = os.path.join(directory, "OUTCAR")
    structures = ase.io.read(path, index=":")
    # read first structure
    init_structure = structures[0].positions
    # read step_size:
    with open(path, "r") as f:
        outcar = f.readlines()
    steps = []
    pattern = re.compile(r'IRC \(A\):\s+([-+]?\d*\.\d+([eE][-+]?\d+)?)\s+')
    for line in outcar:
        if "IRC (A):" in line:
            match = pattern.search(line)
            result = match.group(1)
            steps.append(float(result))
    step_size = []
    for i in range(1, len(steps) - 1):
        step_size.append(steps[i - 1] - steps[i])
    energies = []
    for structure in structures:
        energy = structure.calc.results["energy"]
        energy *= ev_to_kjMol
        energies.append(energy)
    return init_structure, steps, energies


def generate_plot(irc_e, irc_p, ts):
    # check if a transition state was supplied. otherwise use the first image of the product irc as the image
    ts_exists = True
    if ts is None:
        ts_geo = irc_p[0]
        ts_energy = irc_p[2][0]
        ts = [ts_geo, ts_energy]
        ts_exists = False
        print("No transition state supplied, assuming initial image of product path as transition state")
    # check if the positions of the tree structures are compatible
    if not (ts[0].shape == irc_p[0].shape and ts[0].shape == irc_e[0].shape):
        print("Geometries of the calculations are not matching. Please ensure that the atoms are numbered correctly.")
        exit(1)
    # calculate the offset of the directions from the transition state
    reactant_offset = calc_rms(ts[0], irc_e[0])
    product_offset = calc_rms(ts[0], irc_p[0])
    # apply offset to reaction coordinate and ensure the correct direction of the coordinate
    # Product Path
    print(f"Found Reactant ICR calculation to be offset by {reactant_offset} Angstrom")
    print(f"Found Product ICR calculation to be offset by {product_offset} Angstrom")
    if irc_p[1][0] > irc_p[1][-1]:  # Reaction coordinate in negative direction
        for i in range(len(irc_p[1])):
            irc_p[1][i] = (irc_p[1][i] * (-1)) + product_offset
    else:
        for i in range(len(irc_p[1])):
            irc_p[1][i] = irc_p[1][i] + product_offset
    # Reactant path:
    if irc_e[1][0] > irc_e[1][-1]:  # Reaction coordinate in negative direction
        for i in range(len(irc_e[1])):
            irc_e[1][i] = irc_e[1][i] - reactant_offset
    else:
        for i in range(len(irc_e[1])):
            irc_e[1][i] = (irc_e[1][i] * (-1)) - reactant_offset
    # Generate basic plot data
    # Offset energy
    if args.offset is None:
        offset = 0 - ts[1]
    else:
        offset = args.offset - ts[1]
    for i in range(len(irc_e[2])):
        irc_e[2][i] = irc_e[2][i] + offset
    for i in range(len(irc_p[2])):
        irc_p[2][i] = irc_p[2][i] + offset
    ts = (ts[0], ts[1] + offset)
    # Generate additional plot data
    if ts_exists:
        s_plot_x = [irc_e[1][0], 0, irc_p[1][0]]
        s_plot_y = [irc_e[2][0], ts[1], irc_p[2][0]]
    else:
        s_plot_x = [irc_e[1][0], irc_p[1][0]]
        s_plot_y = [irc_e[2][0], irc_p[2][0]]
    return irc_e[1], irc_e[2], s_plot_x, s_plot_y, irc_p[1], irc_p[2]


def plot_irc(irc_data, silent=False):
    mpl.use("pgf")
    plt.rc('text', usetex=True)
    plt.rc('font', size=12)
    plt.rcParams["font.family"] = "Arial"
    _ = plt.figure(constrained_layout=True)
    plt.plot(irc_data[0], irc_data[1], label='IRC_e', color='#B02F2C')
    plt.plot(irc_data[4], irc_data[5], label='IRC_p', color='#B02F2C')
    plt.plot(irc_data[2], irc_data[3], linestyle='dashed', color='#8AC2D1', marker='o',
             label='TransitionState')
    plt.xlabel(r'IRC coordinate $[\r{A}]$')
    plt.ylabel(r'$\Delta E\,[\mathrm{kJ\,mol^{-1}}]$')
    plt.savefig("plot.svg")
    if not silent:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='plotIRC',
        description='Tool that creates a plot of VASP IRC calculations in both direction and is compatible with shifts in the starting structure.',
        epilog='Tonner-Zech Research Group -- https://github.com/Tonner-Zech-Group')
    parser.add_argument("-r", "--reactant_dir", type=str, required=True,
                        help="Directory of the ICR calculation towards the reactant structure.")
    parser.add_argument("-p", "--product_dir", type=str, required=True,
                        help="Directory of the IRC calculation towards the product structure.")
    parser.add_argument("-t", "--transition_state", type=str, required=False, default="none",
                        help="Directory of a single point calculation of the transition state")
    parser.add_argument("-f", "--allow_freq", action='store_true', required=False, default=False,
                        help="Allow for the transition state calculation to be replaced by a frequency calculation")
    parser.add_argument("-s", "--silent", action='store_true',
                        help="Disable interactive plotting, do not open plot")
    parser.add_argument("-o", "--offset", type=float, default=None, required=False,
                        help="Offset the energy value in the graph by providing the energy of the transition state in \
                        kJ/mol. If not set, the transition state will be labled as 0")
    args = parser.parse_args()
    irc_e = read_irc(args.reactant_dir)
    irc_p = read_irc(args.product_dir)
    if args.transition_state == "none":
        plot_data = generate_plot(irc_e, irc_p, None)
    else:
        ts = read_sp(args.transition_state, args.allow_freq)
        plot_data = generate_plot(irc_e, irc_p, ts)
    plot_irc(plot_data, args.silent)
