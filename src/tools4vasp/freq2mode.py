#!/usr/bin/env python3
import argparse
import math
from ase.data import atomic_masses
import ase.io
from ase.units import _amu as amu, _me as me


def get_atomic_mass(element_symbol):
    mass_in_amu = atomic_masses[element_symbol]
    # Convert mass to atomic units (1 amu = 1.66053904e-27 kg)
    mass_in_au = mass_in_amu * amu / me
    return mass_in_au


def getAtomsFromOutcar(outcar_file):
    atoms = ase.io.read(outcar_file)
    return atoms


def get_frequencies(outcar):
    freqLines = []
    for line in outcar:
        if " f/i=" in line:
            freqLines.append(line)
        if "Eigenvectors after division by SQRT(mass)" in line:
            break
    return freqLines


def generate_mw(frequency, atoms):
    weights = atoms.get_masses()
    freq_mw = []
    for i in range(len(frequency)):
        freq_line = frequency[i]
        factor = math.sqrt(weights[i])
        new_freq_line = []
        for freq in freq_line:
            if not freq == 0:
                new_freq_line.append(freq / factor)
            else:
                new_freq_line.append(0.0)
        freq_mw.append(new_freq_line)
    return freq_mw


def write_modecar(frequency, filename):
    filestring = ""
    for freq_line in frequency:
        line = ""
        for freq in freq_line:
            if freq >= 0:
                line += "    "
            else:
                line += "   "
            line += "{:.10E}".format(freq)
        filestring += line + "\n"
        with open(filename, "w") as file:
            file.write(filestring)


def read_frequency_from_outcar(outcar, freq_line, atoms):
    current_frequency = []
    start_index = outcar.index(freq_line) + 2
    end_index = start_index + 0 + len(atoms)
    for line in outcar[start_index:end_index]:
        vals = line.split()
        current_frequency.append([float(vals[3]), float(vals[4]), float(vals[5])])
    return current_frequency


def main():
    parser = argparse.ArgumentParser(
        prog="freq2mode",
        description="A VASP tool, which generates MODECAR and mass-weighted MODCAR files from frequency calculations",
        epilog=""
    )
    parser.add_argument("-o", "--outcar", type=str, help="Optional: specify OUTCAR file", required=False, default="OUTCAR")
    parser.add_argument("-i", "--index", type=int, help="Force non interactive mode by specifying the index of the imaginary frequency to use", required=False, default=-1)
    args = parser.parse_args()
    outcar_file = args.outcar
    with open(outcar_file, "r") as f:
        outcar = f.readlines()
    atoms = getAtomsFromOutcar(outcar_file)
    freqs = get_frequencies(outcar)
    if len(freqs) == 0:
        print("No imaginary frequencies found")
        exit(0)
    if len(freqs) < args.index + 1:
        print("The specified mode was not found in the OUTCAR file")
        exit(1)
    if len(freqs) > 1:
        print(f"Found {len(freqs)} imaginary frequencies. Select the frequency for MODECAR generation")
        freq_index = 0
        for freq in freqs:
            print(f"{freq_index}    {freq}")
            freq_index += 1
        if args.index == -1:
            freq_index = int(input("Frequency for MODECAR generation:"))
        else:
            print(f"Using Frequency {args.index}")
            freq_index = args.index
    else:
        print("Found one imaginary mode:")
        print(freqs[0])
        print("Using this mode for MODECAR generation")
        freq_index = 0
    frequency = read_frequency_from_outcar(outcar, freqs[freq_index], atoms)
    print("Writing MODECAR file")
    write_modecar(frequency, "MODECAR")
    print("Writing mass-weighted MODECAR file")
    frequency_mw = generate_mw(frequency, atoms)
    write_modecar(frequency_mw, "MODECAR.MW")

if __name__ == "__main__":
    main()