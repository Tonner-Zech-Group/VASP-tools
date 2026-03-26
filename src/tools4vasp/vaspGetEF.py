#!/usr/bin/env python3
"""
Combine VASP xml files of all numerical subdirectories and the parent directory
and plot the forces and energies.  Useful for comparing the convergence of
forces and energies in VASP calculations.

Selective-dynamics-aware force masking (read_free_mask, free_mask parameter of
get_max_f) is adapted from the per-component T/F handling in utils4VASP
check_geoopt.py (MIT License, © 2025 J. Steffen, A. Mölkner, M. A. Bechtel,
https://github.com/Trebonius91/utils4VASP), which itself was based on grad2.py
(Apache License 2.0, © 2008–2012 Peter Larsson).  The present implementation
reads POSCAR directly and applies the mask to forces from vasprun.xml rather
than OUTCAR, but the per-component masking idea is the same.
"""

from natsort import natsorted
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.io.vasp import read_vasp_xml
from typing import Tuple, List, Dict, Optional


print_format = "{:5d} {:20.8f} {:20.6f} {:20.6g}"


def read_free_mask(poscar_path: str) -> Optional[np.ndarray]:
    """Read a per-component free/frozen boolean mask from a POSCAR.

    Parses the selective-dynamics T/F flags and returns a mask where
    ``True`` means the component is free (unfrozen).  Per-component
    constraints (e.g. ``T T F``) are handled correctly.

    Adapted from the selective-dynamics handling in utils4VASP
    check_geoopt.py (MIT License, © 2025 J. Steffen, A. Mölkner,
    M. A. Bechtel) which was based on grad2.py (Apache License 2.0,
    © 2008–2012 Peter Larsson).

    Parameters
    ----------
    poscar_path : str
        Path to the POSCAR file.

    Returns
    -------
    np.ndarray of shape (natoms, 3), dtype bool, or None.
    Returns None if the file does not exist or has no selective dynamics.
    """
    if not os.path.isfile(poscar_path):
        return None
    with open(poscar_path, 'r') as f:
        lines = f.readlines()

    # 'Selective dynamics' appears between the element-counts line and the
    # Direct/Cartesian line.  Search lines 5–8 (covers both VASP4 and VASP5
    # formats where element-name line may or may not be present).
    sel_idx = None
    for i in range(5, min(9, len(lines))):
        if lines[i].strip().lower().startswith('s'):
            sel_idx = i
            break
    if sel_idx is None:
        return None  # no selective dynamics

    # Element counts are on the line immediately before 'Selective dynamics'
    try:
        natoms = sum(int(x) for x in lines[sel_idx - 1].split())
    except ValueError:
        return None

    # Coordinate lines start two lines after 'Selective dynamics'
    # (one line for the 'Direct'/'Cartesian' header)
    coord_start = sel_idx + 2
    mask = np.ones((natoms, 3), dtype=bool)
    has_frozen = False
    for i in range(natoms):
        parts = lines[coord_start + i].split()
        if len(parts) >= 6:
            mask[i, 0] = parts[3].upper() != 'F'
            mask[i, 1] = parts[4].upper() != 'F'
            mask[i, 2] = parts[5].upper() != 'F'
            if not all(mask[i]):
                has_frozen = True

    if not has_frozen:
        return None  # selective dynamics block present but nothing is frozen
    return mask


def get_max_f(atoms, free_mask: Optional[np.ndarray] = None) -> float:
    """Get the maximum force magnitude from an ASE atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with an attached calculator.
    free_mask : np.ndarray of shape (natoms, 3), dtype bool, optional
        Per-component mask where True = free (unfrozen).  When provided,
        frozen components are zeroed before computing the maximum, so only
        active degrees of freedom contribute.  This correctly handles
        selective dynamics including partial per-component constraints
        (e.g. T T F).  When None the existing behaviour is preserved
        (ASE apply_constraint=True).
    """
    if free_mask is not None:
        forces = atoms.get_forces(apply_constraint=False)
        forces = forces * free_mask
    else:
        forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())


def read_xml(path, verbose=False, write_fe=False,
             free_mask: Optional[np.ndarray] = None) -> Tuple[List[float], List[float]]:
    """Read a vasprun.xml and return max forces and energies per ionic step.

    Parameters
    ----------
    path : str
        Path to the vasprun.xml file.
    verbose : bool
        Print per-step values.
    write_fe : bool
        Write values to fe.dat in the same directory.
    free_mask : np.ndarray or None
        Per-component free/frozen mask; passed to get_max_f.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("No such file: {}".format(path))

    print(path)
    traj = read_vasp_xml(path, index=slice(0, None))

    if write_fe:
        fe_path = os.path.join(os.path.dirname(os.path.abspath(path)), 'fe.dat')
        fe = open(fe_path, 'w')

    forces = []
    energies = []
    for i, atoms in enumerate(traj):
        energies.append(atoms.get_potential_energy())
        forces.append(get_max_f(atoms, free_mask=free_mask))
        if verbose:
            print(print_format.format(i, forces[-1], energies[-1], energies[-1] - energies[0]))
        if write_fe:
            fe.write(print_format.format(i, forces[-1], energies[-1], energies[-1] - energies[0]) + '\n')

    if write_fe:
        fe.close()
    return forces, energies


def get_all_xmls(path, verbose=False) -> List[str]:
    """Get all vasprun.xml files in path and its numeric subdirectories."""
    files = natsorted(glob.glob(os.path.join(path, '*/vasprun.xml'), recursive=True))
    if os.path.isfile(os.path.join(path, 'vasprun.xml')):
        files.append(os.path.join(path, 'vasprun.xml'))
    return files


def process_all_xmls(path, verbose=False, write_json=False,
                     poscar_path: Optional[str] = None) -> Dict[str, List[float]]:
    """Process all vasprun.xml files in path and numeric subdirectories.

    Parameters
    ----------
    path : str
        Path to the VASP calculation directory.
    verbose : bool
        Print per-step values.
    write_json : bool
        Write combined forces and energies to fe-combined.json.
    poscar_path : str or None
        Path to the POSCAR file used to read selective-dynamics constraints.
        Defaults to ``<path>/POSCAR``.  Pass an empty string to disable.
    """
    files = get_all_xmls(path, verbose)

    # Resolve POSCAR path and build free mask (once, shared across all XMLs)
    if poscar_path is None:
        poscar_path = os.path.join(path, 'POSCAR')
    free_mask = read_free_mask(poscar_path) if poscar_path else None
    if free_mask is not None:
        n_frozen_atoms = int(np.any(~free_mask, axis=1).sum())
        n_frozen_dof = int((~free_mask).sum())
        print(
            f"Selective dynamics detected in {poscar_path}: "
            f"{n_frozen_atoms} frozen atom(s), {n_frozen_dof} frozen DOF — "
            f"max force computed over free components only."
        )

    data = []
    for f in files:
        folder = os.path.dirname(os.path.abspath(f))
        if (not os.path.basename(folder).isdigit()) and \
                (not os.path.realpath(folder) == os.path.realpath(path)):
            if verbose:
                print("Not using {}".format(folder))
            continue
        fe_dat = os.path.join(folder, 'fe.dat')
        # Skip cache when a selective-dynamics mask is active to avoid stale
        # values from a previous run without the mask.
        if free_mask is None and os.path.isfile(fe_dat):
            if verbose:
                print("Adding {}".format(folder))
        else:
            print("Generating fe.dat in {}".format(folder))
            read_xml(f, verbose=verbose, write_fe=True, free_mask=free_mask)
            assert os.path.isfile(fe_dat), \
                "Problem generating fe.dat in {:}".format(folder)
        to_add = np.loadtxt(fe_dat)
        # handle single-entry files
        if to_add.shape == (4,):
            to_add = to_add.reshape(1, 4)
        assert to_add.shape[1] == 4, \
            "Problem with the shape of the data in {:}".format(folder)
        data.append(to_add)
        if verbose:
            print("Found {} values".format(len(data[-1])))

    combined = {'force': [], 'energy': []}
    for d in data:
        if d.shape == (4,):
            combined['force'].append(d[1])
            combined['energy'].append(d[2])
        else:
            combined['force'].extend(d[:, 1].tolist())
            combined['energy'].extend(d[:, 2].tolist())
    if write_json:
        with open(os.path.join(path, 'fe-combined.json'), 'w') as f:
            json.dump(combined, f)
    return combined


def plot_fe(combined, filename, lw=2, show=False) -> None:
    """Plot the forces and energies."""
    nItems = len(combined['force'])
    xAxis = list(range(1, nItems + 1))
    fig, ax1 = plt.subplots()
    plt.xlabel('Step #')
    color = 'black'
    ax1.set_ylabel(r'$\Delta E$ [eV]', color=color)
    ax1.plot(xAxis, combined['energy'], color=color, ls='-', lw=lw)
    color = 'grey'
    ax2 = ax1.twinx()
    ax2.grid(None)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'max($F$) [eV Å$^{-1}$]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(xAxis, combined['force'], color=color, ls='-', lw=lw)
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Plot energy and max force across multiple VASP geometry optimisation "
            "runs.  Collects vasprun.xml from numeric subdirectories and the "
            "current directory.  When a POSCAR with selective dynamics is found, "
            "forces are computed over free (unfrozen) components only."
        ),
        epilog=(
            "Examples:\n"
            "  vaspGetEF /path/to/calc\n"
            "  vaspGetEF /path/to/calc --poscar /path/to/calc/POSCAR\n"
            "  vaspGetEF /path/to/calc --poscar ''   # disable SD masking"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default=None,
                        help="Path to VASP calculation directory (default: cwd)")
    parser.add_argument("--poscar", default=None, metavar="FILE",
                        help=(
                            "POSCAR file for selective-dynamics masking "
                            "(default: <path>/POSCAR; pass '' to disable)"
                        ))
    args = parser.parse_args()
    path = args.path if args.path is not None else os.getcwd()
    # Resolve poscar_path: None → auto-detect; '' → disable
    poscar_path = args.poscar  # stays None (auto) or user-supplied string
    combined = process_all_xmls(path, verbose=True, write_json=True,
                                poscar_path=poscar_path)
    plot_fe(combined, "fe-combined.png")
    print("...Done!")


if __name__ == '__main__':
    main()
