# VASP-tools

[![Basic tests](https://github.com/Tonner-Zech-Group/VASP-tools/actions/workflows/python-app.yml/badge.svg)](https://github.com/Tonner-Zech-Group/VASP-tools/actions/workflows/python-app.yml)
[![PyPI version](https://img.shields.io/pypi/v/tools4vasp)](https://pypi.org/project/tools4vasp/)
[![Python versions](https://img.shields.io/pypi/pyversions/tools4vasp)](https://pypi.org/project/tools4vasp/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15525217.svg)](https://doi.org/10.5281/zenodo.15525217)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A collection of Python and Bash tools for pre- and post-processing [VASP](https://www.vasp.at/) DFT calculations, maintained by the [Tonner-Zech Group](https://github.com/Tonner-Zech-Group).

## Installation

Install the latest release from PyPI:

```bash
pip install tools4vasp
```

Or clone the repository for the latest development version:

```bash
git clone https://github.com/Tonner-Zech-Group/VASP-tools.git
cd VASP-tools
pip install .
```

Use an editable install to track repository changes without reinstalling:

```bash
pip install -e .
```

## Dependencies

| Package | Purpose |
|---------|---------|
| [ASE](https://wiki.fysik.dtu.dk/ase/) | Atoms and trajectory handling |
| [Pymatgen](https://pymatgen.org/) | CHGCAR/ELFCAR parsing, LOBSTER helper |
| [matplotlib](https://matplotlib.org/) | Plotting |
| [natsort](https://github.com/SethMMorris/natsort) | Natural-sort for folder discovery |
| [numpy](https://numpy.org/) | Numerical operations |

Optional dependencies (not installed automatically):

- [Geodesic Interpolation](https://github.com/virtualzx-nad/geodesic-interpolate) — required by `mixed_interpolate`
- [VTST tools](http://theory.cm.utexas.edu/vtsttools/) — used alongside NEB tools
- [VMD](https://www.ks.uiuc.edu/Research/vmd/) — required by `plot_neb_movie` and `visualize_magnetization`

## Usage

All tools are available as command-line scripts after installation:

```bash
vaspcheck          # check SCF/geometry convergence
vaspGetEF          # plot energy & forces across multiple GO restarts
plotNEB            # plot NEB barrier
plotIRC            # plot IRC pathway
chgcar2cube        # convert CHGCAR to cube file
elf2cube           # convert ELFCAR to cube file
neb2movie          # convert NEB images to ext-xyz movie
vasp2traj          # convert geometry optimisation to trajectory
freq2jmol          # write JMol-compatible xyz for vibrational modes
freq2mode          # generate MODECAR from frequency calculation
split_vasp_freq    # split/recombine VASP frequency calculation
poscar2nbands      # compute NBANDS for LOBSTER
kgrid2kspacing     # convert KPOINTS grid to KSPACING
kspacing2kgrid     # convert KSPACING to k-point grid
add-MODECAR        # add MODECAR displacements to POSCAR
```

## Pre-processing tools

| Tool | Description |
|------|-------------|
| `add-MODECAR` | Add displacements from a MODECAR file to a POSCAR. |
| `freq2mode` | Generate MODECAR and mass-weighted MODECAR from a frequency calculation. |
| `kgrid2kspacing` | Get a KSPACING value from a KPOINTS file and a POSCAR. |
| `kspacing2kgrid` | Get a k-point grid from a KSPACING value and a POSCAR. |
| `mixed_interpolate` | Geodesic interpolation for the molecule + IDPP for the surface. |
| `poscar2nbands` | Compute the recommended NBANDS for LOBSTER from POSCAR/INCAR/POTCAR. |

## Post-processing tools

| Tool | Description |
|------|-------------|
| `calc-deformation-density` | Calculate deformation density from AB, A and B VASP run folders. |
| `chgcar2cube` | Convert CHGCAR-like files to cube files using Pymatgen and ASE. |
| `elf2cube` | Convert ELFCAR files to cube files. |
| `freq2jmol` | Write JMol-compatible xyz files for vibrational mode visualisation. |
| `neb2movie` | Convert VASP NEB images to an ASE ext-xyz movie (like `nebmovie.pl`). |
| `plotIRC` | Plot VASP IRC calculations in both directions, shift-compatible. |
| `plotNEB` | Plot VASP+VTST NEB calculation results. |
| `plot_neb_movie` | Create images for NEB curve presentation using VMD and plotNEB. |
| `replace_potcar_symlinks` | Replace POTCARs in subdirectories with symlinks. **Use with care.** |
| `split_vasp_freq` | Split a VASP frequency calculation into parts and recombine results. |
| `vasp2traj` | Convert VASP geometry optimisation output to an ext-xyz trajectory. |
| `vaspcheck` | Assert proper occupations and SCF+GO convergence using ASE. |
| `vaspGetEF` | Plot energy and forces across multiple GO runs (handles restart jobs). |
| `viewMode` | Graphical preview of a MODECAR using `ase gui`. |
| `visualize-magnetization` | Create a VMD visualisation state for the magnetisation density. |

## Development

```bash
git clone https://github.com/Tonner-Zech-Group/VASP-tools.git
cd VASP-tools
pip install -e .
pip install pytest pytest-cov ruff
pytest
ruff check .
```

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines, CI setup, and
instructions for adding new tools.

## Contributing

1. Fork the repository and create a branch: `feat/<description>`.
2. Write tests for new functionality in the `test/` directory.
3. Ensure `ruff check .` and `pytest` both pass locally before opening a PR.
4. Open a pull request against `main`.

## License

MIT — see [LICENSE](LICENSE).
