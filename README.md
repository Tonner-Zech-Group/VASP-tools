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

## Quick start

All tools are available as command-line scripts after installation. Every tool supports `--help`:

```bash
vaspcheck --help
plotNEB --help
```

## Typical workflows

<details>
<summary>NEB calculation post-processing</summary>

```bash
# After your NEB converged, collect images into a movie and plot the barrier
neb2movie --output neb.xyz            # creates movie from NEB image dirs
plotNEB --unit eV --file barrier.png  # plot spline.dat + neb.dat
```

</details>

<details>
<summary>Frequency / vibrational analysis</summary>

```bash
# Extract imaginary mode and prepare for IRC
freq2mode                        # writes MODECAR and MODECAR.MW
add-MODECAR                      # preview displaced POSCAR as xyz

# Visualise all modes
freq2jmol                        # writes vib-NNN.xyz (JMol format)
viewMode --modecar MODECAR       # animated ASE GUI preview

# Split large frequency calculation across multiple jobs
split_vasp_freq split 20         # create freq_001/, freq_002/, …
# (run VASP in each subfolder, then)
split_vasp_freq combine          # merge results into freq/ cache
split_vasp_freq write_jmol       # export JMol xyz
```

</details>

<details>
<summary>Charge density / ELF</summary>

```bash
chgcar2cube CHGCAR --output chg     # → chg.cube
elf2cube ELFCAR --output elf        # → elf.cube (or elf_up/down for spinpol)
```

</details>

<details>
<summary>Geometry optimisation analysis</summary>

```bash
vaspcheck ./run          # check occupations and convergence
vaspGetEF ./run          # plot energy+forces across restart jobs
vaspGetEF ./run --poscar ./run/POSCAR  # selective-dynamics aware (forces over free DOF only)
vasp2traj traj.xyz OUTCAR  # convert OUTCAR to trajectory
```

</details>

<details>
<summary>IRC calculation plotting</summary>

```bash
plotIRC --reactant_dir irc_r/ --product_dir irc_p/ \
        --transition_state ts/ --file irc.svg
```

</details>

## Pre-processing tools

| Tool | Description | Example |
|------|-------------|---------|
| `add-MODECAR` | Add MODECAR displacements to a POSCAR, output xyz animation. | `add-MODECAR --poscar POSCAR --modecar MODECAR` |
| `freq2mode` | Generate MODECAR and mass-weighted MODECAR from a frequency calculation. | `freq2mode -i 0` |
| `kgrid2kspacing` | Get KSPACING equivalent for the current POSCAR+KPOINTS. | `kgrid2kspacing` |
| `kspacing2kgrid` | Get k-point grid for a given KSPACING and the current POSCAR. | `kspacing2kgrid 0.15` |
| `mixed_interpolate` | Geodesic interpolation for the molecule + IDPP for the surface. | `mixed_interpolate` |
| `poscar2nbands` | Compute the recommended NBANDS for LOBSTER from POSCAR/INCAR/POTCAR. | `poscar2nbands` |

## Post-processing tools

| Tool | Description | Example |
|------|-------------|---------|
| `calc-deformation-density` | Calculate deformation density from AB, A and B VASP run folders. | `calc-deformation-density` |
| `chgcar2cube` | Convert CHGCAR-like files to cube files (converts units to e⁻/Å³). | `chgcar2cube CHGCAR --output chg` |
| `elf2cube` | Convert ELFCAR files to cube files. | `elf2cube ELFCAR --output elf` |
| `freq2jmol` | Write JMol-compatible xyz files for all vibrational modes. | `freq2jmol --directory ./` |
| `neb2movie` | Convert VASP NEB images to an ASE ext-xyz movie (like `nebmovie.pl`). | `neb2movie --output neb.xyz` |
| `plotIRC` | Plot VASP IRC calculations in both directions, shift-compatible. | `plotIRC -r irc_r/ -p irc_p/ -t ts/` |
| `plotNEB` | Plot VASP+VTST NEB results (reads `spline.dat` + `neb.dat`). | `plotNEB --unit eV --file neb.png` |
| `plot_neb_movie` | Create presentation images for NEB using VMD and plotNEB. | `plot_neb_movie` |
| `replace_potcar_symlinks` | Replace POTCARs in subdirectories with symlinks. **Use with care.** | `replace_potcar_symlinks` |
| `split_vasp_freq` | Split a VASP frequency calculation into partial jobs and recombine. | `split_vasp_freq split 20` |
| `vasp2traj` | Convert VASP OUTCAR or XDATCAR to an ext-xyz trajectory. | `vasp2traj traj.xyz OUTCAR` |
| `vaspcheck` | Assert proper occupations and SCF+GO convergence using ASE. | `vaspcheck ./run` |
| `vaspGetEF` | Plot energy and max force across multiple GO restart jobs. When a POSCAR with selective dynamics is found, forces are computed over free (unfrozen) components only — per-component constraints (e.g. `T T F`) are handled correctly. | `vaspGetEF ./run` |
| `viewMode` | Animated preview of a MODECAR in the ASE GUI. | `viewMode --scale 2` |
| `visualize-magnetization` | Create a VMD visualisation state for the magnetisation density. | `visualize-magnetization` |

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

## Citing this work

If you use tools4vasp in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff) or via the **Cite this repository** button on
GitHub. The DOI is [10.5281/zenodo.15525217](https://doi.org/10.5281/zenodo.15525217).

## License

MIT — see [LICENSE](LICENSE).
