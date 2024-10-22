# VASP-tools

Our collection of tools for pre- and post-processing VASP calculations. Mainly Python and Bash.

## Installation

Clone this repository and run `pip install .` inside the main directory. If you want to always use the latest content of the repo you can use the 'developement' install of pip by running `pip install -e .`. Just doing `git pull` to get the latest content of the repo will then automatically result in the usage of the latest code without need to reinstall.

You can also use the latest release by installing it from PyPi:

```bash
pip install tools4vasp
```

## Dependencies

Different for each script, but mainly

- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [VTST](http://theory.cm.utexas.edu/vtsttools/)
- [Pymatgen](https://pymatgen.org/)
- [Geodesic Interpolation](https://github.com/virtualzx-nad/geodesic-interpolate)

## Usage

All scripts can be accessed directly from the shell after installation with pip.

## Pre-Processing

- add_MODECAR: Add the displacements from a MODECAR file to the positions in a POSCAR file.
- freq2mode: generates MODECAR and mass-weighted MODCAR files from frequency calculations.

## Post-Processing

- calc_deformation_density: Calculate the deformation density from three VASP run folders AB, A and B.
- chgcar2cube: Convert CHGCAR-like files to cube files using Pymatgen and ASE.
- elf2cube: Script to convert ELFCAR files to cube files.
- freq2jmol: Write JMol compatible xyz for visualization of vibrational modes.
- freq2mode: A VASP tool, which generates MODECAR and mass-weighted MODCAR files from frequency calculations.
- kgrid2kspacing: Script to get a KSPACING from a KPOINTS file and a POSCAR.
- kspacing2kgrid: Script to get a KGrid from a KSPACING value and a POSCAR. 
- mixed_interpolate: Uses geodesic interpolation for the molecule and idpp interpolation for the surface of a molecule.
- neb2movie: Convert VASP NEB to ASE ext-xyz movie, just like nebmovie.pl of VTST.
- plot_neb_movie: Use VMD and plotNEB to create images for NEB curve presentation.
- plotIRC: Tool that creates a plot of VASP IRC calculations in both direction and is compatible with shifts in the starting structure.
- plotNEB: Script to plot VASP+TST NEB calculation results.
- poscar2nbands: Helper to get the NBANDS value for LOBSTER calculations using the current POSCAR, INCAR and POTCAR setup with 'standard' options.
- replace_potcar_symlinks: Searches for POTCARS in subdirs and replaces them with symlinks. CAREFUL!
- split_vasp_freq: A script to split a VASP frequency calculation into individual parts and recombine the results.
- vasp2traj: Convert VASP geometry optimization output to ASE compatible ext-xyz trajectory file.
- vaspcheck: Assert proper occupations and SCF+GO convergence in VASP using ASE.
- vaspGetEF: Creates a plot of energy and forces along multiple GO runs (e.g. for restart jobs). Gathers data in all numeric subfolders and this folder containing a vasprun.xml file (depth one) and combines them in a single plot.
- viewMode: Shows a graphical preview of a MODECAR file using ase gui
- visualize_magnetization: Creates a VMD visualisation state file for the magnetization denisty by splitting the CHGCAR (by running chgsplit.pl), converting it to a cube file (by running chgcar2cube.sh) and then creating representations for VMD.
