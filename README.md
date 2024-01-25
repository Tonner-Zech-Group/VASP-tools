## VASP-tools
Our collection of tools for pre- and post-processing VASP calculations. Mainly Python and Bash.

### Dependencies
Different for each script, but mainly
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [VTST](http://theory.cm.utexas.edu/vtsttools/)
- [Pymatgen](https://pymatgen.org/)

### Pre-Processing
- freq2mode: generates MODECAR and mass-weighted MODCAR files from frequency calculations

### Post-Processing
- chgcar2cube.py: Convert CHGCAR-like files to cube files using Pymatgen and ASE.
- neb2movie.py: Convert VASP NEB to ASE ext-xyz movie, just like nebmovie.pl of VTST.
- poscar2nbands.py: Helper to get the NBANDS value for LOBSTER calculations using the current POSCAR, INCAR and POTCAR setup with 'standard' options.
- vasp2traj.py: Convert VASP geometry optimization output to ASE compatible ext-xyz trajectory file.
- vasp-check.py: Assert proper occupations and SCF+GO convergence in VASP using ASE.
- vasp-combine-vef.py: Creates a plot of energy and forces along multiple GO runs (e.g. for restart jobs). Runs TST vef.py in all subfolders and this folder containing a vasprun.xml file (depth one) and combines them in a single plot. (Got a bad absolute path in there)
- visualize-magnetization.sh: Creates a VMD visualisation state file for the magnetization denisty by splitting the CHGCAR (by running chgsplit.pl), converting it to a cube file (by running chgcar2cube.sh) and then creating representations for VMD.
- viewMode.py: Shows a graphical preview of a MODECAR file using ase gui
