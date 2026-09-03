# Changelog

## [1.4.0] - 2026-09-01

### Bug Fixes

- **Packaging** — the wheel and sdist now contain `bash_scripts/*.sh`. They previously shipped only `bash_scripts/__init__.py`, so every console script that shells out to a shell script (`getPOTCAR`, `replace_potcar_symlinks`, `calc-deformation-density`, `plot_neb_movie`) was broken for anyone who installed from PyPI rather than from a checkout. Verified by building both artifacts and installing the wheel: all five scripts present, mode 0755.
- **CI** — `python-app.yml` now runs on every pull request, not only those targeting `main`. A pull request stacked on another feature branch previously received no checks at all, which on this repository matters because merging to `main` auto-publishes a release.

### Enhancements

- **`vasplint --outcar`** — compare an INCAR against the parameters VASP reports having used. This is the check a pre-submission linter structurally cannot do: it sees an INCAR edited after linting, a value VASP overrode, and a tag VASP silently ignored because it was misspelled. It also makes the `KPAR` check exact, using the irreducible k-point count the OUTCAR states rather than the mesh product, which is only an upper bound. VASP's echo is not literal — values are truncated (`accura`), reformatted (`BMIX = 0.0001` prints as `0.00`), renamed (`ALGO = Fast` appears as `IALGO = 68`) or resolved (`ISTART = 1` drops to 0 without a WAVECAR, `GGA = --` means the POTCAR default) — so comparison normalises booleans, compares numbers to the echo's own printed precision including its exponent, and lists what VASP does not report rather than assuming it matched.
- **`vasplint`** — new checks: `ENCUT` against the largest `ENMAX` in the POTCAR (and against 1.3 x that when `ISIF >= 3`), `MAGMOM` length against the atom count when `ISPIN = 2`, `LDAU` against `LMAXMIX`, and a NEB run against the presence of its image directories.
- **Template resolution** — the run directory is searched first, then `--template` (repeatable), then `$VASPLINT_TEMPLATES`. A calculation carrying a copy of its own template therefore verifies anywhere.

### New Tools

- **`vasplint`** — Check a VASP input directory *before* it is submitted, the pre-run counterpart to `vaspcheck`. Validates POSCAR/POTCAR species-block alignment (symlinks resolved, both VASP 4 and VASP 5 POSCARs), pseudopotential provenance, selective-dynamics blocks in fixed-geometry runs, transition-state tags left in an ordinary INCAR, `IBRION`/`NSW`/`INTERACTIVE`/`ISIF` consistency per run type, the interactive-mode structure count against `NSW`, `ICHARG` against a missing CHGCAR, k-mesh against a template, `KPAR` against the mesh size, `NCORE * KPAR` against the requested rank count, required `#SBATCH` directives (site rules opt-in via `--site`), continuation runs whose source is still writing its CONTCAR, and that every symlink is relative and resolves. Any check whose input is missing reports itself as skipped instead of passing quietly. `--json` for scripting, `--strict` to fail on warnings.

### New Modules

- **`tools4vasp.vaspsetup`** — Importable machinery for building VASP input directories: order-preserving `write_poscar` (never merges repeated species blocks), POTCAR assembly from a pseudopotential library (delegating to `getPOTCAR.sh`, so the recommended-extension table is not duplicated) or from an existing reference POTCAR, relative POTCAR symlinks to one shared file per batch, INCAR rendering with declared overrides and a self-describing provenance comment, refusal of NEB/dimer templates for ordinary runs, interactive-mode stdin files, job-script patching with required-directive assertions, and `continuation_dir()` for restarts whose POSCAR is a relative symlink to the previous CONTCAR (refusing sources that are still running, warning on unconverged ones).


## [1.3.1] - 2026-06-11

### Bug Fixes

- **`vaspcheck`** — The electronic-entropy check no longer shells out to `tail | grep` with an unquoted path. Paths containing spaces or shell metacharacters now work, and the command-injection vector is removed; the OUTCAR is parsed in pure Python.
- **`vaspcheck`** — The electronic-entropy check now uses the energies of the *final* electronic step: the OUTCAR is read backwards from the end (instead of only the last 200 lines, which crashed for large systems) and the last `TOTEN` / `energy without entropy` block wins instead of the first match in the window. Matching is anchored on the literal OUTCAR line formats via regex, and even multi-GB OUTCARs only have their tail touched. (Issue #24)
- **`plotNEB`** — Dispersion energies (`Edisp`) are now read from the image OUTCARs in pure Python instead of via `grep | tail` with `shell=True`, removing the runtime dependency on `grep`/`tail` and the unquoted-path hazard.

## [1.3.0] - 2026-05-12

### New Tools

- **`set_vacuum`** — Set the total vacuum size in a POSCAR file (space below the lowest atom + space above the highest atom along a chosen lattice direction). Supports batch processing via `--recursive`, optional in-place overwrite with automatic `_old` backup, and configurable `--direction` / `--bottom_space`. (PR #22)

## [1.2.0] - 2026-04-17

### New Tools

- **`getPOTCAR`** — Generate POTCAR files with the same element ordering as POSCAR. Includes a Bash backend (`getPOTCAR.sh`) and a Python CLI wrapper registered as a console script. (PR #15)
- **`plotHOMA_withPBC`** — Calculate and plot HOMA (Harmonic Oscillator Model of Aromaticity) values from atomic coordinates, with periodic boundary condition support. (PR #15)
- **`xyz2POSCAR`** — Insert a molecule from an `.xyz` file into a POSCAR cell, with options for rotation to XY plane, centering, sorting, and constraining atoms. (PR #15)
- **`split_surf_and_mol`** — Automatically split an adsorbate-surface complex into separate surface and molecule structures based on layer detection. (PR #16)

### Enhancements

- **`mixed_interpolate`** — Major rework: automatic surface/molecule detection via `split_surf_and_mol`, PBC-shift handling with `--removepbc`, optional XY-plane alignment with `--alignXY`, constraint restoration on merged trajectories, and intermediate transition-state support. (PR #16)
- **`getPOTCAR.sh`** — Replaced hardcoded `POTDIR` path with `$VASP_PP_PATH` environment variable (matches ASE/pymatgen/VTST convention). Clear error message if unset. (PR #18)
- **`plot-neb-movie.sh`** — Auto-detect Tachyon renderer via `$TACHYON_PATH` env var, `command -v tachyon`, or hardcoded fallback instead of only the hardcoded VMD path. (PR #18)
- **`vaspGetEF`** — Selective-dynamics-aware force masking: forces on frozen atoms are now correctly zeroed when reporting convergence. (PR #13)
- All new Python tools follow the `run()`/`main()` convention for testability and are registered as console scripts in `pyproject.toml`. (PR #15)

### Bug Fixes

- Fixed division-by-zero in plane rotation when the surface normal is already aligned with +Z (`mixed_interpolate`, `xyz2POSCAR`).
- Fixed `leastsq` args not being passed as a tuple (`mixed_interpolate`, `xyz2POSCAR`).
- Fixed `np.arctan` division-by-zero replaced with `np.arctan2` (`plotHOMA_withPBC`).
- Fixed hardcoded `Atoms("C", ...)` in periodic copies now using original atom symbol (`plotHOMA_withPBC`).
- Fixed shebangs from `#!/usr/bin/python` to `#!/usr/bin/env python3` across new scripts.
- Various typo fixes.

### Dependencies

- Added `scipy >= 1.10.0` as a declared dependency (required by `mixed_interpolate`, `xyz2POSCAR`). (PR #17)

### CI/CD

- Switched PyPI and TestPyPI publishing to OIDC trusted publishing (no more API tokens in secrets). (PR #13)
- Added tests for all new tools (`test_coverage.py`). (PR #15, #16)
