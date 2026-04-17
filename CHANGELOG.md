# Changelog

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
