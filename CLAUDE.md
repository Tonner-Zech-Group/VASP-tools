# CLAUDE.md — Development Guide for tools4vasp

## Project overview

`tools4vasp` is a Python package (+ Bash scripts) for pre- and post-processing
[VASP](https://www.vasp.at/) DFT calculations. It is published on PyPI as
`tools4vasp` and on Zenodo. Source lives under `src/tools4vasp/`; Bash helper
scripts are in `src/tools4vasp/bash_scripts/`.

## Repository layout

```
src/tools4vasp/          # Python modules (one tool per file)
src/tools4vasp/bash_scripts/  # Bash wrappers (ShellCheck-linted in CI)
test/                    # pytest suite
  conftest.py            # shared fixtures (POSCAR, KPOINTS, NEB data, …)
  test_units.py          # unit tests
  test_regression.py     # regression / integration tests
  test_coverage.py       # additional coverage-oriented tests
pyproject.toml           # build config, entry-points, pytest settings
.github/workflows/       # CI/CD pipelines
```

## Development setup

```bash
pip install -e ".[dev]"   # or just: pip install -e .
pip install pytest pytest-cov ruff
```

Python ≥ 3.10 is recommended (CI tests 3.10 and 3.12).

## Running tests

```bash
pytest                                      # run all tests
pytest --cov=tools4vasp --cov-report=term-missing  # with coverage
```

Tests use `unittest.mock` heavily — no real VASP installation is required.
Modules that need external binaries (VMD, LaTeX/pgf) are fully mocked at the
`matplotlib` / `subprocess` level.

`mixed_interpolate.py` is excluded from coverage because it requires the
optional `geodesic-interpolate` package which is not installed in the test
environment.

## Linting

```bash
ruff check .          # must pass with zero errors before merging
```

CI enforces ruff with `continue-on-error: false`.
Key rules in effect: F401, F841, F811, E731, E741, E401.

Bash scripts are linted by ShellCheck (via `azohra/shell-linter@latest`) in CI.

## Module structure convention

Every tool follows a two-function pattern:

```python
def run(arg1, arg2, ...):
    """Importable logic — called directly in tests and by main()."""
    ...

def main():
    """CLI entry point registered in pyproject.toml [project.scripts]."""
    import argparse
    parser = argparse.ArgumentParser(
        description="...",
        epilog="Example: toolname --flag value")
    # add arguments ...
    args = parser.parse_args()
    run(args.arg1, args.arg2)
```

Key points:
- `main()` **must** contain the argparse — it is what pip installs as the CLI command.
  Putting argparse only in `if __name__ == "__main__":` means the installed command
  ignores all CLI arguments.
- `run()` holds the actual logic so it can be imported and unit-tested without
  touching `sys.argv`.
- Tools whose `main()` uses argparse should be tested with `patch("sys.argv", [...])`.

## Adding a new tool

1. Create `src/tools4vasp/<toolname>.py` following the `run()`/`main()` convention above.
2. Register it in `pyproject.toml` under `[project.scripts]`.
3. Add tests in `test/test_coverage.py` (or a dedicated file); mock external
   dependencies so tests run without VASP. Patch `sys.argv` when calling `main()`.
4. Bash scripts go in `src/tools4vasp/bash_scripts/` and must pass ShellCheck.

## Branching & CI workflow

- Feature work: branch off `main` as `feat/<name>`, open a PR targeting `main`.
- CI (`python-app.yml`) triggers on **pull_request → main** and on **push → main**.
  It does **not** trigger on feature-branch pushes to avoid duplicate runs.
- Release workflow (`release.yml`) triggers on `v*` tags and publishes to PyPI.
- TestPyPI publish (`pypi-publish.yml`) is manual (`workflow_dispatch` only).

## Key dependencies

| Package | Purpose |
|---------|---------|
| `ase` | Atoms/trajectory handling throughout |
| `pymatgen` | CHGCAR/ELFCAR parsing, LOBSTER nbands helper |
| `matplotlib` | All plotting tools |
| `natsort` | Natural-sort for run-folder discovery |
| `numpy` | Numerical operations |

Optional (not installed in CI):
- `geodesic-interpolate` — required by `mixed_interpolate`
- VMD — required by `plot_neb_movie` / `visualize_magnetization`
- LaTeX/pgf — required by `plotIRC` for publication-quality figures

## Environment notes (this dev container)

```bash
# SSH agent is available at /ssh-agent
SSH_AUTH_SOCK=/ssh-agent git push

# Add GitHub to known hosts if needed
ssh-keyscan github.com >> ~/.ssh/known_hosts
```
