#!/usr/bin/env python3
"""Check SCF/ionic convergence and POTCAR/POSCAR alignment from a VASP OUTCAR.

Unlike vaspcheck (which requires a full VASP directory and only inspects the
final step via ASE's Vasp calculator), this module reads the raw OUTCAR text
and reports convergence for every ionic step independently.  It accepts plain
OUTCAR files and gzip-compressed OUTCAR.gz files.

Typical use cases
-----------------
- Filter unconverged SCF steps from training data before ML force-field fitting:
  frames where SCF did not reach the required accuracy have unreliable energies
  and forces and should be excluded.
- Verify that a final geometry, TS geometry, or converged NEB image is genuinely
  converged before accepting it as a reference structure.
- Catch accidental POTCAR/POSCAR element-order mismatches that silently corrupt
  energies and forces.

CLI usage
---------
    vaspcheck-outcar                      # OUTCAR in current directory
    vaspcheck-outcar path/to/dir          # OUTCAR in given directory
    vaspcheck-outcar path/to/OUTCAR       # direct file path
    vaspcheck-outcar path/to/OUTCAR.gz    # gzip-compressed OUTCAR
"""

import gzip
import os
import re
from typing import List, Optional, Tuple

# VASP OUTCAR SCF convergence strings.
# VASP 5.x writes "aborting loop because EDIFF is reached".
# VASP 6.x writes "reached required accuracy - stopping SCF-cycle".
# Both are checked so the module works across VASP versions.
_SCF_CONVERGED_STRS = (
    "aborting loop because EDIFF is reached",   # VASP 5.x
    "reached required accuracy - stopping SCF-cycle",  # VASP 6.x
)

# Ionic convergence string (identical across VASP 5.x and 6.x).
_IONIC_CONVERGED_STR = ("reached required accuracy - stopping structural "
                        "energy minimisation")

# Matches the first SCF iteration of a new ionic step:
#   "------- Iteration    N(   1) -------"  (VASP 5.x — no leading space)
#   " ------ Iteration    N(   1) ------"   (some builds — one leading space)
# Leading whitespace is optional; dashes are always at least 38 characters.
_ITER_STEP_RE = re.compile(r'^\s*-{38,} Iteration\s+\d+\(\s*1\s*\)')


def _read_outcar_text(path: str) -> str:
    """Read an OUTCAR or OUTCAR.gz and return its full text."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', errors='replace') as fh:
            return fh.read()
    with open(path, 'r', errors='replace') as fh:
        return fh.read()


def _find_outcar(path: str) -> Optional[str]:
    """Return the OUTCAR file path for a file or directory argument.

    Accepts:
    - A direct path to an OUTCAR or OUTCAR.gz file.
    - A directory: looks for OUTCAR first, then OUTCAR.gz.

    Returns None if no OUTCAR is found.
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ('OUTCAR', 'OUTCAR.gz'):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _parse_potcar_poscar_elements(text: str) -> Tuple[List[str], List[str]]:
    """Parse element lists from the OUTCAR header.

    VASP echoes the POTCAR titles and the POSCAR element line near the top of
    every OUTCAR.  The POTCAR block appears first (each element written twice —
    once at the start and once at the end of the run), followed by the POSCAR
    line.  Example::

        POTCAR: PAW_PBE Si 05Jan2001
        POTCAR: PAW_PBE H  15Jun2001
        POTCAR: PAW_PBE C  08Apr2002
        ...
        POSCAR: Si H C

    PAW potential suffixes (_pv, _d, _sv, …) are stripped so that e.g.
    ``K_pv`` compares equal to ``K``.

    Parameters
    ----------
    text : str
        Full text of an OUTCAR file.

    Returns
    -------
    (poscar_elements, potcar_elements) : Tuple[List[str], List[str]]
        Element symbols in the order declared by POSCAR and POTCAR
        respectively.

    Raises
    ------
    ValueError
        If the POTCAR or POSCAR element lines cannot be found in the text.
    """
    potcar_lines = []
    poscar_elements: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('POTCAR:'):
            potcar_lines.append(stripped)
        elif stripped.startswith('POSCAR:'):
            poscar_elements = stripped.split(':', 1)[1].strip().split()
            break  # POSCAR line appears after all POTCAR lines

    if not potcar_lines:
        raise ValueError("No 'POTCAR:' lines found in OUTCAR — file may be truncated.")
    if not poscar_elements:
        raise ValueError("No 'POSCAR:' line found in OUTCAR — file may be truncated.")

    # VASP prints each POTCAR entry twice (header + footer); keep first half.
    n = len(potcar_lines)
    if n > 1 and n % 2 == 0:
        potcar_lines = potcar_lines[:n // 2]

    # Extract element symbol (second whitespace-delimited token after "PAW_PBE")
    # e.g. "POTCAR: PAW_PBE Si_pv 05Jan2001" → "Si"
    potcar_elements = []
    for line in potcar_lines:
        parts = line.split(':', 1)[1].strip().split()
        # parts[0] is the functional (PAW_PBE / PAW_LDA / …), parts[1] is element
        if len(parts) < 2:
            raise ValueError(f"Cannot parse element from POTCAR line: {line!r}")
        sym = parts[1].split('_')[0]  # strip _pv, _d, _sv, …
        potcar_elements.append(sym)

    # Strip PAW suffixes from POSCAR elements too (defensive)
    poscar_elements = [e.split('_')[0] for e in poscar_elements]

    return poscar_elements, potcar_elements


def check_potcar_poscar_alignment(outcar_path: str) -> dict:
    """Check that the POTCAR element order matches the POSCAR element order.

    A mismatch means VASP applied the wrong pseudopotential to each species,
    producing silently wrong energies and forces.  This is one of the most
    common setup mistakes when reusing input files across systems.

    Parameters
    ----------
    outcar_path : str
        Path to an OUTCAR or OUTCAR.gz file.

    Returns
    -------
    dict with keys:
        aligned         : bool       — True if POTCAR and POSCAR orders match
        poscar_elements : List[str]  — element symbols from the POSCAR line
        potcar_elements : List[str]  — element symbols from the POTCAR lines
        message         : str        — human-readable status or mismatch detail

    Raises
    ------
    ValueError
        If the POTCAR or POSCAR element lines cannot be parsed.
    """
    text = _read_outcar_text(outcar_path)
    poscar_elems, potcar_elems = _parse_potcar_poscar_elements(text)
    aligned = poscar_elems == potcar_elems
    if aligned:
        message = "OK"
    else:
        message = (f"MISMATCH — POSCAR: {poscar_elems}  POTCAR: {potcar_elems}")
    return {
        'aligned':         aligned,
        'poscar_elements': poscar_elems,
        'potcar_elements': potcar_elems,
        'message':         message,
    }


def check_scf_convergence_per_step(outcar_path: str) -> List[bool]:
    """Return SCF convergence status for each ionic step in the OUTCAR.

    Each element is True if the SCF cycle for that ionic step reached the
    required accuracy ("reached required accuracy - stopping SCF-cycle"),
    or False if it hit NELM without converging.

    A single-point calculation (NSW=0, IBRION=-1) returns a one-element list.
    An OUTCAR with no ionic step markers (empty or corrupted) returns [].

    Parameters
    ----------
    outcar_path : str
        Path to an OUTCAR or OUTCAR.gz file.

    Returns
    -------
    List[bool]
        One bool per ionic step (True = SCF converged for that step).
    """
    text = _read_outcar_text(outcar_path)
    lines = text.splitlines(keepends=True)

    # Locate line indices where a new ionic step begins (first SCF iteration)
    step_starts = [i for i, ln in enumerate(lines) if _ITER_STEP_RE.match(ln)]

    if not step_starts:
        return []

    results = []
    for idx, start in enumerate(step_starts):
        end = step_starts[idx + 1] if idx + 1 < len(step_starts) else len(lines)
        block = ''.join(lines[start:end])
        results.append(any(s in block for s in _SCF_CONVERGED_STRS))
    return results


def check_ionic_convergence(outcar_path: str) -> bool:
    """Return True if the ionic relaxation converged.

    Checks for the VASP message "reached required accuracy - stopping
    structural energy minimisation" anywhere in the OUTCAR.  This string
    is written by VASP for geometry optimisations, dimer TS searches, and
    NEB image relaxations alike.

    A single-point (NSW=0) OUTCAR has no ionic relaxation; returns False.
    A truncated/crashed OUTCAR returns False.

    Parameters
    ----------
    outcar_path : str
        Path to an OUTCAR or OUTCAR.gz file.

    Returns
    -------
    bool
    """
    text = _read_outcar_text(outcar_path)
    return _IONIC_CONVERGED_STR in text


def check_outcar(path: str) -> dict:
    """Run all convergence checks on an OUTCAR and return a summary dict.

    Accepts a direct file path (OUTCAR or OUTCAR.gz) or a directory
    containing one.

    Parameters
    ----------
    path : str
        Path to an OUTCAR, OUTCAR.gz, or a directory containing one.

    Returns
    -------
    dict with keys:
        outcar_path      : str             — resolved OUTCAR file path used
        potcar_aligned   : bool or None   — True=OK, False=mismatch, None=header absent
        potcar_message   : str            — "OK", mismatch detail, or reason not checked
        poscar_elements  : List[str]  — elements declared in POSCAR line
        potcar_elements  : List[str]  — elements declared in POTCAR lines
        n_steps          : int        — number of ionic steps found
        scf_converged    : List[bool] — per-step SCF convergence flags
        n_scf_failed     : int        — number of steps where SCF did not converge
        ionic_converged  : bool       — True if ionic/geometry relaxation converged

    Raises
    ------
    FileNotFoundError
        If no OUTCAR or OUTCAR.gz can be found at the given path.
    """
    outcar = _find_outcar(path)
    if outcar is None:
        raise FileNotFoundError(f"No OUTCAR or OUTCAR.gz found at: {path}")

    try:
        alignment = check_potcar_poscar_alignment(outcar)
    except ValueError as exc:
        alignment = {
            'aligned':         None,
            'poscar_elements': [],
            'potcar_elements': [],
            'message':         f"Could not check alignment: {exc}",
        }
    scf = check_scf_convergence_per_step(outcar)
    ionic = check_ionic_convergence(outcar)

    return {
        'outcar_path':      outcar,
        'potcar_aligned':   alignment['aligned'],
        'potcar_message':   alignment['message'],
        'poscar_elements':  alignment['poscar_elements'],
        'potcar_elements':  alignment['potcar_elements'],
        'n_steps':          len(scf),
        'scf_converged':    scf,
        'n_scf_failed':     sum(1 for ok in scf if not ok),
        'ionic_converged':  ionic,
    }


def run(path: str = '.') -> dict:
    """Check convergence for the OUTCAR at path and print a human-readable summary.

    Parameters
    ----------
    path : str
        Path to an OUTCAR, OUTCAR.gz, or directory.  Defaults to '.'.

    Returns
    -------
    dict — same structure as check_outcar().
    """
    result = check_outcar(path)
    print(f"OUTCAR      : {result['outcar_path']}")
    alignment_str = result['potcar_message']
    if result['potcar_aligned'] is None:
        alignment_str = f"UNKNOWN ({result['potcar_message']})"
    print(f"POTCAR/POSCAR alignment : {alignment_str}")
    print(f"Ionic steps : {result['n_steps']}")
    if result['n_scf_failed'] == 0:
        print("SCF         : all steps converged")
    else:
        failed_steps = [i + 1 for i, ok in enumerate(result['scf_converged']) if not ok]
        print(f"SCF         : {result['n_scf_failed']} step(s) did NOT converge"
              f" (ionic steps: {failed_steps})")
    status = "CONVERGED" if result['ionic_converged'] else "NOT CONVERGED"
    print(f"Ionic       : {status}")
    return result


def main():
    """CLI entry point: vaspcheck-outcar [path]"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Check SCF and ionic convergence of a VASP calculation "
                    "from its OUTCAR (plain or gzip-compressed).",
        epilog="Example: vaspcheck-outcar /path/to/vasp/run")
    parser.add_argument(
        "path", nargs="?", default=".",
        help="Path to OUTCAR, OUTCAR.gz, or directory containing one "
             "(default: current directory)")
    args = parser.parse_args()
    run(args.path)


if __name__ == "__main__":
    main()
