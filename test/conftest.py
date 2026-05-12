"""
Shared pytest configuration and fixtures for tools4vasp tests.
"""
import os
import pytest

# Force non-interactive matplotlib backend for all tests
os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
# Minimal VASP file content used as fixtures
# ---------------------------------------------------------------------------

POSCAR_CUBIC_H = """\
H atom in cubic box
1.0
10.0  0.0  0.0
 0.0 10.0  0.0
 0.0  0.0 10.0
H
1
Cartesian
0.0 0.0 0.0
"""

KPOINTS_GAMMA_444 = """\
Automatic
0
Gamma
4 4 4
0 0 0
"""

# Minimal OUTCAR fragment used to test element-extraction logic
OUTCAR_ELEMENTS = """\
asdfasdf
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
asdfasdf
POSCAR: Au N
asdfasdf
"""


# ---------------------------------------------------------------------------
# Minimal OUTCAR fragments for outcar_convergence tests
# ---------------------------------------------------------------------------

# Iteration line formats seen in the wild:
#   VASP 5.x — no leading space, "aborting loop because EDIFF is reached" for SCF
#   VASP 6.x — may have a leading space, "reached required accuracy - stopping SCF-cycle"
# Fixtures below use the VASP 5.x format (the dominant version in this codebase).

# One ionic step whose SCF converged (VASP 5.x format); ionic relaxation also converged.
OUTCAR_ONE_STEP_CONVERGED = """\
--------------------------------------- Iteration      1(   1)  ---------------------------------------
 DAV:   1    -0.100000E+03   some data
 DAV:   2    -0.101000E+03   some data
------------------------ aborting loop because EDIFF is reached ----------------------------------------
       reached required accuracy - stopping structural energy minimisation
"""

# One ionic step whose SCF did NOT converge (hit NELM); no ionic convergence.
OUTCAR_ONE_STEP_SCF_FAILED = """\
--------------------------------------- Iteration      1(   1)  ---------------------------------------
 DAV:   1    -0.100000E+03   some data
 DAV:   2    -0.100100E+03   some data
"""

# Three ionic steps: steps 1 and 3 SCF-converged, step 2 SCF-failed.
# No ionic convergence (geometry did not finish).
OUTCAR_MULTI_STEP_PARTIAL = """\
--------------------------------------- Iteration      1(   1)  ---------------------------------------
 DAV:   1    -0.100000E+03   some data
------------------------ aborting loop because EDIFF is reached ----------------------------------------
--------------------------------------- Iteration      2(   1)  ---------------------------------------
 DAV:   1    -0.101000E+03   some data
--------------------------------------- Iteration      3(   1)  ---------------------------------------
 DAV:   1    -0.102000E+03   some data
------------------------ aborting loop because EDIFF is reached ----------------------------------------
"""

# Two ionic steps both SCF-converged; ionic relaxation converged.
OUTCAR_MULTI_STEP_CONVERGED = """\
--------------------------------------- Iteration      1(   1)  ---------------------------------------
 DAV:   1    -0.100000E+03   some data
------------------------ aborting loop because EDIFF is reached ----------------------------------------
--------------------------------------- Iteration      2(   1)  ---------------------------------------
 DAV:   1    -0.101000E+03   some data
------------------------ aborting loop because EDIFF is reached ----------------------------------------
       reached required accuracy - stopping structural energy minimisation
"""

# Same content but using VASP 6.x format (leading space + different SCF string).
OUTCAR_ONE_STEP_CONVERGED_V6 = """\
 ----------------------------------------- Iteration    1(   1) -----------------------------------------
 DAV:   1    -0.100000E+03   some data
 DAV:   2    -0.101000E+03   some data
       reached required accuracy - stopping SCF-cycle
       reached required accuracy - stopping structural energy minimisation
"""

# ---------------------------------------------------------------------------
# OUTCAR header fragments for POTCAR/POSCAR alignment tests
# ---------------------------------------------------------------------------

# VASP writes POTCAR titles first (each element listed twice: header + footer),
# then the POSCAR element line.  Minimal realistic header for Si+H+C:

_POTCAR_POSCAR_HEADER_ALIGNED = """\
POTCAR: PAW_PBE Si 05Jan2001
POTCAR: PAW_PBE H  15Jun2001
POTCAR: PAW_PBE C  08Apr2002
POTCAR: PAW_PBE Si 05Jan2001
POTCAR: PAW_PBE H  15Jun2001
POTCAR: PAW_PBE C  08Apr2002
POSCAR: Si H C
"""

# Same but POTCAR order is Si C H while POSCAR says Si H C — mismatch.
_POTCAR_POSCAR_HEADER_MISMATCHED = """\
POTCAR: PAW_PBE Si 05Jan2001
POTCAR: PAW_PBE C  08Apr2002
POTCAR: PAW_PBE H  15Jun2001
POTCAR: PAW_PBE Si 05Jan2001
POTCAR: PAW_PBE C  08Apr2002
POTCAR: PAW_PBE H  15Jun2001
POSCAR: Si H C
"""

# Single-element calculation (only one POTCAR line — no "doubled" header).
_POTCAR_POSCAR_HEADER_SINGLE = """\
POTCAR: PAW_PBE Si 05Jan2001
POSCAR: Si
"""

# VASP 5.x style: lines have a leading space (real-world format seen in the field).
_POTCAR_POSCAR_HEADER_LEADING_SPACE = """\
 POTCAR:    PAW_PBE Si 05Jan2001
 POTCAR:    PAW_PBE H 15Jun2001
 POTCAR:    PAW_PBE O 08Apr2002
 POTCAR:    PAW_PBE C 08Apr2002
 POTCAR:    PAW_PBE Si 05Jan2001
 POTCAR:    PAW_PBE H 15Jun2001
 POTCAR:    PAW_PBE O 08Apr2002
 POTCAR:    PAW_PBE C 08Apr2002
 POSCAR: Si  H  O  C
"""

# PAW potential with suffix (_pv): should still compare equal to plain symbol.
_POTCAR_POSCAR_HEADER_PAW_SUFFIX = """\
POTCAR: PAW_PBE K_pv 06Sep2000
POTCAR: PAW_PBE O  08Apr2002
POTCAR: PAW_PBE K_pv 06Sep2000
POTCAR: PAW_PBE O  08Apr2002
POSCAR: K O
"""

# Full minimal OUTCARs combining a header with convergence body.
OUTCAR_ALIGNED_CONVERGED = (
    _POTCAR_POSCAR_HEADER_ALIGNED + OUTCAR_ONE_STEP_CONVERGED
)
OUTCAR_MISMATCHED_CONVERGED = (
    _POTCAR_POSCAR_HEADER_MISMATCHED + OUTCAR_ONE_STEP_CONVERGED
)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poscar_path(tmp_path):
    """Write a minimal POSCAR to a temp dir and return the file path."""
    p = tmp_path / "POSCAR"
    p.write_text(POSCAR_CUBIC_H)
    return p


@pytest.fixture
def kpoints_path(tmp_path):
    """Write a minimal KPOINTS to a temp dir and return the file path."""
    p = tmp_path / "KPOINTS"
    p.write_text(KPOINTS_GAMMA_444)
    return p


@pytest.fixture
def poscar_kpoints_dir(tmp_path):
    """Return a tmp_path that contains both POSCAR and KPOINTS."""
    (tmp_path / "POSCAR").write_text(POSCAR_CUBIC_H)
    (tmp_path / "KPOINTS").write_text(KPOINTS_GAMMA_444)
    return tmp_path


@pytest.fixture
def neb_data():
    """Minimal NEB plot data: 5 images on a symmetric barrier."""
    import math
    n_images = 5
    image_x = [i / (n_images - 1) for i in range(n_images)]
    energies = [math.sin(x * math.pi) for x in image_x]
    forces = [0.0] * n_images  # zero forces avoids tangent-clamping loop

    n_spline = 20
    spline_x = [i / (n_spline - 1) for i in range(n_spline)]
    spline_e = [math.sin(x * math.pi) for x in spline_x]

    return spline_x, image_x, energies, spline_e, forces
