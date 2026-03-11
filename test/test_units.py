"""
Unit tests for pure/logic functions that need no VASP installation.

These tests exercise real computation (not mocked) and are fast to run.
"""
import math
from io import StringIO

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# vaspcheck — element extraction from OUTCAR
# (existing test preserved here for completeness; also kept in test_tools4vasp.py)
# ---------------------------------------------------------------------------

OUTCAR_MULTI = """
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
POTCAR: PAW_PBE H 08Apr2002
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
POTCAR: PAW_PBE H 08Apr2002
POSCAR: Au N H
"""

OUTCAR_UNDERSCORE = OUTCAR_MULTI.replace("Au", "Au_s")
OUTCAR_MISMATCH = OUTCAR_MULTI.replace("POSCAR: Au N H", "POSCAR: Au H N")


def test_get_elements_match():
    from tools4vasp.vaspcheck import _get_elements_from_outcar
    poscar, potcar = _get_elements_from_outcar(StringIO(OUTCAR_MULTI))
    assert poscar == ["Au", "N", "H"]
    assert potcar == ["Au", "N", "H"]


def test_get_elements_underscore_stripped():
    from tools4vasp.vaspcheck import _get_elements_from_outcar
    poscar, potcar = _get_elements_from_outcar(StringIO(OUTCAR_UNDERSCORE))
    assert poscar == ["Au", "N", "H"]
    assert potcar == ["Au", "N", "H"]


def test_get_elements_mismatch_detected():
    from tools4vasp.vaspcheck import _get_elements_from_outcar
    poscar, potcar = _get_elements_from_outcar(StringIO(OUTCAR_MISMATCH))
    assert poscar != potcar


# ---------------------------------------------------------------------------
# plotIRC — calc_rms
# ---------------------------------------------------------------------------

def test_calc_rms_identical_positions():
    from tools4vasp.plotIRC import calc_rms
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert calc_rms(pos, pos) == pytest.approx(0.0)


def test_calc_rms_known_value():
    from tools4vasp.plotIRC import calc_rms
    pos1 = np.zeros((1, 3))
    pos2 = np.array([[3.0, 4.0, 0.0]])
    # step = [3, 4, 0], mean(square) = (9+16+0)/3, sqrt = sqrt(25/3)
    expected = math.sqrt(25.0 / 3.0)
    assert calc_rms(pos1, pos2) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# vaspGetEF — get_max_f
# ---------------------------------------------------------------------------

def test_get_max_f_single_atom():
    from tools4vasp.vaspGetEF import get_max_f
    from unittest.mock import MagicMock

    atoms = MagicMock()
    atoms.get_forces.return_value = np.array([[3.0, 4.0, 0.0]])
    assert get_max_f(atoms) == pytest.approx(5.0)


def test_get_max_f_picks_maximum():
    from tools4vasp.vaspGetEF import get_max_f
    from unittest.mock import MagicMock

    atoms = MagicMock()
    atoms.get_forces.return_value = np.array([
        [1.0, 0.0, 0.0],   # |f| = 1
        [3.0, 4.0, 0.0],   # |f| = 5  ← max
        [0.0, 2.0, 0.0],   # |f| = 2
    ])
    assert get_max_f(atoms) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# kspacing2kgrid — get_kgrid (logic, with POSCAR fixture)
# ---------------------------------------------------------------------------

def test_get_kgrid_cubic_box(poscar_path, monkeypatch):
    """
    For a 10 Å cubic box, kspacing=0.3 Å⁻¹ should give kgrid ≥ 1 in all dims.
    |b_i| = 2π/10 ≈ 0.628 Å⁻¹.  kgrid = ceil(0.628 / 0.3) = ceil(2.09) = 3.
    """
    from tools4vasp.kspacing2kgrid import get_kgrid
    monkeypatch.chdir(poscar_path.parent)

    # Should not raise; the output is just printed
    get_kgrid(0.3)


def test_get_kgrid_large_spacing_gives_one(poscar_path, monkeypatch):
    """Very large kspacing should result in a 1×1×1 grid (minimum enforced)."""
    import io, sys
    from tools4vasp.kspacing2kgrid import get_kgrid
    monkeypatch.chdir(poscar_path.parent)

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    get_kgrid(100.0)
    out = captured.getvalue()
    assert "1 1 1" in out


# ---------------------------------------------------------------------------
# kgrid2kspacing — get_kspacing (logic, with POSCAR + KPOINTS fixtures)
# ---------------------------------------------------------------------------

def test_get_kspacing_runs(poscar_kpoints_dir, monkeypatch):
    """get_kspacing() must run without error for a valid POSCAR + KPOINTS."""
    from tools4vasp.kgrid2kspacing import get_kspacing
    monkeypatch.chdir(poscar_kpoints_dir)
    get_kspacing()


def test_get_kspacing_output_contains_values(poscar_kpoints_dir, monkeypatch):
    """get_kspacing() output must report the kgrid and computed kspacing."""
    import io, sys
    from tools4vasp.kgrid2kspacing import get_kspacing
    monkeypatch.chdir(poscar_kpoints_dir)

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    get_kspacing()
    out = captured.getvalue()
    assert "4 4 4" in out       # kgrid from KPOINTS fixture
    assert "KSPACING" in out


# ---------------------------------------------------------------------------
# freq2mode — write_modecar formatting
# ---------------------------------------------------------------------------

def test_write_modecar_positive_spacing(tmp_path):
    """Positive values must have 4-space padding; negative values 3-space."""
    from tools4vasp.freq2mode import write_modecar

    frequency = [[1.0, -2.0, 0.0]]
    outfile = str(tmp_path / "MODECAR")
    write_modecar(frequency, outfile)

    line = open(outfile).readlines()[0]
    # Positive value → "    1.0000000000E+00"  (4 spaces before)
    assert "    1.0000000000E+00" in line
    # Negative value → "   -2.0000000000E+00"  (3 spaces before minus sign)
    assert "   -2.0000000000E+00" in line


# ---------------------------------------------------------------------------
# plotNEB — main() is callable (smoke test)
# ---------------------------------------------------------------------------

def test_plotNEB_main_is_callable():
    from tools4vasp import plotNEB
    assert callable(plotNEB.main)


# ---------------------------------------------------------------------------
# neb2movie — wrap parameter type
# ---------------------------------------------------------------------------

def test_neb2movie_wrap_is_bool():
    """The wrap parameter default must be a bool, not a string."""
    import inspect
    from tools4vasp import neb2movie
    sig = inspect.signature(neb2movie.main)
    default = sig.parameters["wrap"].default
    assert isinstance(default, bool), \
        f"Expected bool, got {type(default).__name__!r}"
