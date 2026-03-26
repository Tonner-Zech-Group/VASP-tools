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
# vaspGetEF — get_max_f and read_free_mask
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


def test_get_max_f_mask_suppresses_frozen_atom():
    """A fully-frozen atom (mask all False) must not contribute to max force."""
    from tools4vasp.vaspGetEF import get_max_f
    from unittest.mock import MagicMock

    atoms = MagicMock()
    # atom 0 has large forces but is fully frozen; atom 1 is free
    atoms.get_forces.return_value = np.array([
        [10.0, 10.0, 10.0],  # frozen → zeroed
        [3.0,  4.0,  0.0],   # free   → |f| = 5
    ])
    mask = np.array([[False, False, False],
                     [True,  True,  True]], dtype=bool)
    assert get_max_f(atoms, free_mask=mask) == pytest.approx(5.0)


def test_get_max_f_mask_partial_component():
    """A per-component mask (T T F) must zero only the frozen component."""
    from tools4vasp.vaspGetEF import get_max_f
    from unittest.mock import MagicMock

    atoms = MagicMock()
    # One atom, z component frozen.  Active force: sqrt(3^2 + 4^2) = 5.
    atoms.get_forces.return_value = np.array([[3.0, 4.0, 99.0]])
    mask = np.array([[True, True, False]], dtype=bool)
    assert get_max_f(atoms, free_mask=mask) == pytest.approx(5.0)


def test_get_max_f_no_mask_unchanged():
    """Without a mask the function behaves exactly as before."""
    from tools4vasp.vaspGetEF import get_max_f
    from unittest.mock import MagicMock

    atoms = MagicMock()
    atoms.get_forces.return_value = np.array([[0.0, 0.0, 7.0]])
    assert get_max_f(atoms) == pytest.approx(7.0)


# POSCAR fixtures for read_free_mask tests

_POSCAR_NO_SD = """\
H2 molecule
1.0
10.0  0.0  0.0
 0.0 10.0  0.0
 0.0  0.0 10.0
H
2
Cartesian
0.0 0.0 0.0
5.0 0.0 0.0
"""

_POSCAR_ALL_FROZEN = """\
H2 fully frozen
1.0
10.0  0.0  0.0
 0.0 10.0  0.0
 0.0  0.0 10.0
H
2
Selective dynamics
Cartesian
0.0 0.0 0.0 F F F
5.0 0.0 0.0 F F F
"""

_POSCAR_MIXED = """\
H2 mixed constraints
1.0
10.0  0.0  0.0
 0.0 10.0  0.0
 0.0  0.0 10.0
H
2
Selective dynamics
Cartesian
0.0 0.0 0.0 T T T
5.0 0.0 0.0 T T F
"""

_POSCAR_PARTIAL = """\
H2 per-component
1.0
10.0  0.0  0.0
 0.0 10.0  0.0
 0.0  0.0 10.0
H
2
Selective dynamics
Cartesian
0.0 0.0 0.0 F F T
5.0 0.0 0.0 T F T
"""


def test_read_free_mask_no_selective_dynamics(tmp_path):
    """POSCAR without selective dynamics must return None."""
    from tools4vasp.vaspGetEF import read_free_mask
    p = tmp_path / "POSCAR"
    p.write_text(_POSCAR_NO_SD)
    assert read_free_mask(str(p)) is None


def test_read_free_mask_missing_file():
    """Non-existent file must return None without raising."""
    from tools4vasp.vaspGetEF import read_free_mask
    assert read_free_mask("/nonexistent/POSCAR") is None


def test_read_free_mask_all_frozen_returns_none(tmp_path):
    """All-frozen POSCAR (F F F for every atom) still returns a mask."""
    from tools4vasp.vaspGetEF import read_free_mask
    p = tmp_path / "POSCAR"
    p.write_text(_POSCAR_ALL_FROZEN)
    mask = read_free_mask(str(p))
    assert mask is not None
    assert mask.shape == (2, 3)
    assert not mask.any()


def test_read_free_mask_mixed_shape(tmp_path):
    """Mask must have correct shape (natoms, 3)."""
    from tools4vasp.vaspGetEF import read_free_mask
    p = tmp_path / "POSCAR"
    p.write_text(_POSCAR_MIXED)
    mask = read_free_mask(str(p))
    assert mask is not None
    assert mask.shape == (2, 3)


def test_read_free_mask_mixed_first_atom_free(tmp_path):
    """First atom (T T T) must be entirely free."""
    from tools4vasp.vaspGetEF import read_free_mask
    p = tmp_path / "POSCAR"
    p.write_text(_POSCAR_MIXED)
    mask = read_free_mask(str(p))
    assert mask[0].all()


def test_read_free_mask_mixed_second_atom_z_frozen(tmp_path):
    """Second atom (T T F) must have only z frozen."""
    from tools4vasp.vaspGetEF import read_free_mask
    p = tmp_path / "POSCAR"
    p.write_text(_POSCAR_MIXED)
    mask = read_free_mask(str(p))
    assert mask[1, 0] is np.bool_(True)
    assert mask[1, 1] is np.bool_(True)
    assert mask[1, 2] is np.bool_(False)


def test_read_free_mask_partial_components(tmp_path):
    """Per-component constraints (F F T / T F T) must be read correctly."""
    from tools4vasp.vaspGetEF import read_free_mask
    p = tmp_path / "POSCAR"
    p.write_text(_POSCAR_PARTIAL)
    mask = read_free_mask(str(p))
    assert mask is not None
    # atom 0: F F T
    np.testing.assert_array_equal(mask[0], [False, False, True])
    # atom 1: T F T
    np.testing.assert_array_equal(mask[1], [True, False, True])


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
    import io
    import sys
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
    import io
    import sys
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
    sig = inspect.signature(neb2movie.run)
    default = sig.parameters["wrap"].default
    assert isinstance(default, bool), \
        f"Expected bool, got {type(default).__name__!r}"


# ---------------------------------------------------------------------------
# viewMode — read_modecar, increment_positions, make_animation
# ---------------------------------------------------------------------------

_MODECAR_CONTENT = (
    "    1.0000000000E+00    2.0000000000E+00    3.0000000000E+00\n"
    "   -1.0000000000E+00    0.0000000000E+00    5.0000000000E-01\n"
)


def test_read_modecar_parses_correctly(tmp_path):
    """read_modecar() must parse a MODECAR file into a (N,3) float64 array."""
    from tools4vasp.viewMode import read_modecar
    modecar = tmp_path / "MODECAR"
    modecar.write_text(_MODECAR_CONTENT)
    mode = read_modecar(str(modecar))
    assert mode.shape == (2, 3)
    assert mode.dtype == np.float64
    assert mode[0, 0] == pytest.approx(1.0)
    assert mode[1, 0] == pytest.approx(-1.0)
    assert mode[1, 2] == pytest.approx(0.5)


def test_increment_positions_applies_scaled_mode():
    """increment_positions() must shift positions by mode * factor."""
    import ase
    from tools4vasp.viewMode import increment_positions
    atoms = ase.Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10])
    mode = np.array([[1.0, 2.0, 3.0]])
    result = increment_positions(atoms, mode, 0.5)
    np.testing.assert_allclose(result.positions, [[0.5, 1.0, 1.5]])
    # Original must not be mutated
    assert atoms.positions[0, 0] == pytest.approx(0.0)


def test_make_animation_internal_frames_divisible_by_4():
    """make_animation() must adjust frames so (len(anim) - 1) % 4 == 0."""
    import ase
    from tools4vasp.viewMode import make_animation
    atoms = ase.Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10])
    mode = np.array([[0.1, 0.0, 0.0]])
    for frames_in in [5, 6, 7, 9, 13]:
        anim = make_animation(atoms.copy(), mode, frames_in, 0.05)
        assert (len(anim) - 1) % 4 == 0, (
            f"frames_in={frames_in}: len(anim)-1={len(anim)-1} not divisible by 4"
        )


def test_make_animation_first_frame_is_original():
    """make_animation() first frame must equal the original atoms."""
    import ase
    from tools4vasp.viewMode import make_animation
    atoms = ase.Atoms('H', positions=[[1.0, 2.0, 3.0]], cell=[10, 10, 10])
    mode = np.array([[1.0, 0.0, 0.0]])
    anim = make_animation(atoms, mode, 5, 0.05)
    np.testing.assert_allclose(anim[0].positions, [[1.0, 2.0, 3.0]])


# ---------------------------------------------------------------------------
# freq2mode — get_atomic_mass, generate_mw, get_frequencies
# ---------------------------------------------------------------------------

def test_get_atomic_mass_hydrogen_positive():
    """get_atomic_mass() must return a positive value for hydrogen (Z=1)."""
    from tools4vasp.freq2mode import get_atomic_mass
    mass = get_atomic_mass(1)  # atomic number of H
    assert mass > 0


def test_generate_mw_divides_by_sqrt_mass():
    """generate_mw() must divide each displacement component by sqrt(atom mass)."""
    from unittest.mock import MagicMock
    from tools4vasp.freq2mode import generate_mw
    atoms = MagicMock()
    atoms.get_masses.return_value = np.array([4.0])  # sqrt(4) = 2
    frequency = [[2.0, 0.0, -4.0]]
    result = generate_mw(frequency, atoms)
    assert result[0][0] == pytest.approx(1.0)   # 2.0 / 2
    assert result[0][1] == pytest.approx(0.0)   # zero stays zero
    assert result[0][2] == pytest.approx(-2.0)  # -4.0 / 2


def test_get_frequencies_finds_imaginary_lines():
    """get_frequencies() must return only lines containing 'f/i='."""
    from tools4vasp.freq2mode import get_frequencies
    lines = [
        "   1 f  =    1.2 THz\n",
        "   2 f/i=    0.5 THz\n",
        "   3 f/i=    0.3 THz\n",
        "   4 f  =    2.0 THz\n",
    ]
    result = get_frequencies(lines)
    assert len(result) == 2
    assert all("f/i=" in r for r in result)


def test_get_frequencies_stops_at_eigenvector_sentinel():
    """get_frequencies() must not include lines after the eigenvector sentinel."""
    from tools4vasp.freq2mode import get_frequencies
    lines = [
        "   1 f/i=    0.5 THz\n",
        "Eigenvectors after division by SQRT(mass)\n",
        "   2 f/i=    0.1 THz\n",  # after sentinel — must be excluded
    ]
    result = get_frequencies(lines)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# split_vasp_freq — read_input_structure, get_nfree_delta, split
# ---------------------------------------------------------------------------

_POSCAR_TWO_H = """\
H2 in cubic box
1.0
10.0  0.0  0.0
 0.0 10.0  0.0
 0.0  0.0 10.0
H
2
Cartesian
0.0 0.0 0.0
5.0 0.0 0.0
"""

_POSCAR_ONE_FIXED = """\
H2 in cubic box (one fixed)
1.0
10.0  0.0  0.0
 0.0 10.0  0.0
 0.0  0.0 10.0
H
2
Selective dynamics
Cartesian
0.0 0.0 0.0 T T T
5.0 0.0 0.0 F F F
"""


def test_read_input_structure_unconstrained(tmp_path):
    """read_input_structure() must report all atoms as free when no constraints."""
    from tools4vasp.split_vasp_freq import read_input_structure
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_TWO_H)
    result = read_input_structure(str(poscar), verbose=False)
    assert result['n_free_atoms'] == 2
    assert len(result['indices']) == 2


def test_read_input_structure_with_fixatoms(tmp_path):
    """read_input_structure() must exclude FixAtoms indices from free atoms."""
    from tools4vasp.split_vasp_freq import read_input_structure
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_ONE_FIXED)
    result = read_input_structure(str(poscar), verbose=False)
    assert result['n_free_atoms'] == 1
    assert len(result['indices']) == 1


def test_get_nfree_delta_nfree2(tmp_path):
    """get_nfree_delta() must parse NFREE=2 and POTIM correctly."""
    from tools4vasp.split_vasp_freq import get_nfree_delta
    incar = tmp_path / "INCAR"
    incar.write_text("NFREE = 2\nPOTIM = 0.015\n")
    n_free, delta = get_nfree_delta(str(incar), verbose=False)
    assert n_free == 2
    assert delta == pytest.approx(0.015)


def test_get_nfree_delta_nfree4(tmp_path):
    """get_nfree_delta() must parse NFREE=4 correctly."""
    from tools4vasp.split_vasp_freq import get_nfree_delta
    incar = tmp_path / "INCAR"
    incar.write_text("NFREE = 4\nPOTIM = 0.030\n")
    n_free, delta = get_nfree_delta(str(incar), verbose=False)
    assert n_free == 4
    assert delta == pytest.approx(0.030)


def test_get_nfree_delta_invalid_nfree_raises(tmp_path):
    """get_nfree_delta() must raise ValueError for NFREE not in [2, 4]."""
    from tools4vasp.split_vasp_freq import get_nfree_delta
    incar = tmp_path / "INCAR"
    incar.write_text("NFREE = 3\nPOTIM = 0.015\n")
    with pytest.raises(ValueError):
        get_nfree_delta(str(incar), verbose=False)


def test_split_creates_freq_subdirs(tmp_path):
    """split() must create freq_001, freq_002 directories each with a POSCAR."""
    from tools4vasp.split_vasp_freq import split
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_TWO_H)
    # 2 free atoms, 1 per calc → 2 subdirs
    split(str(poscar), 1, cwd=str(tmp_path), verbose=False)
    assert (tmp_path / "freq_001").is_dir()
    assert (tmp_path / "freq_002").is_dir()
    assert (tmp_path / "freq_001" / "POSCAR").is_file()
    assert (tmp_path / "freq_002" / "POSCAR").is_file()


def test_split_raises_if_dir_already_exists(tmp_path):
    """split() must raise RuntimeError if a freq_NNN dir already exists."""
    from tools4vasp.split_vasp_freq import split
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_TWO_H)
    (tmp_path / "freq_001").mkdir()
    with pytest.raises(RuntimeError):
        split(str(poscar), 2, cwd=str(tmp_path), verbose=False)


# ---------------------------------------------------------------------------
# neb2movie — CONTCAR/POSCAR file selection logic
# ---------------------------------------------------------------------------

_POSCAR_SIMPLE = """\
H atom
1.0
10.0  0.0  0.0
 0.0 10.0  0.0
 0.0  0.0 10.0
H
1
Cartesian
0.0 0.0 0.0
"""


def test_neb2movie_prefers_contcar(tmp_path):
    """main() must read CONTCAR for intermediate images when it exists."""
    from unittest.mock import MagicMock, patch
    from tools4vasp import neb2movie

    for d in ['00', '01', '02']:
        (tmp_path / d).mkdir()
        (tmp_path / d / 'POSCAR').write_text(_POSCAR_SIMPLE)
    (tmp_path / '01' / 'CONTCAR').write_text(_POSCAR_SIMPLE)
    (tmp_path / '02' / 'CONTCAR').write_text(_POSCAR_SIMPLE)

    fake_atom = MagicMock()
    with patch('tools4vasp.neb2movie.io.read', return_value=fake_atom) as mock_read:
        neb2movie.run(outFile=str(tmp_path / 'movie.xyz'), workdir=str(tmp_path))

    paths = [str(call.args[0]) for call in mock_read.call_args_list]
    assert any('CONTCAR' in p for p in paths), "CONTCAR must be used for middle images"
    assert any('POSCAR' in p for p in paths), "POSCAR must be used for endpoint images"


def test_neb2movie_falls_back_to_poscar(tmp_path):
    """main() must use POSCAR for all images when no CONTCAR exists."""
    from unittest.mock import MagicMock, patch
    from tools4vasp import neb2movie

    for d in ['00', '01', '02']:
        (tmp_path / d).mkdir()
        (tmp_path / d / 'POSCAR').write_text(_POSCAR_SIMPLE)

    fake_atom = MagicMock()
    with patch('tools4vasp.neb2movie.io.read', return_value=fake_atom) as mock_read:
        neb2movie.run(outFile=str(tmp_path / 'movie.xyz'), workdir=str(tmp_path))

    paths = [str(call.args[0]) for call in mock_read.call_args_list]
    assert all('POSCAR' in p for p in paths)
    assert not any('CONTCAR' in p for p in paths)


def test_neb2movie_raises_if_neither_poscar_nor_contcar(tmp_path):
    """main() must raise RuntimeError when 01/ has neither POSCAR nor CONTCAR."""
    from tools4vasp import neb2movie
    (tmp_path / '01').mkdir()
    with pytest.raises(RuntimeError):
        neb2movie.run(outFile=str(tmp_path / 'movie.xyz'), workdir=str(tmp_path))


# ---------------------------------------------------------------------------
# chgcar2cube — return values and spin-polarised output
# ---------------------------------------------------------------------------

def test_chgcar2cube_returns_integral(tmp_path):
    """chgcar2cube() must return the computed integral when return_integrals=True."""
    from unittest.mock import MagicMock, patch
    from tools4vasp.chgcar2cube import chgcar2cube

    infile = tmp_path / "CHGCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "out")

    mock_chgcar = MagicMock()
    mock_chgcar.data = {"total": np.ones((2, 2, 2))}  # sum=8, n_data=8 → integral=1.0
    mock_atoms = MagicMock()

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.chgcar2cube.write_cube"):
        result = chgcar2cube([str(infile)], [outbase], verbose=False, return_integrals=True)

    assert result == pytest.approx(1.0)


def test_chgcar2cube_spinpol_writes_mag_cube(tmp_path):
    """chgcar2cube() must write a _mag.cube file for spin-polarised input."""
    from unittest.mock import MagicMock, patch
    from tools4vasp.chgcar2cube import chgcar2cube

    infile = tmp_path / "CHGCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "out")

    mock_chgcar = MagicMock()
    mock_chgcar.data = {
        "total": np.ones((2, 2, 2)),
        "diff":  np.zeros((2, 2, 2)),
    }
    mock_atoms = MagicMock()

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.chgcar2cube.write_cube") as mock_write:
        chgcar2cube([str(infile)], [outbase], verbose=False)

    written_files = [c.args[0].name for c in mock_write.call_args_list]
    assert any("_mag.cube" in f for f in written_files), "_mag.cube must be written for spinpol"


# ---------------------------------------------------------------------------
# elf2cube — spin-polarised output and return values
# ---------------------------------------------------------------------------

def test_elf2cube_spinpol_writes_three_cubes(tmp_path):
    """elf2cube() must write _up, _down, and _diff cube files for spinpol input."""
    from unittest.mock import MagicMock, patch
    from tools4vasp.elf2cube import elf2cube

    infile = tmp_path / "ELFCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "out")

    mock_elfcar = MagicMock()
    mock_elfcar.data = {
        "total": np.ones((2, 2, 2)) * 0.6,
        "diff":  np.ones((2, 2, 2)) * 0.4,
    }
    mock_atoms = MagicMock()

    with patch("tools4vasp.elf2cube.Elfcar.from_file", return_value=mock_elfcar), \
         patch("tools4vasp.elf2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.elf2cube.write_cube") as mock_write:
        elf2cube([str(infile)], [outbase], verbose=False)

    written_files = [c.args[0].name for c in mock_write.call_args_list]
    assert any("_up.cube" in f for f in written_files)
    assert any("_down.cube" in f for f in written_files)
    assert any("_diff.cube" in f for f in written_files)
    assert mock_write.call_count == 3


def test_elf2cube_returns_integral(tmp_path):
    """elf2cube() must return the computed integral when return_integrals=True."""
    from unittest.mock import MagicMock, patch
    from tools4vasp.elf2cube import elf2cube

    infile = tmp_path / "ELFCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "out")

    mock_elfcar = MagicMock()
    # non-spinpol: full_data = total; sum(ones(2,2,2)) = 8
    mock_elfcar.data = {"total": np.ones((2, 2, 2))}
    mock_atoms = MagicMock()

    with patch("tools4vasp.elf2cube.Elfcar.from_file", return_value=mock_elfcar), \
         patch("tools4vasp.elf2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.elf2cube.write_cube"):
        result = elf2cube([str(infile)], [outbase], verbose=False, return_integrals=True)

    assert result == pytest.approx(8.0)
