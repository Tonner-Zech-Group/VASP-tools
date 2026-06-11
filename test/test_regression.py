"""
Regression tests — one test per bug fixed in the bugfix PR.

Each test is named after the bug it guards against.  They are all designed
to FAIL on the original (unfixed) code and PASS after the fix.
"""
import inspect
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bug 1 & 2 — chgcar2cube / elf2cube: output not written when verbose=False
# ---------------------------------------------------------------------------

def test_chgcar2cube_writes_when_not_verbose(tmp_path):
    """Cube file must be written even when verbose=False (indentation bug)."""
    from tools4vasp.chgcar2cube import chgcar2cube

    infile = tmp_path / "CHGCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "output")

    mock_chgcar = MagicMock()
    mock_chgcar.data = {"total": np.ones((4, 4, 4))}  # non-spin-polarised
    mock_atoms = MagicMock()

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.chgcar2cube.write_cube") as mock_write:
        chgcar2cube([str(infile)], [outbase], verbose=False)

    assert mock_write.called, "write_cube must be called even with verbose=False"


def test_chgcar2cube_writes_when_verbose(tmp_path):
    """Sanity-check: cube file is also written when verbose=True."""
    from tools4vasp.chgcar2cube import chgcar2cube

    infile = tmp_path / "CHGCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "output")

    mock_chgcar = MagicMock()
    mock_chgcar.data = {"total": np.ones((4, 4, 4))}
    mock_atoms = MagicMock()

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.chgcar2cube.write_cube") as mock_write:
        chgcar2cube([str(infile)], [outbase], verbose=True)

    assert mock_write.called


def test_elf2cube_writes_when_not_verbose(tmp_path):
    """Cube file must be written even when verbose=False (indentation bug)."""
    from tools4vasp.elf2cube import elf2cube

    infile = tmp_path / "ELFCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "output")

    mock_elfcar = MagicMock()
    mock_elfcar.data = {"total": np.ones((4, 4, 4))}  # non-spin-polarised
    mock_atoms = MagicMock()

    with patch("tools4vasp.elf2cube.Elfcar.from_file", return_value=mock_elfcar), \
         patch("tools4vasp.elf2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.elf2cube.write_cube") as mock_write:
        elf2cube([str(infile)], [outbase], verbose=False)

    assert mock_write.called, "write_cube must be called even with verbose=False"


# ---------------------------------------------------------------------------
# Bug 3 — plotNEB: NameError when unit has no "/"
# ---------------------------------------------------------------------------

def test_plotNEB_unit_no_slash_does_not_raise(tmp_path, neb_data):
    """plot() must not raise NameError when unit contains no '/' (e.g. 'eV')."""
    from tools4vasp.plotNEB import plot
    spline_x, image_x, energies, spline_e, forces = neb_data
    out = str(tmp_path / "neb.png")
    # This raised NameError before the fix
    plot(spline_x, image_x, energies, spline_e, forces, out, unit="eV")


def test_plotNEB_unit_slash_still_works(tmp_path, neb_data):
    """plot() must continue to work for units with '/' (e.g. 'kJ/mol')."""
    from tools4vasp.plotNEB import plot
    spline_x, image_x, energies, spline_e, forces = neb_data
    out = str(tmp_path / "neb.png")
    plot(spline_x, image_x, energies, spline_e, forces, out, unit="kJ/mol")


# ---------------------------------------------------------------------------
# Bug 4 — plotNEB: hardcoded developer path removed
# ---------------------------------------------------------------------------

def test_plotNEB_no_hardcoded_path():
    """The hardcoded /home/patrickm exec() call must not exist in the source."""
    import tools4vasp.plotNEB as mod
    import inspect as _inspect
    src = _inspect.getsource(mod)
    assert "/home/patrickm" not in src, \
        "Hardcoded developer path still present in plotNEB.py"


# ---------------------------------------------------------------------------
# Bug 5 — plotNEB: delta mutated across loop iterations
# ---------------------------------------------------------------------------

def test_plotNEB_delta_not_mutated_across_images(tmp_path):
    """
    Tangent delta must be reset independently for each NEB image.

    Bug: `delta` was modified inside the clamping while-loop and the reduced
    value leaked into the next outer for-loop iteration.

    We verify this by capturing the x-extents of each drawn tangent line
    (via a mock) and asserting all images use the same (original) delta.
    Force on image 0 is large enough to trigger clamping; all others are 0.
    With the fix every image should have identical x-width = 2 * delta.
    """
    from unittest.mock import patch
    from tools4vasp.plotNEB import plot

    n = 4
    image_x = [i / (n - 1) for i in range(n)]
    energies = [math.sin(x * math.pi) for x in image_x]
    # force=5 on image 0 triggers ~6 while-loop iterations (within the 11 limit)
    forces = [5.0, 0.0, 0.0, 0.0]

    n_sp = 20
    spline_x = [i / (n_sp - 1) for i in range(n_sp)]
    spline_e = [math.sin(x * math.pi) for x in spline_x]

    out = str(tmp_path / "neb_delta.png")

    tangent_xwidths = []

    def noop_plot_line(xs, ys, **kw):
        pass

    def capture_plot(xs, ys, **kwargs):
        color = kwargs.get("color", "")
        if color == "green":
            tangent_xwidths.append(xs[1] - xs[0])
        return noop_plot_line(xs, ys, **kwargs)

    with patch("tools4vasp.plotNEB.plt") as mock_plt:
        mock_plt.figure.return_value.gca.return_value = mock_plt
        mock_plt.plot.side_effect = capture_plot
        mock_plt.savefig = lambda *a, **kw: None
        mock_plt.close = lambda: None
        mock_plt.scatter = lambda *a, **kw: None
        mock_plt.xlabel = lambda *a, **kw: None
        mock_plt.ylabel = lambda *a, **kw: None
        mock_plt.legend = lambda: None
        mock_plt.tight_layout = lambda: None

        plot(spline_x, image_x, energies, spline_e, forces, out, unit="eV")

    assert len(tangent_xwidths) == n, f"Expected {n} tangent lines, got {len(tangent_xwidths)}"
    # Images 1-3 (zero force) must have the same delta as the base delta,
    # not a delta reduced by image 0's clamping.
    assert all(w == pytest.approx(tangent_xwidths[1]) for w in tangent_xwidths[1:]), \
        "Tangent widths differ across zero-force images — delta is leaking between iterations"


# ---------------------------------------------------------------------------
# Bug 6 — elf2cube / plotIRC / kgrid2kspacing / kspacing2kgrid:
#          missing main() entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", [
    "tools4vasp.elf2cube",
    "tools4vasp.plotIRC",
    "tools4vasp.kgrid2kspacing",
    "tools4vasp.kspacing2kgrid",
])
def test_entry_point_main_exists_and_callable(module_name):
    """Every console-script module must expose a callable main()."""
    import importlib
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "main"), f"{module_name} has no main()"
    assert callable(mod.main), f"{module_name}.main is not callable"


# ---------------------------------------------------------------------------
# Bug 7 — plotIRC: generate_plot() accessed global `args`
# ---------------------------------------------------------------------------

def test_plotIRC_generate_plot_no_global_args():
    """generate_plot() must not require a global `args` variable."""
    from tools4vasp.plotIRC import generate_plot

    pos = np.zeros((3, 3))
    irc_e = [pos.copy(), [0.0, 0.1, 0.2], [-100.0, -99.5, -98.0]]
    irc_p = [pos.copy(), [0.0, 0.1, 0.2], [-100.0, -99.5, -98.0]]

    # Before the fix this raised NameError: name 'args' is not defined
    result = generate_plot(irc_e, irc_p, None, offset=None)
    assert len(result) == 6


def test_plotIRC_generate_plot_with_offset():
    """generate_plot() offset parameter must be applied correctly."""
    from tools4vasp.plotIRC import generate_plot

    pos = np.zeros((3, 3))
    irc_e = [pos.copy(), [0.0, 0.1, 0.2], [-100.0, -99.5, -98.0]]
    irc_p = [pos.copy(), [0.0, 0.1, 0.2], [-100.0, -99.5, -98.0]]

    result = generate_plot(irc_e, irc_p, None, offset=0.0)
    assert len(result) == 6


# ---------------------------------------------------------------------------
# Bug 8 — neb2movie: wrap default was string 'False' (truthy)
# ---------------------------------------------------------------------------

def test_neb2movie_wrap_default_is_bool_false():
    """main() wrap parameter default must be the boolean False, not the string 'False'."""
    from tools4vasp import neb2movie

    sig = inspect.signature(neb2movie.run)
    default = sig.parameters["wrap"].default
    assert default is False, \
        f"wrap default is {default!r} (type {type(default).__name__}), expected False (bool)"


# ---------------------------------------------------------------------------
# Bug 9 — freq2mode: write_modecar wrote file inside inner loop
# ---------------------------------------------------------------------------

def test_write_modecar_writes_all_lines(tmp_path):
    """write_modecar must write every frequency line, not just partial content."""
    from tools4vasp.freq2mode import write_modecar

    frequency = [
        [1.0, 2.0, 3.0],
        [-1.0, 0.5, -0.5],
        [0.0, 0.0, 1.0],
        [0.1, -0.2, 0.3],
    ]
    outfile = str(tmp_path / "MODECAR")
    write_modecar(frequency, outfile)

    with open(outfile) as f:
        lines = [line for line in f.readlines() if line.strip()]
    assert len(lines) == len(frequency), \
        f"Expected {len(frequency)} lines, got {len(lines)}"


def test_write_modecar_values_correct(tmp_path):
    """write_modecar output must contain the correct floating-point values."""
    from tools4vasp.freq2mode import write_modecar

    frequency = [[1.5, -2.5, 0.0]]
    outfile = str(tmp_path / "MODECAR")
    write_modecar(frequency, outfile)

    content = open(outfile).read()
    assert "1.5000000000E+00" in content
    assert "-2.5000000000E+00" in content


# ---------------------------------------------------------------------------
# Bug 10 — vaspGetEF: 'vervose' typo in get_all_xmls
# ---------------------------------------------------------------------------

def test_get_all_xmls_parameter_name():
    """get_all_xmls must accept 'verbose' (not 'vervose') as a keyword arg."""
    from tools4vasp.vaspGetEF import get_all_xmls
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        # Should not raise TypeError for unexpected keyword argument
        result = get_all_xmls(d, verbose=False)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Bug 11 — vaspcheck / plotNEB: shell=True command injection via file paths
# ---------------------------------------------------------------------------

OUTCAR_TOTEN_BLOCK = """\
--------------------------------------- Iteration      2(  12)  ----------

  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  ---------------------------------------------------
  free  energy   TOTEN  =       -68.41063650 eV

  energy  without entropy=      -68.40903650  energy(sigma->0) =      -68.41010317
"""


def test_get_entropy_energies_parses_outcar(tmp_path):
    """_get_entropy_energies must parse TOTEN and entropy-free energy."""
    from tools4vasp.vaspcheck import _get_entropy_energies

    outcar = tmp_path / "OUTCAR"
    outcar.write_text(OUTCAR_TOTEN_BLOCK)

    toten, e_wo_entropy = _get_entropy_energies(str(outcar))
    assert toten == pytest.approx(-68.41063650)
    assert e_wo_entropy == pytest.approx(-68.40903650)


def test_get_entropy_energies_path_with_spaces_and_metachars(tmp_path):
    """Paths with spaces/shell metacharacters must work (no shell involved).

    The old implementation interpolated the path into a `tail | grep`
    shell pipeline, which broke on spaces and allowed command injection.
    """
    from tools4vasp.vaspcheck import _get_entropy_energies

    evil_dir = tmp_path / "my calc; $(touch pwned)"
    evil_dir.mkdir()
    outcar = evil_dir / "OUTCAR"
    outcar.write_text(OUTCAR_TOTEN_BLOCK)

    toten, e_wo_entropy = _get_entropy_energies(str(outcar))
    assert toten == pytest.approx(-68.41063650)
    assert not (tmp_path / "pwned").exists()
    assert not (evil_dir / "pwned").exists()


# ---------------------------------------------------------------------------
# Issue #24 — vaspcheck: entropy check must use the FINAL TOTEN block and
# must not depend on the block being within the last 200 lines
# ---------------------------------------------------------------------------

OUTCAR_TOTEN_BLOCK_FINAL = """\
--------------------------------------- Iteration      2(  13)  ----------

  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  ---------------------------------------------------
  free  energy   TOTEN  =       -70.12345678 eV

  energy  without entropy=      -70.12245678  energy(sigma->0) =      -70.12295678
"""


def test_get_entropy_energies_takes_last_toten_block(tmp_path):
    """With multiple TOTEN entries the final one must be used (issue #24)."""
    from tools4vasp.vaspcheck import _get_entropy_energies

    outcar = tmp_path / "OUTCAR"
    outcar.write_text(OUTCAR_TOTEN_BLOCK + "\n" + OUTCAR_TOTEN_BLOCK_FINAL)

    toten, e_wo_entropy = _get_entropy_energies(str(outcar))
    assert toten == pytest.approx(-70.12345678)
    assert e_wo_entropy == pytest.approx(-70.12245678)


def test_get_entropy_energies_block_beyond_last_200_lines(tmp_path):
    """TOTEN deeper than 200 lines from the end must still parse (issue #24).

    Large systems print long property tables after the final energies, so
    the old `tail -n 200`-style window crashed on them.
    """
    from tools4vasp.vaspcheck import _get_entropy_energies

    outcar = tmp_path / "OUTCAR"
    trailing = "\n".join("  some other OUTCAR output line {}".format(i)
                         for i in range(300))
    outcar.write_text(OUTCAR_TOTEN_BLOCK + "\n" + trailing + "\n")

    toten, e_wo_entropy = _get_entropy_energies(str(outcar))
    assert toten == pytest.approx(-68.41063650)
    assert e_wo_entropy == pytest.approx(-68.40903650)


def test_get_entropy_energies_unpaired_toten_raises(tmp_path):
    """A TOTEN line with no entropy line within four lines must not match."""
    from tools4vasp.vaspcheck import _get_entropy_energies

    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "  free  energy   TOTEN  =       -68.41063650 eV\n"
        + "filler\n" * 10)

    with pytest.raises(ValueError):
        _get_entropy_energies(str(outcar))


def test_get_entropy_energies_missing_block_raises(tmp_path):
    """A clear ValueError must be raised when no TOTEN block is found."""
    from tools4vasp.vaspcheck import _get_entropy_energies

    outcar = tmp_path / "OUTCAR"
    outcar.write_text("nothing useful here\n")

    with pytest.raises(ValueError):
        _get_entropy_energies(str(outcar))


def test_plotNEB_dispersion_read_without_shell(tmp_path, monkeypatch):
    """run(plot_dispersion=True) must read Edisp from OUTCARs without a shell."""
    from tools4vasp import plotNEB

    monkeypatch.chdir(tmp_path)
    n_img = 3
    # minimal spline.dat / neb.dat: columns (idx, x, E, F)
    spline_lines = []
    for i in range(10):
        x = i / 9.0
        spline_lines.append("{} {} {} {}".format(i, x, 0.0, 0.0))
    (tmp_path / "spline.dat").write_text("\n".join(spline_lines) + "\n")
    neb_lines = []
    for i in range(n_img):
        neb_lines.append("{} {} {} {}".format(i, i / 2.0, 0.1 * i, 0.0))
    (tmp_path / "neb.dat").write_text("\n".join(neb_lines) + "\n")
    for i in range(n_img):
        d = tmp_path / "{:02d}".format(i)
        d.mkdir()
        (d / "OUTCAR").write_text(
            "  Edisp (eV) =   -0.10000\n"
            "  Edisp (eV) =   -{:.5f}\n".format(0.2 + 0.1 * i)
        )

    plotNEB.run(filename=str(tmp_path / "NEB.png"), plot_dispersion=True)
    assert (tmp_path / "NEB.png").exists()


def test_plotNEB_module_does_not_use_subprocess():
    """plotNEB must not shell out at all (grep/tail dependency removed).

    Inspects the AST rather than the raw source so that mere mentions of
    "subprocess" in comments/docstrings (or other harmless refactors)
    don't trip the test — only real imports or shell=... call keywords do.
    """
    import ast
    import tools4vasp.plotNEB as mod

    tree = ast.parse(inspect.getsource(mod))
    imports = [
        name.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for name in node.names
    ] + [
        node.module
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any(name == "subprocess" or name.startswith("subprocess.")
                   for name in imports if name)

    shell_keywords = [
        kw
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        for kw in node.keywords if kw.arg == "shell"
    ]
    assert not shell_keywords
