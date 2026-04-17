"""
Extended coverage tests — exercises previously uncovered code paths using
mocking and lightweight fixtures. No real VASP installation needed.
"""
import os
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ===========================================================================
# vaspcheck
# ===========================================================================

_OUTCAR_MATCH = """\
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
POSCAR: Au N
"""

_OUTCAR_MISMATCH = """\
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
POSCAR: N Au
"""

_OUTCAR_SINGLE_POTCAR = """\
POTCAR: PAW_PBE Au 08Apr2002
POSCAR: Au
"""


def test_check_vasp_potcar_order_match(tmp_path):
    """check_vasp_potcar_order() must return None when order matches."""
    from tools4vasp.vaspcheck import check_vasp_potcar_order
    (tmp_path / "OUTCAR").write_text(_OUTCAR_MATCH)
    assert check_vasp_potcar_order(str(tmp_path)) is None


def test_check_vasp_potcar_order_mismatch(tmp_path):
    """check_vasp_potcar_order() must return a message when order mismatches."""
    from tools4vasp.vaspcheck import check_vasp_potcar_order
    (tmp_path / "OUTCAR").write_text(_OUTCAR_MISMATCH)
    result = check_vasp_potcar_order(str(tmp_path))
    assert result is not None
    assert "POTCAR" in result or "POSCAR" in result


def test_check_vasp_potcar_order_not_dir(tmp_path):
    """check_vasp_potcar_order() must raise AssertionError for non-directory."""
    from tools4vasp.vaspcheck import check_vasp_potcar_order
    fake = str(tmp_path / "not_a_dir")
    with pytest.raises(AssertionError):
        check_vasp_potcar_order(fake)


def test_get_elements_single_potcar_line():
    """_get_elements_from_outcar handles the single-POTCAR-species branch."""
    from tools4vasp.vaspcheck import _get_elements_from_outcar
    poscar, potcar = _get_elements_from_outcar(io.StringIO(_OUTCAR_SINGLE_POTCAR))
    assert poscar == ["Au"]
    assert potcar == ["Au"]


def _make_mock_calc(spin_polarized=False, occ_val=2.0):
    """Build a mock ASE Vasp calc whose _read_xml() behaves predictably."""
    xml = MagicMock()
    xml.get_spin_polarized.return_value = spin_polarized
    xml.get_ibz_k_points.return_value = [None]  # 1 k-point
    xml.get_occupation_numbers.return_value = np.array([occ_val, 0.0])
    calc = MagicMock()
    calc._read_xml.return_value = xml
    return calc


def test_check_vasp_occupations_integer():
    """check_vasp_occupations() returns None for fully integer occupations."""
    from tools4vasp.vaspcheck import check_vasp_occupations
    assert check_vasp_occupations(_make_mock_calc(occ_val=2.0)) is None


def test_check_vasp_occupations_non_integer():
    """check_vasp_occupations() returns a message for fractional occupations."""
    from tools4vasp.vaspcheck import check_vasp_occupations
    result = check_vasp_occupations(_make_mock_calc(occ_val=1.5))
    assert result is not None


def test_check_vasp_occupations_spin_polarized_integer():
    """check_vasp_occupations() returns None for spin-pol integer occs."""
    from tools4vasp.vaspcheck import check_vasp_occupations
    assert check_vasp_occupations(_make_mock_calc(spin_polarized=True, occ_val=1.0)) is None


def test_check_vasp_occupations_none_occ():
    """check_vasp_occupations() returns a message when occupation is None."""
    from tools4vasp.vaspcheck import check_vasp_occupations
    calc = _make_mock_calc()
    calc._read_xml().get_occupation_numbers.return_value = None
    result = check_vasp_occupations(calc)
    assert result is not None


def test_check_vasp_electronic_entropy_good_occupations():
    """check_vasp_electronic_entropy() returns None when occupations are integer."""
    from tools4vasp.vaspcheck import check_vasp_electronic_entropy
    # check_vasp_occupations returns None → entropy branch not reached
    result = check_vasp_electronic_entropy(".", _make_mock_calc(occ_val=2.0))
    assert result is None


def test_vaspcheck_main_runs(tmp_path):
    """main() must complete without exception when calc reports no problems."""
    from tools4vasp import vaspcheck
    mock_calc = _make_mock_calc(occ_val=2.0)
    mock_calc.read_convergence.return_value = True
    with patch("tools4vasp.vaspcheck.Vasp", return_value=mock_calc), \
         patch("tools4vasp.vaspcheck.check_vasp_electronic_entropy", return_value=None):
        vaspcheck.run(str(tmp_path))


def test_vaspcheck_main_convergence_failed(tmp_path):
    """main() must print a message when convergence check fails."""
    from tools4vasp import vaspcheck
    mock_calc = _make_mock_calc(occ_val=2.0)
    mock_calc.read_convergence.return_value = False
    with patch("tools4vasp.vaspcheck.Vasp", return_value=mock_calc), \
         patch("tools4vasp.vaspcheck.check_vasp_electronic_entropy", return_value=None), \
         patch("builtins.print") as mock_print:
        vaspcheck.run(str(tmp_path))
    printed = " ".join(str(c) for c in mock_print.call_args_list)
    assert "converge" in printed.lower() or "SCF" in printed


def test_vaspcheck_main_entropy_problem(tmp_path):
    """main() must print the entropy message and return early when reported."""
    from tools4vasp import vaspcheck
    mock_calc = _make_mock_calc()
    with patch("tools4vasp.vaspcheck.Vasp", return_value=mock_calc), \
         patch("tools4vasp.vaspcheck.check_vasp_electronic_entropy", return_value="Entropy per atom is 0.5eV"), \
         patch("builtins.print") as mock_print:
        vaspcheck.run(str(tmp_path))
    printed = " ".join(str(c) for c in mock_print.call_args_list)
    assert "Entropy" in printed or "entropy" in printed.lower()


# ===========================================================================
# vaspGetEF
# ===========================================================================

def _make_mock_atoms_traj(energies, forces_list):
    traj = []
    for e, f in zip(energies, forces_list):
        atoms = MagicMock()
        atoms.get_potential_energy.return_value = e
        atoms.get_forces.return_value = np.array(f)
        traj.append(atoms)
    return traj


def test_read_xml_file_not_found():
    """read_xml() must raise FileNotFoundError for a missing file."""
    from tools4vasp.vaspGetEF import read_xml
    with pytest.raises(FileNotFoundError):
        read_xml("/nonexistent/vasprun.xml")


def test_read_xml_returns_forces_and_energies(tmp_path):
    """read_xml() must return (forces, energies) lists of the same length."""
    from tools4vasp.vaspGetEF import read_xml
    xml_path = tmp_path / "vasprun.xml"
    xml_path.write_text("dummy")

    traj = _make_mock_atoms_traj(
        energies=[-10.0, -10.5, -11.0],
        forces_list=[[[1.0, 0, 0]], [[0, 1.0, 0]], [[0, 0, 1.0]]],
    )
    with patch("tools4vasp.vaspGetEF.read_vasp_xml", return_value=iter(traj)):
        forces, energies = read_xml(str(xml_path))

    assert len(forces) == 3
    assert len(energies) == 3
    assert energies == pytest.approx([-10.0, -10.5, -11.0])


def test_read_xml_verbose_output(tmp_path, capsys):
    """read_xml() must print values when verbose=True."""
    from tools4vasp.vaspGetEF import read_xml
    xml_path = tmp_path / "vasprun.xml"
    xml_path.write_text("dummy")
    traj = _make_mock_atoms_traj([-5.0], [[[1.0, 2.0, 0.0]]])
    with patch("tools4vasp.vaspGetEF.read_vasp_xml", return_value=iter(traj)):
        read_xml(str(xml_path), verbose=True)
    out = capsys.readouterr().out
    assert len(out) > 0


def test_read_xml_write_fe(tmp_path):
    """read_xml() must write fe.dat when write_fe=True."""
    from tools4vasp.vaspGetEF import read_xml
    xml_path = tmp_path / "vasprun.xml"
    xml_path.write_text("dummy")
    traj = _make_mock_atoms_traj([-5.0, -6.0], [[[1.0, 0, 0]], [[0, 1.0, 0]]])
    with patch("tools4vasp.vaspGetEF.read_vasp_xml", return_value=iter(traj)):
        read_xml(str(xml_path), write_fe=True)
    assert (tmp_path / "fe.dat").is_file()


def test_get_all_xmls_finds_parent_xml(tmp_path):
    """get_all_xmls() must include a vasprun.xml in the parent directory."""
    from tools4vasp.vaspGetEF import get_all_xmls
    (tmp_path / "vasprun.xml").write_text("dummy")
    result = get_all_xmls(str(tmp_path))
    assert any("vasprun.xml" in r for r in result)


def test_get_all_xmls_finds_subdir_xml(tmp_path):
    """get_all_xmls() must include vasprun.xml files in numeric subdirectories."""
    from tools4vasp.vaspGetEF import get_all_xmls
    subdir = tmp_path / "01"
    subdir.mkdir()
    (subdir / "vasprun.xml").write_text("dummy")
    result = get_all_xmls(str(tmp_path))
    assert len(result) >= 1


def test_get_all_xmls_empty_dir(tmp_path):
    """get_all_xmls() must return an empty list when no xml files exist."""
    from tools4vasp.vaspGetEF import get_all_xmls
    result = get_all_xmls(str(tmp_path))
    assert result == []


def test_plot_fe_creates_file(tmp_path):
    """plot_fe() must save the figure to the given filename."""
    from tools4vasp.vaspGetEF import plot_fe
    combined = {"force": [0.5, 0.3, 0.1], "energy": [-10.0, -10.5, -11.0]}
    outfile = str(tmp_path / "fe.png")
    plot_fe(combined, outfile)
    assert os.path.isfile(outfile)


def test_vaspGetEF_main_callable():
    """vaspGetEF must expose a callable main() for the console_scripts entry point."""
    from tools4vasp import vaspGetEF
    assert callable(vaspGetEF.main)


def test_vaspGetEF_main_runs(tmp_path):
    """vaspGetEF.main() must run without error when mocked."""
    from tools4vasp import vaspGetEF
    with patch("tools4vasp.vaspGetEF.process_all_xmls", return_value={"force": [0.1], "energy": [-5.0]}), \
         patch("tools4vasp.vaspGetEF.plot_fe"), \
         patch("sys.argv", ["vaspGetEF", str(tmp_path)]):
        vaspGetEF.main()


# ===========================================================================
# add_MODECAR
# ===========================================================================

_POSCAR_H = """\
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


def test_add_modecar_main(tmp_path, monkeypatch):
    """add_MODECAR.main() must read POSCAR+MODECAR and write an xyz file."""
    import ase
    from tools4vasp import add_MODECAR

    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_H)
    modecar = tmp_path / "MODECAR"
    modecar.write_text("    1.0000000000E+00    0.0000000000E+00    0.0000000000E+00\n")
    monkeypatch.chdir(tmp_path)

    fake_atoms = ase.Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10])

    written = []

    def fake_write(path, append=False):
        written.append(path)

    fake_atoms.write = fake_write

    with patch("tools4vasp.add_MODECAR.io.read", return_value=fake_atoms), \
         patch("tools4vasp.add_MODECAR.np.loadtxt", return_value=np.array([[1.0, 0.0, 0.0]])), \
         patch("sys.argv", ["add-MODECAR"]):
        add_MODECAR.main()

    assert len(written) == 2


# ===========================================================================
# vasp2traj
# ===========================================================================

def test_vasp2traj_raises_for_missing_input(tmp_path):
    """main() must raise ValueError when an input file does not exist."""
    from tools4vasp.vasp2traj import run as main
    outfile = str(tmp_path / "out.xyz")
    with pytest.raises(ValueError, match="does not exist"):
        main(outfile, [str(tmp_path / "nonexistent.xml")], wrap=False)


def test_vasp2traj_backs_up_existing_output(tmp_path):
    """main() must rename existing output file to *.bak."""
    from tools4vasp.vasp2traj import run as main

    outfile = tmp_path / "traj.xyz"
    outfile.write_text("old content")
    infile = tmp_path / "OUTCAR"
    infile.write_text("dummy")

    fake_frame = MagicMock()
    with patch("tools4vasp.vasp2traj.io.read", return_value=[fake_frame]):
        main(str(outfile), [str(infile)], wrap=False)

    assert (tmp_path / "traj.xyz.bak").is_file()


def test_vasp2traj_xdatcar_format(tmp_path):
    """main() must use format='vasp-xdatcar' for XDATCAR files."""
    from tools4vasp.vasp2traj import run as main

    infile = tmp_path / "XDATCAR"
    infile.write_text("dummy")

    fake_frame = MagicMock()
    with patch("tools4vasp.vasp2traj.io.read", return_value=[fake_frame]) as mock_read:
        main(str(tmp_path / "out.xyz"), [str(infile)], wrap=False)

    _, kwargs = mock_read.call_args
    assert kwargs.get("format") == "vasp-xdatcar"


def test_vasp2traj_vasp_out_format(tmp_path):
    """main() must use format='vasp-out' for non-XDATCAR files."""
    from tools4vasp.vasp2traj import run as main

    infile = tmp_path / "OUTCAR"
    infile.write_text("dummy")

    fake_frame = MagicMock()
    with patch("tools4vasp.vasp2traj.io.read", return_value=[fake_frame]) as mock_read:
        main(str(tmp_path / "out.xyz"), [str(infile)], wrap=False)

    _, kwargs = mock_read.call_args
    assert kwargs.get("format") == "vasp-out"


def test_vasp2traj_wrap_calls_frame_wrap(tmp_path):
    """main() must call frame.wrap() when wrap=True."""
    from tools4vasp.vasp2traj import run as main

    infile = tmp_path / "OUTCAR"
    infile.write_text("dummy")

    fake_frame = MagicMock()
    with patch("tools4vasp.vasp2traj.io.read", return_value=[fake_frame]):
        main(str(tmp_path / "out.xyz"), [str(infile)], wrap=True)

    fake_frame.wrap.assert_called_once()


# ===========================================================================
# freq2jmol
# ===========================================================================

def test_freq2jmol_main(tmp_path, monkeypatch):
    """freq2jmol.main() must call Vasp.get_vibrations() and write_jmol()."""
    from tools4vasp import freq2jmol
    monkeypatch.chdir(tmp_path)

    mock_frame = MagicMock()
    mock_vib_data = MagicMock()
    mock_vib_data.iter_animated_mode.return_value = [mock_frame]

    mock_vibs = MagicMock()
    mock_vibs.get_energies.return_value = [0.1 + 0.0j]
    mock_vibs.iter_animated_mode.return_value = [mock_frame]

    mock_calc = MagicMock()
    mock_calc.get_vibrations.return_value = mock_vibs

    with patch("tools4vasp.freq2jmol.Vasp", return_value=mock_calc), \
         patch("sys.argv", ["freq2jmol"]):
        freq2jmol.main()

    mock_vibs.write_jmol.assert_called_once()


# ===========================================================================
# poscar2nbands
# ===========================================================================

def test_poscar2nbands_main_callable():
    """poscar2nbands must expose a callable main()."""
    from tools4vasp import poscar2nbands
    assert callable(poscar2nbands.main)


def test_poscar2nbands_get_nbands(tmp_path, monkeypatch):
    """get_nbands() must call Lobsterin and print the result."""
    from tools4vasp import poscar2nbands
    monkeypatch.chdir(tmp_path)

    mock_lobsterin = MagicMock()
    mock_lobsterin._get_nbands.return_value = 42

    mock_structure = MagicMock()

    # Lobsterin/Structure are imported inside get_nbands(), patch via pymatgen paths
    with patch("pymatgen.io.lobster.Lobsterin") as MockLobsterin, \
         patch("pymatgen.core.structure.Structure") as MockStructure, \
         patch("builtins.print") as mock_print:
        MockLobsterin.standard_calculations_from_vasp_files.return_value = mock_lobsterin
        MockStructure.from_file.return_value = mock_structure
        poscar2nbands.get_nbands()

    mock_print.assert_called_once_with(42)


# ===========================================================================
# Bash-wrapper modules (calc_deformation_density, replace_potcar_symlinks,
# plot_neb_movie, visualize_magnitization)
# ===========================================================================

@pytest.mark.parametrize("module_name,script_fragment", [
    ("tools4vasp.calc_deformation_density", "calc-deformation-density.sh"),
    ("tools4vasp.replace_potcar_symlinks", "replace_potcar_symlinks.sh"),
    ("tools4vasp.plot_neb_movie", "plot-neb-movie.sh"),
    ("tools4vasp.visualize_magnitization", "visualize-magnetization.sh"),
])
def test_bash_wrapper_main_calls_correct_script(module_name, script_fragment):
    """Each bash-wrapper main() must invoke subprocess.run with the right script."""
    import importlib
    mod = importlib.import_module(module_name)
    with patch("subprocess.run") as mock_run:
        mod.main()
    assert mock_run.called
    called_script = mock_run.call_args[0][0][0]
    assert script_fragment in called_script


# ===========================================================================
# plotNEB — main() + highlight/dispersion paths
# ===========================================================================

def _make_neb_loadtxt(n_spline=20, n_images=5):
    """Return a side_effect for np.loadtxt matching plotNEB.run() calls."""
    spline = np.column_stack([
        np.arange(n_spline),
        np.linspace(0, 1, n_spline),
        np.sin(np.linspace(0, np.pi, n_spline)),
    ])
    neb = np.column_stack([
        np.arange(n_images),
        np.linspace(0, 1, n_images),
        np.sin(np.linspace(0, np.pi, n_images)),
        np.zeros(n_images),
    ])
    return [spline, neb]


def test_plotNEB_main_basic(tmp_path, monkeypatch):
    """plotNEB.run() must complete without error for a basic NEB dataset."""
    from tools4vasp import plotNEB
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spline.dat").write_text("dummy")
    (tmp_path / "neb.dat").write_text("dummy")

    loadtxt_returns = _make_neb_loadtxt()
    with patch("tools4vasp.plotNEB.np.loadtxt", side_effect=loadtxt_returns), \
         patch("tools4vasp.plotNEB.plt"):
        plotNEB.run(filename=str(tmp_path / "out.png"))


def test_plotNEB_main_unit_kj_mol(tmp_path, monkeypatch):
    """plotNEB.run() must handle kJ/mol unit conversion."""
    from tools4vasp import plotNEB
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spline.dat").write_text("dummy")
    (tmp_path / "neb.dat").write_text("dummy")

    loadtxt_returns = _make_neb_loadtxt()
    with patch("tools4vasp.plotNEB.np.loadtxt", side_effect=loadtxt_returns), \
         patch("tools4vasp.plotNEB.plt"):
        plotNEB.run(filename=str(tmp_path / "out.png"), unit="kJ/mol")


def test_plotNEB_main_presentation_mode(tmp_path, monkeypatch):
    """plotNEB.run() must not raise in presentation mode."""
    from tools4vasp import plotNEB
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spline.dat").write_text("dummy")
    (tmp_path / "neb.dat").write_text("dummy")

    loadtxt_returns = _make_neb_loadtxt()
    with patch("tools4vasp.plotNEB.np.loadtxt", side_effect=loadtxt_returns), \
         patch("tools4vasp.plotNEB.plt"):
        plotNEB.run(filename=str(tmp_path / "out.png"), presentation=True)


def test_plotNEB_main_plot_all(tmp_path, monkeypatch):
    """plotNEB.run() with plot_all=True must create per-image plots."""
    from tools4vasp import plotNEB
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spline.dat").write_text("dummy")
    (tmp_path / "neb.dat").write_text("dummy")

    loadtxt_returns = _make_neb_loadtxt(n_images=3)
    with patch("tools4vasp.plotNEB.np.loadtxt", side_effect=loadtxt_returns), \
         patch("tools4vasp.plotNEB.plt"):
        plotNEB.run(filename=str(tmp_path / "out.png"), plot_all=True)


def test_plotNEB_plot_with_highlight(tmp_path, neb_data):
    """plot() must not raise when highlight index is supplied."""
    from tools4vasp.plotNEB import plot
    spline_x, image_x, energies, spline_e, forces = neb_data
    out = str(tmp_path / "neb_highlight.png")
    plot(spline_x, image_x, energies, spline_e, forces, out, highlight=2)


def test_plotNEB_plot_with_dispersion(tmp_path, neb_data):
    """plot() must not raise when dispersion array is supplied."""
    from tools4vasp.plotNEB import plot
    spline_x, image_x, energies, spline_e, forces = neb_data
    dispersion = np.zeros(len(image_x))
    out = str(tmp_path / "neb_disp.png")
    plot(spline_x, image_x, energies, spline_e, forces, out, dispersion=dispersion)


def test_plotNEB_main_load_dispersion(tmp_path, monkeypatch):
    """plotNEB.run() must load dispersion from json when load_dispersion is set."""
    import json
    from tools4vasp import plotNEB
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spline.dat").write_text("dummy")
    (tmp_path / "neb.dat").write_text("dummy")
    disp_file = tmp_path / "disp.json"
    n_images = 5
    disp_file.write_text(json.dumps([0.0] * n_images))

    loadtxt_returns = _make_neb_loadtxt(n_images=n_images)
    with patch("tools4vasp.plotNEB.np.loadtxt", side_effect=loadtxt_returns), \
         patch("tools4vasp.plotNEB.plt"):
        plotNEB.run(
            filename=str(tmp_path / "out.png"),
            plot_dispersion=True,
            load_dispersion=str(disp_file),
        )


# ===========================================================================
# plotIRC — read_sp, read_irc, generate_plot with ts, plot_irc
# ===========================================================================

def _make_sp_atoms(energy=-100.0, n_atoms=2):
    atoms = MagicMock()
    atoms.calc.results = {"energy": energy}
    atoms.positions = np.zeros((n_atoms, 3))
    return atoms


def test_plotIRC_read_sp_single_point(tmp_path):
    """read_sp() must return (positions, energy) for a single-point vasprun.xml."""
    from tools4vasp.plotIRC import read_sp
    (tmp_path / "vasprun.xml").write_text("dummy")
    sp_atoms = _make_sp_atoms(energy=-100.0)
    with patch("tools4vasp.plotIRC.ase.io.read", return_value=[sp_atoms]):
        positions, energy = read_sp(str(tmp_path), allow_freq=False)
    assert positions.shape == (2, 3)
    assert energy != 0


def test_plotIRC_read_sp_no_vasprun_exits(tmp_path):
    """read_sp() must call sys.exit when vasprun.xml is missing."""
    from tools4vasp.plotIRC import read_sp
    with pytest.raises(SystemExit):
        read_sp(str(tmp_path), allow_freq=False)


def test_plotIRC_read_sp_allow_freq_multi_step(tmp_path):
    """read_sp() with allow_freq=True must not exit for multi-step xml."""
    from tools4vasp.plotIRC import read_sp
    (tmp_path / "vasprun.xml").write_text("dummy")
    sp1 = _make_sp_atoms(-100.0)
    sp2 = _make_sp_atoms(-99.0)
    with patch("tools4vasp.plotIRC.ase.io.read", return_value=[sp1, sp2]):
        positions, energy = read_sp(str(tmp_path), allow_freq=True)
    assert energy != 0


def test_plotIRC_read_irc(tmp_path):
    """read_irc() must return (init_positions, steps, energies) from OUTCAR."""
    from tools4vasp.plotIRC import read_irc

    outcar_content = (
        "  IRC (A):   0.0000  Step\n"
        "  IRC (A):   0.1000  Step\n"
        "  IRC (A):   0.2000  Step\n"
    )
    (tmp_path / "OUTCAR").write_text(outcar_content)

    n_atoms = 2
    fake_structs = []
    for e in [-100.0, -99.5, -99.0]:
        a = MagicMock()
        a.positions = np.zeros((n_atoms, 3))
        a.calc.results = {"energy": e}
        fake_structs.append(a)

    with patch("tools4vasp.plotIRC.ase.io.read", return_value=fake_structs):
        init_pos, steps, energies = read_irc(str(tmp_path))

    assert len(steps) == 3
    assert len(energies) == 3


def test_plotIRC_generate_plot_with_ts():
    """generate_plot() with a real ts must use ts_exists=True path."""
    from tools4vasp.plotIRC import generate_plot

    pos = np.zeros((3, 3))
    ts_pos = np.zeros((3, 3))
    ts_energy = -100.0
    irc_e = [pos.copy(), [0.0, 0.1, 0.2], [-100.0, -99.5, -98.0]]
    irc_p = [pos.copy(), [0.0, 0.1, 0.2], [-100.0, -99.5, -98.0]]
    ts = [ts_pos, ts_energy]

    result = generate_plot(irc_e, irc_p, ts, offset=None)
    assert len(result) == 6


def test_plotIRC_generate_plot_ts_exists_s_plot_has_three_points():
    """generate_plot() with ts must produce s_plot_x with 3 points."""
    from tools4vasp.plotIRC import generate_plot

    pos = np.zeros((2, 3))
    irc_e = [pos.copy(), [0.0, 0.2], [-100.0, -99.0]]
    irc_p = [pos.copy(), [0.0, 0.2], [-100.0, -99.0]]
    ts = [pos.copy(), -101.0]

    result = generate_plot(irc_e, irc_p, ts, offset=None)
    s_plot_x = result[2]
    assert len(s_plot_x) == 3  # [reactant_first, 0, product_first]


def test_plotIRC_plot_irc_runs(tmp_path):
    """plot_irc() must save a plot without raising."""
    from tools4vasp.plotIRC import plot_irc

    irc_data = (
        [0.0, 0.1, 0.2],          # irc_e x
        [-100.0, -99.5, -99.0],   # irc_e y
        [-0.1, 0.0, 0.1],         # s_plot_x
        [-100.0, -101.0, -100.0], # s_plot_y
        [0.0, 0.1, 0.2],          # irc_p x
        [-100.0, -99.5, -99.0],   # irc_p y
    )
    # plot_irc uses mpl.use("pgf") and plt.rc(usetex=True) which needs LaTeX;
    # mock the entire plt and mpl interfaces to stay self-contained.
    with patch("tools4vasp.plotIRC.mpl"), \
         patch("tools4vasp.plotIRC.plt") as mock_plt:
        mock_plt.figure.return_value = MagicMock()
        plot_irc(irc_data, silent=True)
    mock_plt.savefig.assert_called_once_with("plot.svg")
    mock_plt.show.assert_not_called()  # silent=True


# ===========================================================================
# split_vasp_freq — load_vibrations, export_jmol, export_xyz_traj
# ===========================================================================

def test_load_vibrations_returns_vibrations_object(tmp_path):
    """load_vibrations() must return an ASE Vibrations object."""
    from tools4vasp.split_vasp_freq import load_vibrations
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_H)
    incar = tmp_path / "INCAR"
    incar.write_text("NFREE = 2\nPOTIM = 0.015\n")

    from ase.vibrations import Vibrations
    result = load_vibrations(str(poscar), cwd=str(tmp_path), verbose=False)
    assert isinstance(result, Vibrations)


def test_export_jmol_calls_write_jmol(tmp_path):
    """export_jmol() must call vib._write_jmol() with an open file handle."""
    from tools4vasp.split_vasp_freq import export_jmol

    mock_vib = MagicMock()
    outfile = str(tmp_path / "modes")
    export_jmol(mock_vib, outfile)

    mock_vib._write_jmol.assert_called_once()
    assert os.path.isfile(outfile + ".xyz")


def test_export_xyz_traj_single_mode(tmp_path):
    """export_xyz_traj() with an integer index must write exactly one file."""
    from tools4vasp.split_vasp_freq import export_xyz_traj

    mock_frame = MagicMock()
    mock_vib_data = MagicMock()
    mock_vib_data.iter_animated_mode.return_value = [mock_frame]

    mock_vib = MagicMock()
    mock_vib.get_vibrations.return_value = mock_vib_data

    outfile = str(tmp_path / "vib")
    export_xyz_traj(mock_vib, outfile, index=0)

    assert os.path.isfile(outfile + "_001.xyz")


def test_export_xyz_traj_all_modes(tmp_path):
    """export_xyz_traj() with index=None must write one file per mode."""
    from tools4vasp.split_vasp_freq import export_xyz_traj

    n_modes = 3
    mock_frame = MagicMock()
    mock_vib_data = MagicMock()
    mock_vib_data.iter_animated_mode.return_value = [mock_frame]

    mock_vib = MagicMock()
    mock_vib.get_vibrations.return_value = mock_vib_data
    mock_vib.get_energies.return_value = [0.1] * n_modes

    outfile = str(tmp_path / "vib")
    export_xyz_traj(mock_vib, outfile, index=None)

    for i in range(1, n_modes + 1):
        assert os.path.isfile("{:s}_{:03d}.xyz".format(outfile, i))


# ===========================================================================
# chgcar2cube — additional paths
# ===========================================================================

def test_chgcar2cube_mult_volume(tmp_path):
    """chgcar2cube() with mult_volume=True must divide by volume."""
    from tools4vasp.chgcar2cube import chgcar2cube

    infile = tmp_path / "CHGCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "out")

    mock_chgcar = MagicMock()
    mock_chgcar.data = {"total": np.ones((2, 2, 2))}
    mock_atoms = MagicMock()
    mock_atoms.get_volume.return_value = 1000.0

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.chgcar2cube.write_cube") as mock_write:
        chgcar2cube([str(infile)], [outbase], verbose=False, mult_volume=True)

    assert mock_write.called


def test_chgcar2cube_return_spin_integrals(tmp_path):
    """chgcar2cube() with return_spin_integrals=True must return both integrals."""
    from tools4vasp.chgcar2cube import chgcar2cube

    infile = tmp_path / "CHGCAR"
    infile.write_text("dummy")
    outbase = str(tmp_path / "out")

    mock_chgcar = MagicMock()
    mock_chgcar.data = {
        "total": np.ones((2, 2, 2)),
        "diff": np.ones((2, 2, 2)) * 0.5,
    }
    mock_atoms = MagicMock()

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.chgcar2cube.write_cube"):
        total, spin = chgcar2cube(
            [str(infile)], [outbase], verbose=False,
            return_integrals=True, return_spin_integrals=True,
        )

    assert total > 0
    assert spin > 0


def test_chgcar2cube_raises_for_non_spinpol_spin_integral(tmp_path):
    """chgcar2cube() must raise ValueError when requesting spin integral for non-spinpol."""
    from tools4vasp.chgcar2cube import chgcar2cube

    infile = tmp_path / "CHGCAR"
    infile.write_text("dummy")

    mock_chgcar = MagicMock()
    mock_chgcar.data = {"total": np.ones((2, 2, 2))}  # no 'diff' key
    mock_atoms = MagicMock()

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms):
        with pytest.raises(ValueError):
            chgcar2cube([str(infile)], [str(tmp_path / "out")], verbose=False, return_spin_integrals=True)


def test_chgcar2cube_backs_up_existing_output(tmp_path):
    """chgcar2cube() must move existing outbase file to *.bak."""
    from tools4vasp.chgcar2cube import chgcar2cube

    infile = tmp_path / "CHGCAR"
    infile.write_text("dummy")
    # The backup check is on outFiles[iFile] (no extension), not on *.cube
    existing = tmp_path / "out"
    existing.write_text("old")

    mock_chgcar = MagicMock()
    mock_chgcar.data = {"total": np.ones((2, 2, 2))}
    mock_atoms = MagicMock()

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.chgcar2cube.write_cube"):
        chgcar2cube([str(infile)], [str(tmp_path / "out")], verbose=True)

    assert (tmp_path / "out.bak").is_file()


def test_chgcar2cube_returns_list_for_multiple_files(tmp_path):
    """chgcar2cube() must return a list of integrals for multiple input files."""
    from tools4vasp.chgcar2cube import chgcar2cube

    infiles = []
    outbases = []
    for i in range(2):
        f = tmp_path / f"CHGCAR_{i}"
        f.write_text("dummy")
        infiles.append(str(f))
        outbases.append(str(tmp_path / f"out_{i}"))

    mock_chgcar = MagicMock()
    mock_chgcar.data = {"total": np.ones((2, 2, 2))}
    mock_atoms = MagicMock()

    with patch("tools4vasp.chgcar2cube.Chgcar.from_file", return_value=mock_chgcar), \
         patch("tools4vasp.chgcar2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.chgcar2cube.write_cube"):
        result = chgcar2cube(infiles, outbases, verbose=False, return_integrals=True)

    assert isinstance(result, list)
    assert len(result) == 2


# ===========================================================================
# elf2cube — additional paths
# ===========================================================================

def test_elf2cube_return_spin_integrals(tmp_path):
    """elf2cube() with return_spin_integrals=True must return both integrals."""
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
         patch("tools4vasp.elf2cube.write_cube"):
        total_integ, spin_integ = elf2cube(
            [str(infile)], [outbase], verbose=False,
            return_integrals=True, return_spin_integrals=True,
        )

    assert total_integ is not None
    assert spin_integ is not None


def test_elf2cube_raises_for_non_spinpol_spin_integral(tmp_path):
    """elf2cube() must raise ValueError when requesting spin integral for non-spinpol."""
    from tools4vasp.elf2cube import elf2cube

    infile = tmp_path / "ELFCAR"
    infile.write_text("dummy")

    mock_elfcar = MagicMock()
    mock_elfcar.data = {"total": np.ones((2, 2, 2))}
    mock_atoms = MagicMock()

    with patch("tools4vasp.elf2cube.Elfcar.from_file", return_value=mock_elfcar), \
         patch("tools4vasp.elf2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms):
        with pytest.raises(ValueError):
            elf2cube([str(infile)], [str(tmp_path / "out")], verbose=False, return_spin_integrals=True)


def test_elf2cube_verbose_spinpol_output(tmp_path, capsys):
    """elf2cube() with verbose=True must print spin integrals for spinpol input."""
    from tools4vasp.elf2cube import elf2cube

    infile = tmp_path / "ELFCAR"
    infile.write_text("dummy")

    mock_elfcar = MagicMock()
    mock_elfcar.data = {
        "total": np.ones((2, 2, 2)) * 0.6,
        "diff":  np.ones((2, 2, 2)) * 0.4,
    }
    mock_atoms = MagicMock()

    with patch("tools4vasp.elf2cube.Elfcar.from_file", return_value=mock_elfcar), \
         patch("tools4vasp.elf2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.elf2cube.write_cube"):
        elf2cube([str(infile)], [str(tmp_path / "out")], verbose=True)

    out = capsys.readouterr().out
    assert "up" in out.lower() or "down" in out.lower() or "spin" in out.lower()


def test_elf2cube_backs_up_existing_output(tmp_path):
    """elf2cube() must rename existing outbase file to *.bak."""
    from tools4vasp.elf2cube import elf2cube

    infile = tmp_path / "ELFCAR"
    infile.write_text("dummy")
    # Backup check is on outFiles[iFile] (no extension)
    existing = tmp_path / "out"
    existing.write_text("old")

    mock_elfcar = MagicMock()
    mock_elfcar.data = {"total": np.ones((2, 2, 2))}
    mock_atoms = MagicMock()

    with patch("tools4vasp.elf2cube.Elfcar.from_file", return_value=mock_elfcar), \
         patch("tools4vasp.elf2cube.AseAtomsAdaptor.get_atoms", return_value=mock_atoms), \
         patch("tools4vasp.elf2cube.write_cube"):
        elf2cube([str(infile)], [str(tmp_path / "out")], verbose=True)

    assert (tmp_path / "out.bak").is_file()


# ===========================================================================
# neb2movie — explicit `use` parameter paths
# ===========================================================================

_POSCAR_NEB = """\
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


def test_neb2movie_explicit_use_poscar(tmp_path):
    """main() with use='POSCAR' must read POSCAR for all images."""
    from tools4vasp import neb2movie

    for d in ["00", "01", "02"]:
        (tmp_path / d).mkdir()
        (tmp_path / d / "POSCAR").write_text(_POSCAR_NEB)
    # Also add CONTCAR to middle image — must be ignored when use='POSCAR'
    (tmp_path / "01" / "CONTCAR").write_text(_POSCAR_NEB)

    fake_atom = MagicMock()
    with patch("tools4vasp.neb2movie.io.read", return_value=fake_atom) as mock_read:
        neb2movie.run(outFile=str(tmp_path / "movie.xyz"), workdir=str(tmp_path), use="POSCAR")

    paths = [str(c.args[0]) for c in mock_read.call_args_list]
    assert not any("CONTCAR" in p for p in paths)
    assert all("POSCAR" in p for p in paths)


def test_neb2movie_explicit_use_contcar(tmp_path):
    """main() with use='CONTCAR' must read CONTCAR for middle images."""
    from tools4vasp import neb2movie

    for d in ["00", "01", "02"]:
        (tmp_path / d).mkdir()
        (tmp_path / d / "POSCAR").write_text(_POSCAR_NEB)
        (tmp_path / d / "CONTCAR").write_text(_POSCAR_NEB)

    fake_atom = MagicMock()
    with patch("tools4vasp.neb2movie.io.read", return_value=fake_atom) as mock_read:
        neb2movie.run(outFile=str(tmp_path / "movie.xyz"), workdir=str(tmp_path), use="CONTCAR")

    paths = [str(c.args[0]) for c in mock_read.call_args_list]
    assert any("CONTCAR" in p for p in paths)


def test_neb2movie_backs_up_existing_outfile(tmp_path):
    """main() must rename an existing output file to *.bak."""
    from tools4vasp import neb2movie

    outfile = tmp_path / "movie.xyz"
    outfile.write_text("old")

    for d in ["00", "01", "02"]:
        (tmp_path / d).mkdir()
        (tmp_path / d / "POSCAR").write_text(_POSCAR_NEB)

    fake_atom = MagicMock()
    with patch("tools4vasp.neb2movie.io.read", return_value=fake_atom):
        neb2movie.run(outFile=str(outfile), workdir=str(tmp_path))

    assert (tmp_path / "movie.xyz.bak").is_file()


def test_neb2movie_wrap_applies_to_frames(tmp_path):
    """main() with wrap=True must call frame.wrap() for every image."""
    from tools4vasp import neb2movie

    for d in ["00", "01", "02"]:
        (tmp_path / d).mkdir()
        (tmp_path / d / "POSCAR").write_text(_POSCAR_NEB)

    fake_atom = MagicMock()
    with patch("tools4vasp.neb2movie.io.read", return_value=fake_atom):
        neb2movie.run(outFile=str(tmp_path / "movie.xyz"), workdir=str(tmp_path), wrap=True)

    assert fake_atom.wrap.call_count == 3


# ===========================================================================
# freq2mode — read_frequency_from_outcar, getAtomsFromOutcar
# ===========================================================================

_OUTCAR_FREQ = """\
 1 f/i=   0.50 THz    3.14 2PiTHz    16.7 cm-1    2.1 meV
     X         Y         Z           dx          dy          dz
      0.0       0.0       0.0         0.1         0.2         0.3
      5.0       0.0       0.0        -0.1        -0.2        -0.3
 2 f/i=   0.30 THz    1.88 2PiTHz    10.0 cm-1    1.2 meV
     X         Y         Z           dx          dy          dz
      0.0       0.0       0.0         0.4         0.5         0.6
      5.0       0.0       0.0        -0.4        -0.5        -0.6
"""


def test_read_frequency_from_outcar():
    """read_frequency_from_outcar() must parse 3 displacement values per atom."""
    from tools4vasp.freq2mode import read_frequency_from_outcar, get_frequencies

    lines = _OUTCAR_FREQ.splitlines(keepends=True)
    freqs = get_frequencies(lines)
    assert len(freqs) == 2

    mock_atoms = MagicMock()
    mock_atoms.__len__ = MagicMock(return_value=2)

    freq_data = read_frequency_from_outcar(lines, freqs[0], mock_atoms)
    assert len(freq_data) == 2
    assert freq_data[0] == pytest.approx([0.1, 0.2, 0.3])
    assert freq_data[1] == pytest.approx([-0.1, -0.2, -0.3])


# ===========================================================================
# split_surf_and_mol
# ===========================================================================

def _make_slab_with_molecule():
    """Create a 3x3 Cu(100) 2-layer slab (9 atoms/layer) with CO on top."""
    from ase import Atoms
    positions = []
    # Layer 1 at z=0, 3x3 grid
    for i in range(3):
        for j in range(3):
            positions.append([i * 2.55, j * 2.55, 0.0])
    # Layer 2 at z=1.8, 3x3 grid offset
    for i in range(3):
        for j in range(3):
            positions.append([i * 2.55 + 1.275, j * 2.55 + 1.275, 1.8])
    # CO molecule on top
    positions.append([3.825, 3.825, 5.0])
    positions.append([3.825, 3.825, 6.13])
    symbols = ['Cu'] * 18 + ['C', 'O']
    cell = [7.65, 7.65, 25.0]
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def test_detect_surf_separates_slab_and_molecule():
    """detect_surf() must assign Cu to surface and C,O to molecule."""
    from tools4vasp.split_surf_and_mol import detect_surf
    atoms = _make_slab_with_molecule()
    surf, mol = detect_surf(atoms, plot=False)
    assert len(surf) == 18
    assert len(mol) == 2
    assert set(mol.get_chemical_symbols()) == {'C', 'O'}
    assert set(surf.get_chemical_symbols()) == {'Cu'}


def test_detect_surf_preserves_tags():
    """detect_surf() must store original indices in atom tags."""
    from tools4vasp.split_surf_and_mol import detect_surf
    atoms = _make_slab_with_molecule()
    surf, mol = detect_surf(atoms, plot=False)
    all_tags = sorted(list(surf.get_tags()) + list(mol.get_tags()))
    assert all_tags == list(range(20))


def test_count_z_voxels_using_window():
    """count_z_voxels_using_window() must count atoms in z-windows."""
    from tools4vasp.split_surf_and_mol import count_z_voxels_using_window
    z_coords = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    points, counts = count_z_voxels_using_window(20.0, 0.5, 1.0, z_coords)
    assert counts[0] == 3


def test_find_plateaus_with_known_data():
    """find_plateaus() must identify consecutive equal non-zero values."""
    from tools4vasp.split_surf_and_mol import find_plateaus
    data = [0, 0, 3, 3, 3, 0, 0, 2, 2, 0]
    plateaus, heights = find_plateaus(data)
    assert len(plateaus) == 2
    assert plateaus[0] == (2, 4)
    assert heights[0] == 3


def test_split_surf_and_mol_run(tmp_path, monkeypatch):
    """run() must write POSCAR_surf and POSCAR_mol."""
    from tools4vasp.split_surf_and_mol import run
    atoms = _make_slab_with_molecule()
    poscar = tmp_path / "POSCAR"
    atoms.write(str(poscar), format='vasp')
    monkeypatch.chdir(tmp_path)
    surf, mol = run(str(poscar))
    assert (tmp_path / "POSCAR_surf").is_file()
    assert (tmp_path / "POSCAR_mol").is_file()
    assert len(surf) == 18
    assert len(mol) == 2


def test_split_surf_and_mol_main_cli(tmp_path, monkeypatch):
    """main() must parse sys.argv and produce output files."""
    from tools4vasp.split_surf_and_mol import main
    atoms = _make_slab_with_molecule()
    poscar = tmp_path / "POSCAR"
    atoms.write(str(poscar), format='vasp')
    monkeypatch.chdir(tmp_path)
    with patch("sys.argv", ["split_surf_and_mol", str(poscar)]):
        main()
    assert (tmp_path / "POSCAR_surf").is_file()
    assert (tmp_path / "POSCAR_mol").is_file()


# ===========================================================================
# xyz2POSCAR
# ===========================================================================

_POSCAR_CUBIC_H = """\
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

_XYZ_BENZENE = """\
6
benzene
C  0.000  1.398  0.000
C  1.210  0.699  0.000
C  1.210 -0.699  0.000
C  0.000 -1.398  0.000
C -1.210 -0.699  0.000
C -1.210  0.699  0.000
"""


def test_xyz2poscar_run_basic(tmp_path):
    """run() must write a VASP POSCAR from an xyz and cell source."""
    from tools4vasp.xyz2POSCAR import run
    xyz = tmp_path / "mol.xyz"
    xyz.write_text(_XYZ_BENZENE)
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_CUBIC_H)
    out = str(tmp_path / "POSCAR_new")
    run(str(xyz), str(poscar), out, rot=False, cen=False, sor=False, const=False)
    assert (tmp_path / "POSCAR_new").is_file()
    content = (tmp_path / "POSCAR_new").read_text()
    assert "C" in content


def test_xyz2poscar_run_with_rotation(tmp_path):
    """run() with rot=True must produce output without error."""
    from tools4vasp.xyz2POSCAR import run
    xyz = tmp_path / "mol.xyz"
    xyz.write_text(_XYZ_BENZENE)
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_CUBIC_H)
    out = str(tmp_path / "POSCAR_new")
    run(str(xyz), str(poscar), out, rot=True, cen=True, sor=True, const=False)
    assert (tmp_path / "POSCAR_new").is_file()


def test_xyz2poscar_run_with_constraints(tmp_path):
    """run() with const=True must add selective dynamics."""
    from tools4vasp.xyz2POSCAR import run
    xyz = tmp_path / "mol.xyz"
    xyz.write_text(_XYZ_BENZENE)
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_CUBIC_H)
    out = str(tmp_path / "POSCAR_new")
    run(str(xyz), str(poscar), out, rot=False, cen=False, sor=False, const=True)
    content = (tmp_path / "POSCAR_new").read_text()
    assert "Selective" in content or "F" in content


def test_xyz2poscar_main_cli(tmp_path):
    """main() must parse sys.argv and call run()."""
    from tools4vasp.xyz2POSCAR import main
    xyz = tmp_path / "mol.xyz"
    xyz.write_text(_XYZ_BENZENE)
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_POSCAR_CUBIC_H)
    out = str(tmp_path / "POSCAR_new")
    with patch("sys.argv", ["xyz2POSCAR", str(xyz), str(poscar),
                            "--outfile", out, "--no_rotation_to_xy"]):
        main()
    assert (tmp_path / "POSCAR_new").is_file()


# ===========================================================================
# plotHOMA_withPBC
# ===========================================================================

def test_plotHOMA_benzene_with_rings(tmp_path):
    """run() on benzene with manually specified ring must not error."""
    from tools4vasp.plotHOMA_withPBC import run
    coords = tmp_path / "benzene.xyz"
    coords.write_text(_XYZ_BENZENE)
    outfile = str(tmp_path / "homa.svg")
    with patch("tools4vasp.plotHOMA_withPBC.plt") as mock_plt:
        run(str(coords), outfile, C1=0, C2=1, a="y",
            d_opt=1.398, norm=362.9, rings=[[0, 1, 2, 3, 4, 5]],
            pbc_cutoff=2.8, max_path_len=8, no_of_cyc_combs=4,
            atom_types=["C"], pbc=False, no_values=False)
    mock_plt.savefig.assert_called_once()


def test_plotHOMA_empty_cycles_error(tmp_path):
    """run() must exit when no cycles are found in an acyclic system."""
    from tools4vasp.plotHOMA_withPBC import run
    isolated = tmp_path / "isolated.xyz"
    isolated.write_text("2\nisolated\nC 0.0 0.0 0.0\nC 10.0 10.0 10.0\n")
    with patch("tools4vasp.plotHOMA_withPBC.plt"), \
         pytest.raises(SystemExit):
        run(str(isolated), str(tmp_path / "out.svg"), C1=0, C2=1, a="y",
            d_opt=1.398, norm=362.9, rings=[],
            pbc_cutoff=2.8, max_path_len=8, no_of_cyc_combs=4,
            atom_types=["C"], pbc=False, no_values=False)


def test_plotHOMA_main_cli(tmp_path):
    """main() must parse sys.argv and call run()."""
    from tools4vasp.plotHOMA_withPBC import main
    coords = tmp_path / "benzene.xyz"
    coords.write_text(_XYZ_BENZENE)
    outfile = str(tmp_path / "homa.svg")
    with patch("tools4vasp.plotHOMA_withPBC.plt"), \
         patch("sys.argv", ["plotHOMA_withPBC", str(coords), "--file", outfile,
                            "--rings", "0", "1", "2", "3", "4", "5"]):
        main()
