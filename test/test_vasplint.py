"""Tests for tools4vasp.vasplint.

The suite is built around one directory that passes every check, which each test
then breaks in exactly one way. That structure is deliberate: it proves each
check can fire, and it proves the others do not fire spuriously when it does.
"""

import json
import os
from unittest.mock import patch

import pytest

from tools4vasp.vasplint import infer_run_type, lint, main, run
from tools4vasp.vaspsetup import VaspSetupError, rel_symlink, render_incar

POSCAR_VASP5 = """\
Si H
 1.0000000000000000
    10.0000000000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000   10.0000000000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000   10.0000000000000000
 Si  H
  1   1
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.1000000000000000  0.0000000000000000  0.0000000000000000
"""

INCAR_SINGLE_POINT = """\
System = test
ENCUT = 400
EDIFF = 1.00e-06
IBRION = -1
NSW = 0
NCORE = 24
KPAR = 2
"""

INCAR_INTERACTIVE = """\
System = test
ENCUT = 400
IBRION = 11
INTERACTIVE = .TRUE.
ISIF = 2
NSW = 4
NCORE = 24
KPAR = 2
"""

KPOINTS = "auto\n0\nGamma\n2 2 1\n0 0 0\n"

JOB_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name=test-run
#SBATCH --output=slurm-test-run-%j.out
#SBATCH --mail-user=someone@example.org
#SBATCH --licenses=horse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -t 02:00:00
mpirun vasp_std
"""

POTCAR_BLOCK = """\
 PAW_PBE {el} {date}
   4.00000000000000000
 parameters from PSCTR are:
   TITEL  = PAW_PBE {el} {date}
End of Dataset
"""
_DATE = {"Si": "05Jan2001", "H": "15Jun2001", "O": "08Apr2002"}


def _potcar(path, elements):
    path.write_text("".join(
        POTCAR_BLOCK.format(el=el, date=_DATE[el]) for el in elements))
    return path


def _run_dir(tmp_path, incar=INCAR_SINGLE_POINT, elements=("Si", "H"),
             poscar=POSCAR_VASP5, kpoints=KPOINTS, job=JOB_SCRIPT, name="run"):
    """A directory that passes every check."""
    d = tmp_path / name
    d.mkdir()
    (d / "INCAR").write_text(incar)
    (d / "POSCAR").write_text(poscar)
    _potcar(d / "POTCAR", elements)
    if kpoints is not None:
        (d / "KPOINTS").write_text(kpoints)
    if job is not None:
        (d / "vasp.run").write_text(job)
    return d


def _errors(result, check=None):
    return [f for f in result["findings"]
            if f["level"] == "error" and (check is None or f["check"] == check)]


def _warnings(result, check=None):
    return [f for f in result["findings"]
            if f["level"] == "warning" and (check is None or f["check"] == check)]


# ---------------------------------------------------------------------------
# the known-good baseline
# ---------------------------------------------------------------------------

def test_clean_directory_has_no_findings(tmp_path):
    result = lint(_run_dir(tmp_path))
    assert result["findings"] == [], result["findings"]
    assert result["run_type"] == "single_point"


def test_lint_refuses_a_non_directory(tmp_path):
    (tmp_path / "file").write_text("x")
    with pytest.raises(VaspSetupError, match="not a directory"):
        lint(tmp_path / "file")


def test_infer_run_type_covers_each_marker():
    assert infer_run_type({"INTERACTIVE": ".TRUE."}) == "interactive"
    assert infer_run_type({"IMAGES": "5"}) == "neb"
    assert infer_run_type({"ICHAIN": "2", "IBRION": "3", "NSW": "100"}) == "dimer"
    assert infer_run_type({"IBRION": "-1"}) == "single_point"
    assert infer_run_type({"IBRION": "2", "NSW": "100"}) == "relax"


def test_missing_incar_skips_run_type_checks_instead_of_assuming(tmp_path):
    d = _run_dir(tmp_path)
    (d / "INCAR").unlink()
    result = lint(d)
    assert _errors(result, "incar")
    assert result["run_type"] == "unknown"
    assert any("run_type" in note for note in result["skipped"])


# ---------------------------------------------------------------------------
# POSCAR / POTCAR
# ---------------------------------------------------------------------------

def test_potcar_in_wrong_order_is_an_error(tmp_path):
    d = _run_dir(tmp_path)
    _potcar(d / "POTCAR", ["H", "Si"])           # POSCAR says Si then H
    assert _errors(lint(d), "potcar")


def test_missing_potcar_is_an_error(tmp_path):
    d = _run_dir(tmp_path)
    (d / "POTCAR").unlink()
    assert _errors(lint(d), "potcar")


def test_expected_titels_mismatch_is_an_error(tmp_path):
    d = _run_dir(tmp_path)
    result = lint(d, expected_titels={"Si": "PAW_PBE Si_sv 2000"})
    assert _errors(result, "potcar_provenance")


def test_expected_titels_match_is_clean(tmp_path):
    d = _run_dir(tmp_path)
    result = lint(d, expected_titels={"Si": "PAW_PBE Si 05Jan2001",
                                      "H": "PAW_PBE H 15Jun2001"})
    assert result["findings"] == []


def test_selective_dynamics_in_a_single_point_is_an_error(tmp_path):
    poscar = POSCAR_VASP5.replace("Direct", "Selective dynamics\nDirect")
    poscar = poscar.replace("  0.1000000000000000  0.0000000000000000  0.0000000000000000",
                            "  0.1000000000000000  0.0000000000000000  0.0000000000000000 F F F")
    d = _run_dir(tmp_path, poscar=poscar)
    assert _errors(lint(d), "poscar")


def test_missing_poscar_is_an_error(tmp_path):
    d = _run_dir(tmp_path)
    (d / "POSCAR").unlink()
    result = lint(d)
    assert _errors(result, "poscar")
    assert any("potcar" in note for note in result["skipped"])


# ---------------------------------------------------------------------------
# symlinks
# ---------------------------------------------------------------------------

def test_absolute_symlink_is_an_error(tmp_path):
    d = _run_dir(tmp_path)
    shared = _potcar(tmp_path / "POTCAR", ["Si", "H"])
    (d / "POTCAR").unlink()
    (d / "POTCAR").symlink_to(shared.resolve())          # absolute on purpose
    assert _errors(lint(d), "symlinks")


def test_relative_symlink_is_accepted(tmp_path):
    d = _run_dir(tmp_path)
    shared = _potcar(tmp_path / "POTCAR", ["Si", "H"])
    (d / "POTCAR").unlink()
    rel_symlink(shared, d / "POTCAR")
    assert lint(d)["findings"] == []


def test_dangling_symlink_is_an_error(tmp_path):
    d = _run_dir(tmp_path)
    (d / "CHGCAR").symlink_to("nowhere")
    assert _errors(lint(d), "symlinks")


# ---------------------------------------------------------------------------
# continuation runs
# ---------------------------------------------------------------------------

def test_continuation_from_a_live_contcar_is_an_error(tmp_path):
    src = tmp_path / "run1"
    src.mkdir()
    (src / "CONTCAR").write_text(POSCAR_VASP5)
    (src / "OUTCAR").write_text("General timing and accounting\n")
    now = 1_700_000_000
    os.utime(src / "OUTCAR", (now, now))
    os.utime(src / "CONTCAR", (now + 600, now + 600))
    d = _run_dir(tmp_path, name="run2")
    (d / "POSCAR").unlink()
    rel_symlink(src / "CONTCAR", d / "POSCAR")
    assert _errors(lint(d), "continuation")


def test_continuation_from_a_finished_contcar_is_clean(tmp_path):
    src = tmp_path / "run1"
    src.mkdir()
    (src / "CONTCAR").write_text(POSCAR_VASP5)
    (src / "OUTCAR").write_text("General timing and accounting\n")
    now = 1_700_000_000
    os.utime(src / "CONTCAR", (now, now))
    os.utime(src / "OUTCAR", (now + 60, now + 60))
    d = _run_dir(tmp_path, name="run2")
    (d / "POSCAR").unlink()
    rel_symlink(src / "CONTCAR", d / "POSCAR")
    assert lint(d)["findings"] == []


def test_poscar_symlink_to_something_other_than_a_contcar_warns(tmp_path):
    other = tmp_path / "geometry"
    other.write_text(POSCAR_VASP5)
    d = _run_dir(tmp_path)
    (d / "POSCAR").unlink()
    rel_symlink(other, d / "POSCAR")
    assert _warnings(lint(d), "continuation")


# ---------------------------------------------------------------------------
# interactive mode
# ---------------------------------------------------------------------------

def test_interactive_without_stdin_file_is_an_error(tmp_path):
    d = _run_dir(tmp_path, incar=INCAR_INTERACTIVE)
    assert _errors(lint(d), "interactive")


def test_interactive_nsw_too_small_is_an_error(tmp_path):
    d = _run_dir(tmp_path, incar=INCAR_INTERACTIVE)
    (d / "POSCAR.interactive").write_text("0 0 0\n\n0 0 0\n\n0 0 0\n\n0 0 0\n\n")
    result = lint(d)                     # 4 structures + POSCAR needs NSW >= 5
    assert _errors(result, "interactive")


def test_interactive_nsw_large_enough_is_clean(tmp_path):
    d = _run_dir(tmp_path, incar=INCAR_INTERACTIVE)
    (d / "POSCAR.interactive").write_text("0 0 0\n\n0 0 0\n\n0 0 0\n\n")
    assert lint(d)["findings"] == []


def test_interactive_with_a_moving_cell_is_an_error(tmp_path):
    d = _run_dir(tmp_path, incar=INCAR_INTERACTIVE.replace("ISIF = 2", "ISIF = 3"))
    (d / "POSCAR.interactive").write_text("0 0 0\n\n0 0 0\n\n0 0 0\n\n")
    assert _errors(lint(d), "run_type")


# ---------------------------------------------------------------------------
# restart files, job script, k-points
# ---------------------------------------------------------------------------

def test_icharg_without_chgcar_is_an_error(tmp_path):
    d = _run_dir(tmp_path, incar=INCAR_SINGLE_POINT + "ICHARG = 1\n")
    assert _errors(lint(d), "restart_files")


def test_icharg_with_chgcar_is_clean(tmp_path):
    d = _run_dir(tmp_path, incar=INCAR_SINGLE_POINT + "ICHARG = 1\n")
    (d / "CHGCAR").write_text("density")
    assert lint(d)["findings"] == []


@pytest.mark.parametrize("flag", ["--mail-user=", "--output=", "--job-name="])
def test_each_universally_required_sbatch_flag_is_checked(tmp_path, flag):
    stripped = "\n".join(ln for ln in JOB_SCRIPT.splitlines()
                         if not ln.startswith(f"#SBATCH {flag}"))
    d = _run_dir(tmp_path, job=stripped + "\n")
    assert any(flag in f["message"] for f in _errors(lint(d), "job_script"))


def test_site_flag_is_not_required_by_default(tmp_path):
    """--licenses is a ZIH rule, so a site-agnostic run must not demand it."""
    job = "\n".join(ln for ln in JOB_SCRIPT.splitlines()
                    if not ln.startswith("#SBATCH --licenses="))
    d = _run_dir(tmp_path, job=job + "\n")
    assert _errors(lint(d), "job_script") == []
    assert any("--licenses=" in f["message"]
               for f in _errors(lint(d, require=["--licenses="]), "job_script"))


def test_commented_site_flag_does_not_satisfy_the_requirement(tmp_path):
    job = JOB_SCRIPT.replace("#SBATCH --licenses=horse",
                             '##SBATCH --licenses="SCRATCH"')
    d = _run_dir(tmp_path, job=job)
    result = lint(d, require=["--licenses="])
    assert any("--licenses=" in f["message"] for f in _errors(result, "job_script"))


def test_no_job_script_is_skipped_not_failed(tmp_path):
    d = _run_dir(tmp_path, job=None)
    result = lint(d)
    assert _errors(result, "job_script") == []
    assert any("job_script" in note for note in result["skipped"])


def test_parallel_layout_mismatch_is_a_warning(tmp_path):
    job = JOB_SCRIPT.replace("--ntasks-per-node=48", "--ntasks-per-node=50")
    d = _run_dir(tmp_path, job=job)
    assert _warnings(lint(d), "parallel_layout")


def test_kpar_above_the_mesh_size_is_an_error(tmp_path):
    d = _run_dir(tmp_path, incar=INCAR_SINGLE_POINT.replace("KPAR = 2", "KPAR = 8"))
    assert _errors(lint(d), "kpoints")


def test_no_kpoints_and_no_kspacing_is_an_error(tmp_path):
    d = _run_dir(tmp_path, kpoints=None)
    assert _errors(lint(d), "kpoints")


def test_kspacing_replaces_the_kpoints_file(tmp_path):
    d = _run_dir(tmp_path, kpoints=None,
                 incar=INCAR_SINGLE_POINT + "KSPACING = 0.25\n")
    assert _errors(lint(d), "kpoints") == []


# ---------------------------------------------------------------------------
# INCAR against its template
# ---------------------------------------------------------------------------

def _template_dir(tmp_path, incar=INCAR_SINGLE_POINT, kpoints=KPOINTS):
    t = tmp_path / "templates"
    t.mkdir()
    (t / "INCAR.template").write_text(incar)
    if kpoints is not None:
        (t / "KPOINTS").write_text(kpoints)
    return t


def test_declared_override_passes_the_template_check(tmp_path):
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR",
                 overrides={"ENCUT": ("500", "converged for this system")})
    assert lint(d, template=templates)["findings"] == []


def test_undeclared_hand_edit_is_caught(tmp_path):
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR",
                 overrides={"ENCUT": ("500", "converged for this system")})
    text = (d / "INCAR").read_text().replace("EDIFF = 1.00e-06", "EDIFF = 1.00e-04")
    (d / "INCAR").write_text(text)
    findings = _errors(lint(d, template=templates), "template")
    assert findings and "EDIFF" in findings[0]["message"]


def test_template_changed_after_the_build_is_caught(tmp_path):
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR")
    (templates / "INCAR.template").write_text(INCAR_SINGLE_POINT + "LASPH = .TRUE.\n")
    findings = _errors(lint(d, template=templates), "template")
    assert findings and "has changed" in findings[0]["message"]


def test_incar_without_provenance_warns_when_a_template_is_given(tmp_path):
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    assert _warnings(lint(d, template=templates), "template")


def test_unknown_template_name_is_an_error(tmp_path):
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR")
    (templates / "INCAR.template").rename(templates / "INCAR.other")
    assert _errors(lint(d, template=templates), "template")


def test_kmesh_differing_from_the_template_is_an_error(tmp_path):
    templates = _template_dir(tmp_path, kpoints="auto\n0\nGamma\n4 4 1\n0 0 0\n")
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR")
    assert _errors(lint(d, template=templates), "kpoints")


def test_dipole_tag_only_on_one_side_warns(tmp_path):
    """Compared against INCAR.template, not against a file called "INCAR"."""
    templates = _template_dir(tmp_path)
    assert not (templates / "INCAR").exists()
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR",
                 overrides={"LDIPOL": (".TRUE.", "slab dipole correction")})
    assert _warnings(lint(d, template=templates), "dipole")


def test_without_a_template_the_comparison_is_skipped(tmp_path):
    result = lint(_run_dir(tmp_path))
    assert any("template" in note for note in result["skipped"])


# ---------------------------------------------------------------------------
# reporting and the command line
# ---------------------------------------------------------------------------

def test_run_prints_a_report(tmp_path, capsys):
    result = run(_run_dir(tmp_path))
    out = capsys.readouterr().out
    assert "all checks passed" in out
    assert result["ok"] is True


def test_run_strict_fails_on_a_warning(tmp_path):
    job = JOB_SCRIPT.replace("--ntasks-per-node=48", "--ntasks-per-node=50")
    d = _run_dir(tmp_path, job=job)
    assert run(d, verbose=False)["ok"] is True
    assert run(d, verbose=False, strict=True)["ok"] is False


def test_main_exits_zero_on_a_clean_directory(tmp_path):
    d = _run_dir(tmp_path)
    with patch("sys.argv", ["vasplint", str(d)]), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_main_exits_nonzero_on_an_error(tmp_path):
    d = _run_dir(tmp_path)
    (d / "POTCAR").unlink()
    with patch("sys.argv", ["vasplint", str(d)]), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_json_output_is_machine_readable(tmp_path, capsys):
    d = _run_dir(tmp_path)
    with patch("sys.argv", ["vasplint", "--json", str(d)]), \
            pytest.raises(SystemExit):
        main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_type"] == "single_point"
    assert payload["errors"] == 0


def test_main_reports_a_bad_path_without_crashing(tmp_path, capsys):
    with patch("sys.argv", ["vasplint", str(tmp_path / "nope")]), \
            pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "ERROR" in capsys.readouterr().out


def test_main_handles_several_directories(tmp_path, capsys):
    good = _run_dir(tmp_path, name="good")
    bad = _run_dir(tmp_path, name="bad")
    (bad / "POTCAR").unlink()
    with patch("sys.argv", ["vasplint", "--json", str(good), str(bad)]), \
            pytest.raises(SystemExit) as exc:
        main()
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    assert exc.value.code == 1


def test_declared_override_without_a_reason_line_is_caught(tmp_path):
    """A hand-stripped reason line means an undocumented decision."""
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR",
                 overrides={"ENCUT": ("500", "converged for this system")})
    kept = [ln for ln in (d / "INCAR").read_text().splitlines()
            if not ln.startswith("# ENCUT")]
    (d / "INCAR").write_text("\n".join(kept) + "\n")
    findings = _errors(lint(d), "template")
    assert findings and "no reason line" in findings[0]["message"]


def test_reason_lines_are_checked_without_a_template(tmp_path):
    """The header is self-contained, so this check does not need --template."""
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR",
                 overrides={"ENCUT": ("500", "converged for this system")})
    assert lint(d)["findings"] == []


# ---------------------------------------------------------------------------
# PAW potential suffixes (Copilot review on PR #30)
# ---------------------------------------------------------------------------

POSCAR_SUFFIXED = """\
K Ti
 1.0000000000000000
    10.0000000000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000   10.0000000000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000   10.0000000000000000
 K   Ti
  1   1
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.1000000000000000  0.0000000000000000  0.0000000000000000
"""

SUFFIXED_TITELS = ("PAW_PBE K_pv 17Jan2003", "PAW_PBE Ti_sv 07Sep2000")


def _potcar_titels(path, titels):
    path.write_text("".join(
        f" {t}\n   4.0\n parameters from PSCTR are:\n   TITEL  = {t}\n"
        "End of Dataset\n" for t in titels))
    return path


def test_suffixed_potcar_is_not_reported_as_a_mismatch(tmp_path):
    """K_pv against a POSCAR saying K is correct, not a false order mismatch."""
    d = _run_dir(tmp_path, poscar=POSCAR_SUFFIXED)
    _potcar_titels(d / "POTCAR", SUFFIXED_TITELS)
    assert _errors(lint(d), "potcar") == []


def test_suffixed_potcar_in_the_wrong_order_is_still_caught(tmp_path):
    d = _run_dir(tmp_path, poscar=POSCAR_SUFFIXED)
    _potcar_titels(d / "POTCAR", tuple(reversed(SUFFIXED_TITELS)))
    assert _errors(lint(d), "potcar")


def test_expected_titels_still_pin_the_exact_suffixed_potential(tmp_path):
    d = _run_dir(tmp_path, poscar=POSCAR_SUFFIXED)
    _potcar_titels(d / "POTCAR", SUFFIXED_TITELS)
    clean = lint(d, expected_titels={"K": "PAW_PBE K_pv 17Jan2003",
                                     "Ti": "PAW_PBE Ti_sv 07Sep2000"})
    assert _errors(clean, "potcar_provenance") == []
    swapped = lint(d, expected_titels={"Ti": "PAW_PBE Ti 08Apr2002"})
    assert _errors(swapped, "potcar_provenance")
