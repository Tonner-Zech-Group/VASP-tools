"""Tests for tools4vasp.vasplint.

The suite is built around one directory that passes every check, which each test
then breaks in exactly one way. That structure is deliberate: it proves each
check can fire, and it proves the others do not fire spuriously when it does.
"""

import json
import os
from unittest.mock import patch

import pytest

from tools4vasp.vasplint import (
    compare_incar_to_outcar,
    infer_run_type,
    lint,
    main,
    outcar_effective_tags,
    run,
)
from tools4vasp.vaspsetup import (
    VaspSetupError,
    parse_incar,
    rel_symlink,
    render_incar,
)

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


def test_a_built_incar_with_no_template_configured_warns(tmp_path):
    """The strongest check must never read as a silent pass (council 2026-09-02)."""
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR",
                 overrides={"ENCUT": ("500", "converged for this system")})
    result = lint(d)                       # no --template, no env
    warned = _warnings(result, "template")
    assert warned and "not checked" in warned[0]["message"]
    assert not any("template" in note for note in result["skipped"])


def test_templates_can_come_from_the_environment(tmp_path, monkeypatch):
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR",
                 overrides={"ENCUT": ("500", "converged for this system")})
    monkeypatch.setenv("VASPLINT_TEMPLATES", str(templates))
    assert lint(d)["findings"] == []
    text = (d / "INCAR").read_text().replace("EDIFF = 1.00e-06", "EDIFF = 1.00e-04")
    (d / "INCAR").write_text(text)
    assert _errors(lint(d), "template")    # and it now actually fires


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
    assert _errors(lint(d), "template") == []


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


def test_a_self_contained_directory_verifies_with_no_configuration(tmp_path):
    """A copy of the template inside the run dir makes the check travel."""
    templates = _template_dir(tmp_path)
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR",
                 overrides={"ENCUT": ("500", "converged for this system")})
    (d / "INCAR.template").write_text((templates / "INCAR.template").read_text())
    assert lint(d)["findings"] == []
    text = (d / "INCAR").read_text().replace("EDIFF = 1.00e-06", "EDIFF = 1.00e-04")
    (d / "INCAR").write_text(text)
    assert _errors(lint(d), "template")


def test_the_run_directory_kpoints_is_not_compared_against_itself(tmp_path):
    templates = _template_dir(tmp_path, kpoints="auto\n0\nGamma\n4 4 1\n0 0 0\n")
    d = _run_dir(tmp_path)
    render_incar(templates / "INCAR.template", d / "INCAR")
    (d / "INCAR.template").write_text((templates / "INCAR.template").read_text())
    assert _errors(lint(d, template=templates), "kpoints")   # 2x2x1 vs 4x4x1


# ---------------------------------------------------------------------------
# post-run comparison against what VASP reports having used (council item 5)
# ---------------------------------------------------------------------------

# The echo really looks like this: INCAR syntax, several tags per line separated
# by ';', trailing prose, values sometimes truncated or reformatted. Taken from
# a VASP 5.4.4 OUTCAR.
OUTCAR_ECHO = """\
 running on   48 total cores
 distrk:  each k-point on   24 cores,    2 groups
 distr:  one band on NCORES_PER_BAND=  24 cores,    1 groups
 Found      4 irreducible k-points:
   k-points           NKPTS =      4   k-points in BZ     NKDIM =      4
   PREC   = accura    normal or accurate
   ISTART =      1    job   : 0-new  1-cont  2-samecut
   ENCUT  =  400.0 eV  29.40 Ry
   EDIFF  = 0.1E-05   stopping-criterion for ELM
   NELM   =    150;   NELMIN=  5; NELMDL=-15     # of ELM steps
   ISMEAR =      0;   SIGMA  =   0.05  broadening in eV
   IALGO  =     68    algorithm
   ISPIN  =      1    spin polarized calculation?
   LREAL  =      T    real-space projection
     AMIX     =   0.20;   BMIX     =  0.00
   MAXMIX =     30    max number of ionic steps stored
   IVDW         = 12
------------------------------------------ Iteration      1(   1)  ---------
 aborting loop because EDIFF is reached
 General timing and accounting informations for this job:
"""

INCAR_MATCHING_ECHO = """\
PREC = Accurate
ISTART = 1
ENCUT = 400
EDIFF = 1.00e-06
NELM = 150
NELMIN = 5
NELMDL = -15
ISMEAR = 0
SIGMA = 0.05
ALGO = Fast
ISPIN = 1
LREAL = Auto
AMIX = 0.2
BMIX = 0.0001
MAXMIX = 30
IVDW = 12
KPAR = 2
NCORE = 24
"""


def test_outcar_effective_tags_recovers_the_derived_quantities():
    eff = outcar_effective_tags(OUTCAR_ECHO)
    assert eff["_NKPTS_IRR"] == "4"
    assert eff["_NCORE"] == "24"
    assert eff["_KPAR"] == "2"
    assert eff["_RANKS"] == "48"
    assert eff["IALGO"] == "68"
    assert eff["IVDW"] == "12"
    assert eff["SIGMA"] == "0.05"       # second tag on a ';' line
    assert eff["BMIX"] == "0.00"        # reformatted by VASP


def test_a_faithful_incar_produces_no_outcar_findings():
    """None of VASP's reformatting may be mistaken for a mismatch."""
    findings, _ = compare_incar_to_outcar(parse_incar(INCAR_MATCHING_ECHO), OUTCAR_ECHO)
    assert findings == [], [f["message"] for f in findings]


def test_a_post_run_incar_edit_is_caught():
    tags = parse_incar(INCAR_MATCHING_ECHO.replace("ENCUT = 400", "ENCUT = 520"))
    findings, _ = compare_incar_to_outcar(tags, OUTCAR_ECHO)
    assert any("ENCUT" in f["message"] and f["level"] == "error" for f in findings)


def test_algo_is_verified_through_ialgo():
    tags = parse_incar(INCAR_MATCHING_ECHO.replace("ALGO = Fast", "ALGO = Normal"))
    findings, _ = compare_incar_to_outcar(tags, OUTCAR_ECHO)
    assert any("IALGO" in f["message"] for f in findings)


def test_a_misspelled_declared_override_is_flagged():
    """VASP silently ignores tags it does not know; the echo is the only witness."""
    tags = parse_incar(INCAR_MATCHING_ECHO + "ENCUTT = 520\n")
    findings, _ = compare_incar_to_outcar(tags, OUTCAR_ECHO, declared=["ENCUTT"])
    assert any("ENCUTT" in f["message"] and "spelling" in f["message"]
               for f in findings)


def test_an_undeclared_unknown_tag_is_only_a_note():
    tags = parse_incar(INCAR_MATCHING_ECHO + "ENCUTT = 520\n")
    findings, skipped = compare_incar_to_outcar(tags, OUTCAR_ECHO)
    assert not any("ENCUTT" in f["message"] for f in findings)
    assert any("ENCUTT" in note for note in skipped)


def test_istart_may_be_lowered_by_vasp_but_not_raised():
    faithful, _ = compare_incar_to_outcar(
        parse_incar("ISTART = 1\n"), OUTCAR_ECHO.replace("ISTART =      1", "ISTART =      0"))
    assert faithful == []               # 1 -> 0 is VASP finding no WAVECAR
    raised, _ = compare_incar_to_outcar(parse_incar("ISTART = 0\n"), OUTCAR_ECHO)
    assert any("ISTART" in f["message"] for f in raised)


def test_kpar_is_exact_against_the_irreducible_count():
    tags = parse_incar(INCAR_MATCHING_ECHO.replace("KPAR = 2", "KPAR = 6"))
    findings, _ = compare_incar_to_outcar(tags, OUTCAR_ECHO)
    assert any("irreducible" in f["message"] for f in findings)


def test_ediff_tolerance_respects_the_printed_exponent():
    """0.1E-05 is precise to 5e-7, so a 1e-4 EDIFF must not slip through."""
    ok, _ = compare_incar_to_outcar(parse_incar("EDIFF = 1.00e-06\n"), OUTCAR_ECHO)
    assert ok == []
    bad, _ = compare_incar_to_outcar(parse_incar("EDIFF = 1.00e-04\n"), OUTCAR_ECHO)
    assert any("EDIFF" in f["message"] for f in bad)


def test_a_truncated_outcar_is_an_error():
    findings, _ = compare_incar_to_outcar(parse_incar("ENCUT = 400\n"), "nothing here")
    assert findings and findings[0]["level"] == "error"


def test_lint_outcar_flag_skips_a_directory_that_has_not_run(tmp_path):
    d = _run_dir(tmp_path)
    result = lint(d, outcar=True)
    assert _errors(result, "outcar") == []
    assert any("no OUTCAR" in note for note in result["skipped"])


def test_lint_outcar_flag_reads_a_real_looking_outcar(tmp_path):
    d = _run_dir(tmp_path, incar=INCAR_MATCHING_ECHO + "IBRION = -1\nNSW = 0\n")
    (d / "OUTCAR").write_text(OUTCAR_ECHO)
    assert _errors(lint(d, outcar=True), "outcar") == []
    (d / "INCAR").write_text((d / "INCAR").read_text().replace("ENCUT = 400", "ENCUT = 520"))
    assert _errors(lint(d, outcar=True), "outcar")


# ---------------------------------------------------------------------------
# the fixture matrix by tag space, not by this project's system (council item 7)
# ---------------------------------------------------------------------------

def _potcar_enmax(path, entries):
    """A POTCAR carrying explicit TITEL/ENMAX pairs."""
    path.write_text("".join(
        f" {titel}\n   ENMAX  =  {enmax:.3f}; ENMIN  =  {enmax * 0.75:.3f} eV\n"
        f"   TITEL  = {titel}\nEnd of Dataset\n" for titel, enmax in entries))
    return path


def test_encut_below_the_potcar_enmax_is_an_error(tmp_path):
    """The most common real VASP input error, and absent from every fixture until now."""
    d = _run_dir(tmp_path, poscar=POSCAR_SUFFIXED)
    _potcar_enmax(d / "POTCAR", [("PAW_PBE K_pv 17Jan2003", 259.0),
                                 ("PAW_PBE Ti_sv 07Sep2000", 274.6)])
    findings = _errors(lint(d), "encut")          # template ENCUT = 400 is fine
    assert findings == []
    (d / "INCAR").write_text(INCAR_SINGLE_POINT.replace("ENCUT = 400", "ENCUT = 250"))
    findings = _errors(lint(d), "encut")
    assert findings and "274.6" in findings[0]["message"]


def test_encut_at_exactly_the_enmax_passes(tmp_path):
    d = _run_dir(tmp_path, poscar=POSCAR_SUFFIXED)
    _potcar_enmax(d / "POTCAR", [("PAW_PBE K_pv 17Jan2003", 400.0),
                                 ("PAW_PBE Ti_sv 07Sep2000", 400.0)])
    assert _errors(lint(d), "encut") == []


def test_volume_relaxation_wants_headroom_over_enmax(tmp_path):
    incar = INCAR_SINGLE_POINT.replace("IBRION = -1", "IBRION = 2").replace(
        "NSW = 0", "NSW = 100") + "ISIF = 3\n"
    d = _run_dir(tmp_path, poscar=POSCAR_SUFFIXED, incar=incar)
    _potcar_enmax(d / "POTCAR", [("PAW_PBE K_pv 17Jan2003", 350.0),
                                 ("PAW_PBE Ti_sv 07Sep2000", 350.0)])
    warned = _warnings(lint(d), "encut")
    assert warned and "1.3" in warned[0]["message"]


def test_magmom_of_the_wrong_length_is_an_error(tmp_path):
    """VASP reads MAGMOM positionally, so a short list misassigns every later atom."""
    incar = INCAR_SINGLE_POINT.replace("ISPIN = 1", "") + "ISPIN = 2\nMAGMOM = 3*1.0\n"
    d = _run_dir(tmp_path, incar=incar)           # the POSCAR has 2 atoms
    assert _errors(lint(d), "spin")
    good = INCAR_SINGLE_POINT + "ISPIN = 2\nMAGMOM = 2*1.0\n"
    (d / "INCAR").write_text(good)
    assert _errors(lint(d), "spin") == []


def test_magmom_written_out_atom_by_atom_is_counted_too(tmp_path):
    incar = INCAR_SINGLE_POINT + "ISPIN = 2\nMAGMOM = 1.0 1.0 1.0\n"
    d = _run_dir(tmp_path, incar=incar)
    assert _errors(lint(d), "spin")


def test_ldau_with_too_low_lmaxmix_is_an_error(tmp_path):
    incar = INCAR_SINGLE_POINT + "LDAU = .TRUE.\nLMAXMIX = 2\n"
    d = _run_dir(tmp_path, incar=incar)
    assert _errors(lint(d), "ldau")
    (d / "INCAR").write_text(INCAR_SINGLE_POINT + "LDAU = .TRUE.\nLMAXMIX = 4\n")
    assert _errors(lint(d), "ldau") == []


def test_a_band_without_its_image_directories_is_an_error(tmp_path):
    incar = INCAR_SINGLE_POINT.replace("IBRION = -1", "IBRION = 3").replace(
        "NSW = 0", "NSW = 100") + "IMAGES = 3\nSPRING = -5.0\nICHAIN = 0\n"
    d = _run_dir(tmp_path, incar=incar)
    result = lint(d)
    assert result["run_type"] == "neb"
    findings = _errors(result, "neb")
    assert findings and "00..04" in findings[0]["message"]
    for name in ("00", "01", "02", "03", "04"):
        (d / name).mkdir()
        (d / name / "POSCAR").write_text(POSCAR_VASP5)
    assert _errors(lint(d), "neb") == []


def test_a_negative_poscar_scale_is_read_as_a_volume(tmp_path):
    """A negative scale factor means "target volume" and must not crash the parser."""
    poscar = POSCAR_VASP5.replace(" 1.0000000000000000", "-1000.0000000000000000")
    d = _run_dir(tmp_path, poscar=poscar)
    assert _errors(lint(d), "poscar") == []


def test_vasp6_only_tags_do_not_confuse_the_run_type(tmp_path):
    """ML_LMLFF and friends are unknown here but must not change the verdict."""
    incar = INCAR_SINGLE_POINT + "ML_LMLFF = .FALSE.\nLHYPERFINE = .FALSE.\n"
    d = _run_dir(tmp_path, incar=incar)
    result = lint(d)
    assert result["run_type"] == "single_point"
    assert result["errors"] == 0
