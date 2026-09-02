"""Tests for tools4vasp.vaspsetup.

Every guard gets a fixture that makes it *fire*, not only one that passes it: a
check that cannot fail is indistinguishable from no check at all.
"""

import os
import stat
from unittest.mock import patch

import pytest

from tools4vasp.vaspsetup import (
    REQUIRED_SBATCH,
    SITE_SBATCH,
    VaspSetupError,
    build_potcar_from_pp_path,
    build_potcar_from_reference,
    check_run_type,
    continuation_dir,
    count_interactive_structures,
    element_of_titel,
    forbidden_tags,
    incar_provenance,
    link_potcar,
    normalise_overrides,
    parse_incar,
    patch_runscript,
    poscar_blocks,
    read_poscar_blocks,
    read_titels,
    rel_symlink,
    render_incar,
    split_potcar,
    template_fingerprint,
    write_interactive_stdin,
    write_poscar,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

INCAR_TEMPLATE = """\
System = test-system
ENCUT = 400 #PW cutoff
PREC = Accurate
EDIFF = 1.00e-06 ; NELM = 150 #two tags on one line
ISMEAR = 0 ! bang comment
IBRION = -1
NSW = 0
NCORE = 24
KPAR = 2
"""

NEB_TEMPLATE = INCAR_TEMPLATE + """\
ICHAIN = 0
IMAGES = 5
SPRING = -5.0
"""

POTCAR_BLOCK = """\
 PAW_PBE {el} {date}
   {zval}.00000000000000000
 parameters from PSCTR are:
   VRHFIN ={el}: s2p2
   TITEL  = PAW_PBE {el} {date}
   LEXCH  = PE
End of Dataset
"""

TITEL = {"Si": "PAW_PBE Si 05Jan2001", "H": "PAW_PBE H 15Jun2001",
         "O": "PAW_PBE O 08Apr2002", "C": "PAW_PBE C 08Apr2002"}
_DATE = {el: t.split()[-1] for el, t in TITEL.items()}

OUTCAR_FINISHED = """\
 running on   48 total cores
------------------------------------------ Iteration      1(   1)  ---------
 aborting loop because EDIFF is reached
 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
"""
OUTCAR_UNCONVERGED = """\
 running on   48 total cores
------------------------------------------ Iteration      1(   1)  ---------
 aborting loop because EDIFF is reached
 General timing and accounting informations for this job:
"""
OUTCAR_RUNNING = """\
 running on   48 total cores
------------------------------------------ Iteration      1(   1)  ---------
"""


def _potcar(path, elements):
    path.write_text("".join(
        POTCAR_BLOCK.format(el=el, date=_DATE[el], zval=4) for el in elements))
    return path


def _potcar_titels(path, titels):
    """A POTCAR built from explicit TITEL strings, suffixes included."""
    blocks = []
    for titel in titels:
        symbol = titel.split()[1]
        blocks.append(
            f" {titel}\n   4.00000000000000000\n parameters from PSCTR are:\n"
            f"   VRHFIN ={symbol}: s2p2\n   TITEL  = {titel}\n   LEXCH  = PE\n"
            "End of Dataset\n")
    path.write_text("".join(blocks))
    return path


def _atoms(symbols):
    from ase import Atoms
    n = len(symbols)
    return Atoms(symbols=symbols,
                 positions=[(i * 1.5, 0.0, 0.0) for i in range(n)],
                 cell=[10.0, 10.0, 10.0], pbc=True)


@pytest.fixture
def template(tmp_path):
    path = tmp_path / "INCAR.template"
    path.write_text(INCAR_TEMPLATE)
    return path


# ---------------------------------------------------------------------------
# rel_symlink
# ---------------------------------------------------------------------------

def test_rel_symlink_is_relative_and_resolves(tmp_path):
    target = tmp_path / "shared" / "POTCAR"
    target.parent.mkdir()
    target.write_text("x")
    link = tmp_path / "run" / "POTCAR"
    rel = rel_symlink(target, link)
    assert not os.path.isabs(os.readlink(link))
    assert rel == os.path.join("..", "shared", "POTCAR")
    assert link.read_text() == "x"


def test_rel_symlink_refuses_missing_target(tmp_path):
    with pytest.raises(VaspSetupError, match="target does not exist"):
        rel_symlink(tmp_path / "nope", tmp_path / "link")


def test_rel_symlink_replaces_existing_link(tmp_path):
    (tmp_path / "a").write_text("a")
    (tmp_path / "b").write_text("b")
    rel_symlink(tmp_path / "a", tmp_path / "link")
    rel_symlink(tmp_path / "b", tmp_path / "link")
    assert (tmp_path / "link").read_text() == "b"


# ---------------------------------------------------------------------------
# POSCAR
# ---------------------------------------------------------------------------

def test_poscar_blocks_keeps_repeated_species_separate():
    atoms = _atoms(["Si", "Si", "H", "O", "C", "H"])
    assert poscar_blocks(atoms) == [("Si", 2), ("H", 1), ("O", 1), ("C", 1), ("H", 1)]


def test_poscar_blocks_refuses_empty():
    with pytest.raises(VaspSetupError, match="empty Atoms"):
        poscar_blocks(_atoms([]))


def test_write_poscar_preserves_split_hydrogen_blocks(tmp_path):
    atoms = _atoms(["Si", "Si", "H", "O", "C", "H"])
    path = write_poscar(atoms, tmp_path / "POSCAR")
    blocks, selective = read_poscar_blocks(path)
    assert blocks == [("Si", 2), ("H", 1), ("O", 1), ("C", 1), ("H", 1)]
    assert selective is False


def test_write_poscar_drops_constraints_by_default(tmp_path):
    from ase.constraints import FixAtoms
    atoms = _atoms(["Si", "H"])
    atoms.set_constraint(FixAtoms(indices=[0]))
    write_poscar(atoms, tmp_path / "POSCAR")
    assert "Selective" not in (tmp_path / "POSCAR").read_text()
    write_poscar(atoms, tmp_path / "KEPT", keep_constraints=True)
    assert "Selective" in (tmp_path / "KEPT").read_text()


def test_read_poscar_blocks_vasp4_takes_names_from_comment(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text("Si  H  O\n1.0\n10 0 0\n0 10 0\n0 0 10\n  2  1  1\n"
                    "Selective dynamics\nCartesian\n0 0 0 F F F\n")
    blocks, selective = read_poscar_blocks(path)
    assert blocks == [("Si", 2), ("H", 1), ("O", 1)]
    assert selective is True


def test_read_poscar_blocks_vasp4_without_names_raises(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text("some run\n1.0\n10 0 0\n0 10 0\n0 0 10\n  2  1  1\n"
                    "Direct\n0 0 0\n")
    with pytest.raises(VaspSetupError, match="cannot be recovered"):
        read_poscar_blocks(path)


def test_read_poscar_blocks_count_mismatch_raises(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text("c\n1.0\n10 0 0\n0 10 0\n0 0 10\nSi H O\n2 1\nDirect\n0 0 0\n")
    with pytest.raises(VaspSetupError, match="species names but"):
        read_poscar_blocks(path)


def test_read_poscar_blocks_too_short_raises(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text("c\n1.0\n10 0 0\n")
    with pytest.raises(VaspSetupError, match="too short"):
        read_poscar_blocks(path)


# ---------------------------------------------------------------------------
# INCAR parsing, fingerprint, rendering, provenance
# ---------------------------------------------------------------------------

def test_parse_incar_handles_semicolons_and_both_comment_chars():
    tags = parse_incar(INCAR_TEMPLATE)
    assert tags["EDIFF"] == "1.00e-06"
    assert tags["NELM"] == "150"
    assert tags["ISMEAR"] == "0"
    assert tags["ENCUT"] == "400"


def test_parse_incar_last_assignment_wins():
    assert parse_incar("ENCUT = 400\nENCUT = 500\n")["ENCUT"] == "500"


def test_parse_incar_reads_a_path(template):
    assert parse_incar(template)["PREC"] == "Accurate"


def test_parse_incar_missing_path_raises(tmp_path):
    with pytest.raises(OSError):
        parse_incar(tmp_path / "does-not-exist")


def test_template_fingerprint_ignores_comments_but_not_values(tmp_path):
    base = template_fingerprint("ENCUT = 400 #cutoff\n")
    assert template_fingerprint("ENCUT = 400 #a different comment\n") == base
    assert template_fingerprint("ENCUT = 400\n") == base
    assert template_fingerprint("ENCUT = 500\n") != base


def test_render_incar_replaces_in_place_and_keeps_comment(template, tmp_path):
    """The template's own comment survives verbatim, spacing included."""
    out = tmp_path / "INCAR"
    changes = render_incar(template, out,
                           overrides={"ENCUT": ("500", "converged for this system")})
    assert changes == ["ENCUT: 400 -> 500"]
    text = out.read_text()
    assert "ENCUT = 500 #PW cutoff" in text
    assert parse_incar(out)["ENCUT"] == "500"


def test_render_incar_overriding_one_tag_keeps_its_line_mate(template, tmp_path):
    out = tmp_path / "INCAR"
    render_incar(template, out, overrides={"NELM": ("60", "cap the SCF cycle")})
    tags = parse_incar(out)
    assert tags["NELM"] == "60"
    assert tags["EDIFF"] == "1.00e-06"     # the tag sharing the line survives


def test_render_incar_appends_absent_tags(template, tmp_path):
    out = tmp_path / "INCAR"
    changes = render_incar(template, out,
                           overrides={"LDIPOL": (".TRUE.", "slab dipole correction")})
    assert "LDIPOL: (absent) -> .TRUE." in changes
    assert parse_incar(out)["LDIPOL"] == ".TRUE."
    assert "LDIPOL = .TRUE." in out.read_text().splitlines()[-1]


def test_render_incar_writes_readable_provenance(template, tmp_path):
    out = tmp_path / "INCAR"
    render_incar(template, out, overrides={"NSW": ("11", "eleven structures"),
                                           "IBRION": ("11", "interactive walk")})
    provenance = incar_provenance(out)
    assert provenance["template"] == "INCAR.template"
    assert provenance["overrides"] == ["IBRION", "NSW"]
    assert provenance["sha256"] == template_fingerprint(template)
    assert provenance["reasons"] == {"IBRION": "interactive walk",
                                     "NSW": "eleven structures"}


def test_render_incar_header_is_summary_then_one_reason_per_change(template, tmp_path):
    out = tmp_path / "INCAR"
    render_incar(template, out,
                 overrides={"NSW": ("11", "eleven structures"),
                            "IBRION": ("11", "interactive walk")},
                 extra_comment="job note")
    lines = out.read_text().splitlines()
    assert lines[0].startswith("# tools4vasp: template=INCAR.template")
    assert "overrides=IBRION,NSW" in lines[0]
    assert lines[1] == "# IBRION = 11 (was -1): interactive walk"
    assert lines[2] == "# NSW = 11 (was 0): eleven structures"
    assert lines[3] == "# job note"
    assert not lines[4].startswith("#")          # the template body starts here


def test_render_incar_refuses_an_override_without_a_reason(template, tmp_path):
    out = tmp_path / "INCAR"
    with pytest.raises(VaspSetupError, match="needs a one-line reason"):
        render_incar(template, out, overrides={"NSW": "11"})
    assert not out.exists()


def test_render_incar_refuses_a_multiline_reason(template, tmp_path):
    with pytest.raises(VaspSetupError, match="single line"):
        render_incar(template, tmp_path / "INCAR",
                     overrides={"NSW": ("11", "because\nof reasons")})


def test_normalise_overrides_rejects_a_malformed_pair():
    with pytest.raises(VaspSetupError, match=r"\(value, reason\)"):
        normalise_overrides({"NSW": ("11", "why", "extra")})


def test_an_override_matching_the_template_is_not_a_change(template, tmp_path):
    """The header lists changes; a no-op override would just be noise."""
    out = tmp_path / "INCAR"
    changes = render_incar(template, out,
                           overrides={"ENCUT": ("400", "matches the template")})
    assert changes == []
    text = out.read_text()
    assert "overrides=-" in text.splitlines()[0]
    assert "matches the template" not in text
    assert incar_provenance(out)["overrides"] == []
    assert len(text.splitlines()) == len(template.read_text().splitlines()) + 1


def test_incar_provenance_absent_returns_none(tmp_path):
    path = tmp_path / "INCAR"
    path.write_text("ENCUT = 400\n")
    assert incar_provenance(path) is None


def test_render_incar_does_not_stack_headers(template, tmp_path):
    """Re-rendering replaces the header instead of accumulating one per pass."""
    first, second = tmp_path / "INCAR", tmp_path / "INCAR2"
    render_incar(template, first, overrides={"ENCUT": ("500", "first pass")})
    render_incar(first, second, overrides={"ENCUT": ("600", "second pass")})
    text = second.read_text()
    assert text.count("tools4vasp:") == 1
    assert text.count("(was ") == 1
    assert "second pass" in text and "first pass" not in text


def test_render_incar_refuses_neb_template_for_single_point(tmp_path):
    neb = tmp_path / "INCAR.neb"
    neb.write_text(NEB_TEMPLATE)
    out = tmp_path / "INCAR"
    with pytest.raises(VaspSetupError, match="does not fit run type"):
        render_incar(neb, out, run_type="single_point")
    assert not out.exists()            # nothing written on refusal


def test_render_incar_missing_template_raises(tmp_path):
    with pytest.raises(VaspSetupError, match="template not found"):
        render_incar(tmp_path / "nope", tmp_path / "INCAR")


# ---------------------------------------------------------------------------
# run-type consistency
# ---------------------------------------------------------------------------

def test_forbidden_tags_finds_band_tags():
    tags = parse_incar(NEB_TEMPLATE)
    found = forbidden_tags(tags, ("neb",))
    assert any(f.startswith("IMAGES") for f in found)


def test_check_run_type_single_point_rejects_moving_tags():
    problems = check_run_type({"IBRION": "2", "NSW": "100"}, "single_point")
    assert any("IBRION" in p for p in problems)
    assert any("NSW" in p for p in problems)


def test_check_run_type_interactive_requires_ibrion_11_and_fixed_cell():
    problems = check_run_type(
        {"IBRION": "2", "INTERACTIVE": ".FALSE.", "ISIF": "3"}, "interactive")
    assert any("IBRION = 11" in p for p in problems)
    assert any("INTERACTIVE" in p for p in problems)
    assert any("ISIF" in p for p in problems)


def test_check_run_type_accepts_a_clean_single_point():
    assert check_run_type(parse_incar(INCAR_TEMPLATE), "single_point") == []


def test_check_run_type_unknown_type_raises():
    with pytest.raises(VaspSetupError, match="unknown run type"):
        check_run_type({}, "molecular_dynamics")


# ---------------------------------------------------------------------------
# POTCAR
# ---------------------------------------------------------------------------

def test_split_potcar_and_read_titels_round_trip(tmp_path):
    path = _potcar(tmp_path / "POTCAR", ["Si", "H", "O", "C", "H"])
    blocks = split_potcar(path)
    assert [el for el, _ in blocks] == ["Si", "H", "O", "C", "H"]
    assert read_titels(path) == [TITEL[el] for el in ["Si", "H", "O", "C", "H"]]


def test_split_potcar_without_delimiter_raises(tmp_path):
    path = tmp_path / "POTCAR"
    path.write_text("   TITEL  = PAW_PBE Si 05Jan2001\n")
    with pytest.raises(VaspSetupError, match="End of Dataset"):
        split_potcar(path)


def test_build_potcar_from_reference_follows_poscar_order(tmp_path):
    reference = _potcar(tmp_path / "ref_POTCAR", ["Si", "H", "O", "C", "H"])
    out = tmp_path / "POTCAR"
    blocks = [("O", 1), ("Si", 2), ("H", 3)]
    assert build_potcar_from_reference(blocks, reference, out) == ["O", "Si", "H"]
    assert read_titels(out) == [TITEL["O"], TITEL["Si"], TITEL["H"]]


def test_build_potcar_from_reference_missing_element_raises(tmp_path):
    reference = _potcar(tmp_path / "ref_POTCAR", ["Si", "H"])
    with pytest.raises(VaspSetupError, match="no block for"):
        build_potcar_from_reference([("O", 1)], reference, tmp_path / "POTCAR")


def test_build_potcar_from_reference_rejects_wrong_pseudopotential(tmp_path):
    reference = _potcar(tmp_path / "ref_POTCAR", ["Si"])
    with pytest.raises(VaspSetupError, match="provenance mismatch"):
        build_potcar_from_reference([("Si", 1)], reference, tmp_path / "POTCAR",
                                    expected_titels={"Si": "PAW_PBE Si_sv 2000"})


def test_build_potcar_from_pp_path_needs_the_env(tmp_path):
    (tmp_path / "POSCAR").write_text("c\n1.0\n10 0 0\n0 10 0\n0 0 10\nSi\n1\nDirect\n0 0 0\n")
    with patch.dict(os.environ, {}, clear=True), \
            pytest.raises(VaspSetupError, match="VASP_PP_PATH"):
        build_potcar_from_pp_path(tmp_path)


def test_build_potcar_from_pp_path_rejects_unknown_extension(tmp_path):
    (tmp_path / "POSCAR").write_text("c\n1.0\n10 0 0\n0 10 0\n0 0 10\nSi\n1\nDirect\n0 0 0\n")
    with pytest.raises(VaspSetupError, match="unknown POTCAR extension"):
        build_potcar_from_pp_path(tmp_path, extension="_nonsense")


def test_build_potcar_from_pp_path_delegates_to_the_group_script(tmp_path):
    (tmp_path / "POSCAR").write_text("c\n1.0\n10 0 0\n0 10 0\n0 0 10\nSi H\n1 1\nDirect\n0 0 0\n0 0 0\n")

    def fake_run(cmd, **kwargs):
        _potcar(tmp_path / "POTCAR", ["Si", "H"])
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch.dict(os.environ, {"VASP_PP_PATH": "/pp"}), \
            patch("tools4vasp.vaspsetup.subprocess.run", side_effect=fake_run) as run_mock:
        titels = build_potcar_from_pp_path(tmp_path)
    assert titels == [TITEL["Si"], TITEL["H"]]
    assert run_mock.call_args[0][0][0].endswith("getPOTCAR.sh")
    assert run_mock.call_args[0][0][1] == "-r"


def test_build_potcar_from_pp_path_detects_block_count_mismatch(tmp_path):
    (tmp_path / "POSCAR").write_text("c\n1.0\n10 0 0\n0 10 0\n0 0 10\nSi H\n1 1\nDirect\n0 0 0\n0 0 0\n")

    def fake_run(cmd, **kwargs):
        _potcar(tmp_path / "POTCAR", ["Si"])          # one block, two species
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch.dict(os.environ, {"VASP_PP_PATH": "/pp"}), \
            patch("tools4vasp.vaspsetup.subprocess.run", side_effect=fake_run), \
            pytest.raises(VaspSetupError, match="species blocks but"):
        build_potcar_from_pp_path(tmp_path)


def test_build_potcar_from_pp_path_propagates_script_failure(tmp_path):
    (tmp_path / "POSCAR").write_text("c\n1.0\n10 0 0\n0 10 0\n0 0 10\nSi\n1\nDirect\n0 0 0\n")
    failed = type("P", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()
    with patch.dict(os.environ, {"VASP_PP_PATH": "/pp"}), \
            patch("tools4vasp.vaspsetup.subprocess.run", return_value=failed), \
            pytest.raises(VaspSetupError, match="boom"):
        build_potcar_from_pp_path(tmp_path)


def test_link_potcar_is_relative(tmp_path):
    reference = _potcar(tmp_path / "POTCAR", ["Si"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    link_potcar(reference, run_dir)
    assert os.readlink(run_dir / "POTCAR") == os.path.join("..", "POTCAR")
    assert read_titels(run_dir / "POTCAR") == [TITEL["Si"]]


# ---------------------------------------------------------------------------
# interactive mode
# ---------------------------------------------------------------------------

def test_interactive_stdin_round_trip(tmp_path):
    structures = [_atoms(["Si", "H"]) for _ in range(4)]
    path = tmp_path / "POSCAR.interactive"
    assert write_interactive_stdin(structures, path) == 3
    assert count_interactive_structures(path) == 3


def test_interactive_stdin_needs_two_structures(tmp_path):
    with pytest.raises(VaspSetupError, match="at least 2 structures"):
        write_interactive_stdin([_atoms(["Si"])], tmp_path / "stdin")


def test_interactive_stdin_rejects_changing_atom_count(tmp_path):
    with pytest.raises(VaspSetupError, match="constant"):
        write_interactive_stdin([_atoms(["Si", "H"]), _atoms(["Si"])],
                                tmp_path / "stdin")


# ---------------------------------------------------------------------------
# job script
# ---------------------------------------------------------------------------

RUNSCRIPT = """\
#!/bin/bash
#SBATCH --job-name=placeholder
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=someone@example.org
#SBATCH --licenses=horse
#SBATCH -t 02:00:00
##SBATCH --licenses="SCRATCH"
STDIN=""
mpirun vasp_std
"""


def test_patch_runscript_replaces_and_sets_mode(tmp_path):
    template = tmp_path / "vasp.run"
    template.write_text(RUNSCRIPT)
    out = patch_runscript(template, tmp_path / "run" / "vasp.run",
                          {"#SBATCH --job-name=": "#SBATCH --job-name=abc",
                           "STDIN=": 'STDIN="POSCAR.interactive"'})
    text = out.read_text()
    assert "#SBATCH --job-name=abc" in text
    assert 'STDIN="POSCAR.interactive"' in text
    assert stat.S_IMODE(out.stat().st_mode) & stat.S_IXUSR


def test_patch_runscript_requires_a_site_flag_when_asked(tmp_path):
    template = tmp_path / "vasp.run"
    template.write_text(RUNSCRIPT.replace("#SBATCH --licenses=horse", "#other"))
    patch_runscript(template, tmp_path / "default.run")      # not required by default
    with pytest.raises(VaspSetupError, match=r"--licenses="):
        patch_runscript(template, tmp_path / "zih.run",
                        require=REQUIRED_SBATCH + SITE_SBATCH["zih"])


def test_patch_runscript_commented_directive_does_not_satisfy_requirement(tmp_path):
    """A foreign machine's commented block must not pass for an active line."""
    template = tmp_path / "vasp.run"
    template.write_text("#!/bin/bash\n#SBATCH --job-name=x\n#SBATCH --output=o\n"
                        '#SBATCH --mail-user=a@b\n##SBATCH --licenses="SCRATCH"\n')
    with pytest.raises(VaspSetupError, match=r"--licenses="):
        patch_runscript(template, tmp_path / "out",
                        require=REQUIRED_SBATCH + SITE_SBATCH["zih"])


def test_patch_runscript_rejects_ambiguous_prefix(tmp_path):
    template = tmp_path / "vasp.run"
    template.write_text(RUNSCRIPT + "STDIN=second\n")
    with pytest.raises(VaspSetupError, match="wrong number of"):
        patch_runscript(template, tmp_path / "out", {"STDIN=": 'STDIN="x"'})


def test_patch_runscript_missing_template_raises(tmp_path):
    with pytest.raises(VaspSetupError, match="template not found"):
        patch_runscript(tmp_path / "nope", tmp_path / "out")


# ---------------------------------------------------------------------------
# continuation runs
# ---------------------------------------------------------------------------

def _finished_run(tmp_path, outcar=OUTCAR_FINISHED, name="run1"):
    src = tmp_path / name
    src.mkdir()
    (src / "CONTCAR").write_text(
        "Si H\n1.0\n10 0 0\n0 10 0\n0 0 10\nSi H\n1 1\nDirect\n0 0 0\n0.1 0 0\n")
    (src / "INCAR").write_text(INCAR_TEMPLATE)
    (src / "KPOINTS").write_text("auto\n0\nGamma\n2 2 1\n0 0 0\n")
    _potcar(src / "POTCAR", ["Si", "H"])
    (src / "OUTCAR").write_text(outcar)
    now = 1_700_000_000
    os.utime(src / "CONTCAR", (now, now))
    os.utime(src / "OUTCAR", (now + 60, now + 60))
    return src


def test_continuation_dir_links_poscar_relatively(tmp_path):
    src = _finished_run(tmp_path)
    dest = tmp_path / "run2"
    result = continuation_dir(src, dest)
    link = os.readlink(dest / "POSCAR")
    assert not os.path.isabs(link)
    assert link.endswith(os.path.join("run1", "CONTCAR"))
    assert result["warnings"] == []
    assert (dest / "INCAR").exists()
    assert not os.path.isabs(os.readlink(dest / "POTCAR"))


def test_continuation_dir_follows_an_existing_potcar_symlink(tmp_path):
    """A chain of continuations must point at the one real POTCAR, not at links."""
    shared = _potcar(tmp_path / "POTCAR", ["Si", "H"])
    src = _finished_run(tmp_path)
    (src / "POTCAR").unlink()
    rel_symlink(shared, src / "POTCAR")
    dest = tmp_path / "run2"
    continuation_dir(src, dest)
    assert os.path.realpath(dest / "POTCAR") == os.path.realpath(shared)
    assert os.readlink(dest / "POTCAR") == os.path.join("..", "POTCAR")


def test_continuation_dir_refuses_a_running_source(tmp_path):
    src = _finished_run(tmp_path)
    now = 1_700_000_000
    os.utime(src / "CONTCAR", (now + 600, now + 600))     # newer than the OUTCAR
    with pytest.raises(VaspSetupError, match="still active"):
        continuation_dir(src, tmp_path / "run2")


def test_continuation_dir_refuses_an_unfinished_outcar(tmp_path):
    src = _finished_run(tmp_path, outcar=OUTCAR_RUNNING)
    with pytest.raises(VaspSetupError, match="final timing block"):
        continuation_dir(src, tmp_path / "run2")


def test_continuation_dir_warns_but_proceeds_on_unconverged_source(tmp_path):
    src = _finished_run(tmp_path, outcar=OUTCAR_UNCONVERGED)
    result = continuation_dir(src, tmp_path / "run2")
    assert any("unconverged" in w for w in result["warnings"])
    assert (tmp_path / "run2" / "POSCAR").exists()


def test_continuation_dir_needs_a_contcar(tmp_path):
    src = tmp_path / "run1"
    src.mkdir()
    with pytest.raises(VaspSetupError, match="no CONTCAR"):
        continuation_dir(src, tmp_path / "run2")


def test_continuation_dir_refuses_empty_contcar(tmp_path):
    src = tmp_path / "run1"
    src.mkdir()
    (src / "CONTCAR").write_text("")
    with pytest.raises(VaspSetupError, match="wrote no geometry"):
        continuation_dir(src, tmp_path / "run2")


def test_continuation_dir_applies_incar_overrides(tmp_path):
    src = _finished_run(tmp_path)
    dest = tmp_path / "run2"
    continuation_dir(src, dest, incar_overrides={
        "NSW": ("50", "continue the relaxation"),
        "IBRION": ("2", "conjugate gradient for the restart")})
    tags = parse_incar(dest / "INCAR")
    assert tags["NSW"] == "50" and tags["IBRION"] == "2"
    assert incar_provenance(dest / "INCAR")["overrides"] == ["IBRION", "NSW"]


# ---------------------------------------------------------------------------
# PAW potential suffixes (Copilot review on PR #30)
# ---------------------------------------------------------------------------

TITEL_KPV = "PAW_PBE K_pv 17Jan2003"
TITEL_TISV = "PAW_PBE Ti_sv 07Sep2000"
TITEL_TI = "PAW_PBE Ti 08Apr2002"


def test_element_of_titel_strips_paw_suffixes():
    assert element_of_titel(TITEL_KPV) == "K"
    assert element_of_titel(TITEL_TISV) == "Ti"
    assert element_of_titel("PAW_PBE Si 05Jan2001") == "Si"


def test_element_of_titel_rejects_a_malformed_line():
    with pytest.raises(VaspSetupError, match="cannot read an element"):
        element_of_titel("PAW_PBE")


def test_split_potcar_keys_blocks_by_the_bare_symbol(tmp_path):
    """A POSCAR species line has no suffix, so K_pv must key as K."""
    path = _potcar_titels(tmp_path / "POTCAR", [TITEL_KPV, TITEL_TISV])
    assert [el for el, _ in split_potcar(path)] == ["K", "Ti"]


def test_build_potcar_from_reference_accepts_suffixed_potentials(tmp_path):
    reference = _potcar_titels(tmp_path / "ref_POTCAR", [TITEL_KPV, TITEL_TISV])
    out = tmp_path / "POTCAR"
    assert build_potcar_from_reference([("K", 1), ("Ti", 2)], reference, out) == ["K", "Ti"]
    assert read_titels(out) == [TITEL_KPV, TITEL_TISV]


def test_build_potcar_from_reference_verifies_a_suffixed_titel(tmp_path):
    reference = _potcar_titels(tmp_path / "ref_POTCAR", [TITEL_TISV])
    build_potcar_from_reference([("Ti", 1)], reference, tmp_path / "ok",
                                expected_titels={"Ti": TITEL_TISV})
    with pytest.raises(VaspSetupError, match="provenance mismatch"):
        build_potcar_from_reference([("Ti", 1)], reference, tmp_path / "bad",
                                    expected_titels={"Ti": TITEL_TI})


def test_build_potcar_from_reference_refuses_an_ambiguous_element(tmp_path):
    """Ti and Ti_sv both key as Ti; a POSCAR cannot say which is meant."""
    reference = _potcar_titels(tmp_path / "ref_POTCAR", [TITEL_TI, TITEL_TISV])
    with pytest.raises(VaspSetupError, match=r"different\s+potentials for Ti"):
        build_potcar_from_reference([("Ti", 1)], reference, tmp_path / "POTCAR")


def test_expected_titels_disambiguates_an_ambiguous_element(tmp_path):
    reference = _potcar_titels(tmp_path / "ref_POTCAR", [TITEL_TI, TITEL_TISV])
    out = tmp_path / "POTCAR"
    build_potcar_from_reference([("Ti", 1)], reference, out,
                                expected_titels={"Ti": TITEL_TISV})
    assert read_titels(out) == [TITEL_TISV]
