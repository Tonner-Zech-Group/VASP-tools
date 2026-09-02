#!/usr/bin/env python3
"""Build VASP input directories from a template directory.

This module is the *pre-run* counterpart to :mod:`tools4vasp.vaspcheck` and
:mod:`tools4vasp.outcar_convergence`: those two inspect a calculation after it
ran, this one assembles one so that the common setup mistakes cannot happen in
the first place. :mod:`tools4vasp.vasplint` validates a directory this module
(or a human) produced.

Design notes
------------
*There is no configuration file format.* The specification of a calculation is
a directory of ordinary VASP files -- an ``INCAR`` in normal INCAR syntax,
optionally a ``KPOINTS`` and a job script. Anything else would duplicate the
INCAR in a second syntax, and the duplicate would drift.

*Built INCARs describe themselves.* :func:`render_incar` prepends a provenance
comment naming the template, a fingerprint of its tag/value set and the tags
that were deliberately overridden. :mod:`tools4vasp.vasplint` reads that line
back and re-derives the INCAR, so an undeclared hand edit is detectable while a
declared deviation stays readable to a human.

*POTCARs are shared and symlinked.* One real POTCAR per distinct species order,
relative symlinks from the run directories, which is the layout
``bash_scripts/replace_potcar_symlinks.sh`` produces after the fact. Every
symlink written here is relative: absolute ones break as soon as the tree is
moved to another machine.

Errors raise :class:`VaspSetupError` rather than calling ``sys.exit`` so that
callers and tests can react to them.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

__all__ = [
    "PROVENANCE_PREFIX",
    "REQUIRED_SBATCH",
    "RUN_TYPE_FORBIDS",
    "SITE_SBATCH",
    "TAG_FAMILIES",
    "VaspSetupError",
    "build_potcar_from_pp_path",
    "build_potcar_from_reference",
    "check_run_type",
    "continuation_dir",
    "forbidden_tags",
    "incar_provenance",
    "link_potcar",
    "normalise_overrides",
    "parse_incar",
    "patch_runscript",
    "poscar_blocks",
    "read_poscar_blocks",
    "read_titels",
    "rel_symlink",
    "render_incar",
    "split_potcar",
    "template_fingerprint",
    "write_interactive_stdin",
    "write_poscar",
]


class VaspSetupError(RuntimeError):
    """A calculation could not be built correctly, so nothing was written."""


# ── tag families ────────────────────────────────────────────────────────────
#: Tags that belong to a transition-state search rather than to an ordinary
#: energy or relaxation run. Left in an INCAR, VASP with the VTST patch runs a
#: band or dimer search and the energies mean something else entirely. Grouped
#: by family so a run type can forbid whole groups instead of listing tags.
TAG_FAMILIES = {
    "neb": ("IMAGES", "SPRING", "LCLIMB", "LNEBCELL"),
    "dimer": ("DDR", "DROTMAX", "DFNMAX", "DFNMIN"),
    "vtst_optimizer": ("ICHAIN", "IOPT", "MAXMOVE", "ILBFGSMEM", "LGLOBAL",
                       "LAUTOSCALE", "INVCURV", "LLINEOPT", "FDSTEP"),
}

#: Which families each run type refuses. ``neb`` and ``dimer`` runs of course
#: forbid nothing.
RUN_TYPE_FORBIDS = {
    "single_point": ("neb", "dimer", "vtst_optimizer"),
    "interactive": ("neb", "dimer", "vtst_optimizer"),
    "relax": ("neb", "dimer"),
    "neb": (),
    "dimer": (),
}

#: SLURM directives a job script must carry as an *active* line, everywhere.
#: Deliberately short: this package is site-agnostic, so only directives whose
#: absence is a problem on any cluster belong here.
REQUIRED_SBATCH = ("--job-name=", "--output=", "--mail-user=")

#: Additional directives required by particular sites, to be passed to
#: :func:`patch_runscript` or ``vasplint --require`` where they apply. ZIH
#: (TU Dresden) needs ``--licenses``: it is the filesystem interlock, and
#: without it a job can start while its file system is in maintenance and lose
#: the whole walltime. Nothing here is enforced by default.
SITE_SBATCH = {"zih": ("--licenses=",)}

PROVENANCE_PREFIX = "# tools4vasp:"

_COMMENT_RE = re.compile(r"[#!]")
_PROVENANCE_RE = re.compile(
    r"^#\s*tools4vasp:\s*template=(?P<template>\S+)\s+"
    r"sha256=(?P<sha256>[0-9a-f]+)"
    r"(?:\s+overrides=(?P<overrides>\S*))?\s*$"
)
#: One reason line per change, written directly under the summary line:
#: ``# TAG = new (was old): reason``
_REASON_RE = re.compile(
    r"^#\s*(?P<tag>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*"
    r"\(was\s+(?P<old>[^)]*)\):\s*(?P<reason>.+?)\s*$"
)
#: extension -> getPOTCAR.sh flag. ``None`` means "recommended defaults".
_PP_FLAG = {
    None: "-r", "": "-n", "_sv": "-s", "_pv": "-p", "_d": "-d", "_2": "-b",
    "_3": "-c", "_s": "-S", "_h": "-h", "_GW": "-g", "_sv_GW": "-G",
}


# ── symlinks ────────────────────────────────────────────────────────────────
def rel_symlink(target, link_path) -> str:
    """Create ``link_path`` as a *relative* symlink to ``target``.

    Absolute symlinks break the moment the tree is copied to another machine,
    which for VASP work is the normal case (workstation -> cluster -> archive).
    Returns the relative path that was written.
    """
    target, link_path = Path(target), Path(link_path)
    if not target.exists():
        raise VaspSetupError(f"symlink target does not exist: {target}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    rel = os.path.relpath(target.resolve(), link_path.parent.resolve())
    link_path.symlink_to(rel)
    return rel


# ── POSCAR ──────────────────────────────────────────────────────────────────
def poscar_blocks(atoms) -> list[tuple[str, int]]:
    """Run-length encode an Atoms object's symbols into POSCAR species blocks.

    Run-length rather than a set count: a POSCAR may legitimately list the same
    element twice (``Si H O C H``), and that order is what the POTCAR must
    follow.
    """
    symbols = list(atoms.get_chemical_symbols())
    if not symbols:
        raise VaspSetupError("cannot build POSCAR blocks from an empty Atoms object")
    blocks, current, n = [], symbols[0], 0
    for sym in symbols:
        if sym == current:
            n += 1
        else:
            blocks.append((current, n))
            current, n = sym, 1
    blocks.append((current, n))
    return blocks


def read_poscar_blocks(path) -> tuple[list[tuple[str, int]], bool]:
    """Species blocks and the selective-dynamics flag of a POSCAR on disk.

    Returns ``([(symbol, count), ...], selective_dynamics)``. Follows symlinks,
    so it also works on the ``POSCAR -> ../prev/CONTCAR`` layout used for
    continuation runs.

    Both file generations are handled, because real datasets contain both:

    * VASP 5 puts the species names on line 6 and the counts on line 7.
    * VASP 4 has only the counts on line 6. The names are then taken from the
      comment on line 1, which is where VASP's own tools and ASE put them; if
      line 1 does not carry exactly as many non-numeric tokens as there are
      counts, the species order genuinely cannot be recovered from the file and
      this raises rather than guessing.

    The selective-dynamics flag is read from whichever line follows the counts.
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    if len(lines) < 8:
        raise VaspSetupError(f"{path} is too short to be a POSCAR")

    def _all_int(tokens):
        return bool(tokens) and all(tok.lstrip("+-").isdigit() for tok in tokens)

    line6 = lines[5].split()
    if _all_int(line6):                     # VASP 4: counts on line 6
        counts, names, next_line = line6, lines[0].split(), 6
        if len(names) != len(counts):
            raise VaspSetupError(
                f"{path} is a VASP 4 POSCAR with {len(counts)} species counts, "
                f"but its comment line lists {len(names)} name(s) "
                f"({lines[0].strip()!r}); the species order cannot be recovered")
        if _all_int(names):
            raise VaspSetupError(
                f"{path} is a VASP 4 POSCAR and its comment line carries no "
                "species names; the species order cannot be recovered from the "
                "file alone (the POTCAR or OUTCAR has it)")
    else:                                    # VASP 5: names on line 6
        names, counts, next_line = line6, lines[6].split(), 7
        if len(names) != len(counts):
            raise VaspSetupError(
                f"{path}: {len(names)} species names but {len(counts)} counts "
                f"({lines[5].strip()!r} vs {lines[6].strip()!r})")
    try:
        blocks = [(n, int(c)) for n, c in zip(names, counts)]
    except ValueError as exc:
        raise VaspSetupError(f"{path}: non-integer species count: {exc}") from exc
    selective = lines[next_line].strip().lower().startswith("s")
    return blocks, selective


def write_poscar(atoms, path, keep_constraints=False):
    """Write a POSCAR preserving the given atom order.

    ``sort=False`` is essential: ASE's sort merges repeated species blocks, so a
    ``Si H O C H`` structure would come out as ``Si H O C`` with the two
    hydrogen blocks fused, silently breaking POSCAR/POTCAR alignment.

    Constraints are dropped unless ``keep_constraints`` is set. A stray
    ``Selective dynamics`` block is worse than useless in a fixed-geometry run,
    and ASE's VASP reader can pick constraints up from a neighbouring POSCAR, so
    emitting them unasked propagates them.
    """
    from ase.io import write

    atoms = atoms.copy()
    if not keep_constraints:
        atoms.set_constraint()
    write(str(path), atoms, format="vasp", direct=True, sort=False, vasp5=True)
    return Path(path)


# ── INCAR ───────────────────────────────────────────────────────────────────
def _split_code_comment(line: str) -> tuple[str, str]:
    """Split an INCAR line into its code part and its trailing comment."""
    match = _COMMENT_RE.search(line)
    if match:
        return line[:match.start()], line[match.start():]
    return line, ""


def _segment_tag(segment: str) -> str | None:
    """The tag a ``TAG = value`` segment assigns, upper-cased, or None."""
    if "=" not in segment:
        return None
    tag = segment.split("=", 1)[0].strip().upper()
    return tag or None


def parse_incar(source) -> dict[str, str]:
    """Tag -> value for an INCAR, given a path or its text.

    A :class:`~pathlib.Path`, or a string with neither a newline nor an ``=``,
    is read from disk and must exist; anything else is taken as INCAR text.

    Handles both INCAR comment characters (``#`` and ``!``) and several tags on
    one line separated by ``;``. Tags are upper-cased; values are stripped but
    otherwise untouched. A tag assigned twice keeps its last value, which is
    what VASP itself does.
    """
    raw = str(source)
    if isinstance(source, Path) or ("\n" not in raw and "=" not in raw):
        text = Path(source).read_text(errors="replace")   # raises if absent
    else:
        text = raw
    tags: dict[str, str] = {}
    for line in text.splitlines():
        code, _ = _split_code_comment(line)
        for segment in code.split(";"):
            tag = _segment_tag(segment)
            if tag:
                tags[tag] = segment.split("=", 1)[1].strip()
    return tags


def template_fingerprint(source) -> str:
    """Short fingerprint of an INCAR's *meaning*, not of its bytes.

    Computed over the sorted, case-normalised tag/value pairs, so editing a
    comment in a template does not invalidate every INCAR built from it, while
    changing a value does. Twelve hex characters of SHA-256.
    """
    tags = source if isinstance(source, dict) else parse_incar(source)
    payload = "\n".join(f"{t}={tags[t].upper()}" for t in sorted(tags))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def forbidden_tags(tags: dict[str, str], families) -> list[str]:
    """Tags present in ``tags`` that belong to any of the named families."""
    banned = {t for fam in families for t in TAG_FAMILIES[fam]}
    return [f"{tag} = {value}" for tag, value in tags.items() if tag in banned]


def check_run_type(tags: dict[str, str], run_type: str) -> list[str]:
    """Complaints about an INCAR's tags for the intended ``run_type``.

    Returns a list of human-readable messages; empty means consistent. Covers
    the tag families a run type must not carry and the handful of tag
    combinations that define each run type.
    """
    if run_type not in RUN_TYPE_FORBIDS:
        raise VaspSetupError(
            f"unknown run type {run_type!r}; known: {sorted(RUN_TYPE_FORBIDS)}")
    problems = []
    offending = forbidden_tags(tags, RUN_TYPE_FORBIDS[run_type])
    if offending:
        problems.append(
            f"run type {run_type!r} must not carry transition-state tags, found: "
            + ", ".join(sorted(offending)))

    interactive = tags.get("INTERACTIVE", ".FALSE.").upper().startswith(".T")
    ibrion = tags.get("IBRION", "").strip()
    nsw = tags.get("NSW", "").strip()

    if run_type == "single_point":
        if ibrion not in ("-1", ""):
            problems.append(f"single point wants IBRION = -1, found {ibrion}")
        if nsw not in ("0", ""):
            problems.append(f"single point wants NSW = 0, found {nsw}")
        if interactive:
            problems.append("single point must not set INTERACTIVE = .TRUE.")
    elif run_type == "interactive":
        if ibrion != "11":
            problems.append(f"interactive mode wants IBRION = 11, found {ibrion or 'unset'}")
        if not interactive:
            problems.append("interactive mode wants INTERACTIVE = .TRUE.")
        isif = tags.get("ISIF", "2").strip()
        if isif.isdigit() and int(isif) >= 3:
            problems.append(
                f"interactive mode requires a fixed lattice (ISIF < 3), found ISIF = {isif}")
    return problems


def normalise_overrides(overrides) -> dict:
    """Validate overrides and split them into ``{TAG: (value, reason)}``.

    An override may be given as ``(value, reason)`` or, only when no reason is
    meaningful, as a bare value. A reason is **required**: the whole point of
    recording it is that a value deviating from the template is a decision, and
    an undocumented decision is the thing this module exists to prevent. Reasons
    must be a single line, because they are written as INCAR comments.
    """
    parsed, missing = {}, []
    for tag, spec in dict(overrides or {}).items():
        tag = tag.upper()
        if isinstance(spec, (tuple, list)):
            if len(spec) != 2:
                raise VaspSetupError(
                    f"override {tag} must be (value, reason), got {spec!r}")
            value, reason = str(spec[0]), str(spec[1]).strip()
        else:
            value, reason = str(spec), ""
        if not reason:
            missing.append(tag)
        if "\n" in reason or "\r" in reason:
            raise VaspSetupError(
                f"the reason for {tag} must be a single line, got {reason!r}")
        parsed[tag] = (value, reason)
    if missing:
        raise VaspSetupError(
            "every override needs a one-line reason, which is written into the "
            "INCAR header; missing for: " + ", ".join(sorted(missing))
            + "\n  pass overrides={\"TAG\": (value, \"why\")}")
    return parsed


def render_incar(template, out_path, overrides=None, run_type=None,
                 extra_comment=None):
    """Write ``out_path`` from ``template``, applying ``overrides``.

    ``overrides`` maps an upper-case tag to ``(value, reason)``. Every override
    needs a one-line reason; see :func:`normalise_overrides`.

    **The template's own comments are left exactly as they are.** A tag whose
    value is replaced keeps its trailing comment verbatim, including its
    spacing: an INCAR comment describes what a tag does, and rewriting it to
    explain one job's choice is what the header is for. A tag the template lacks
    is appended bare.

    The result is headed by a summary line naming the template, its fingerprint
    and every tag this render actually changed, followed by one line per change
    giving its reason. An override whose value already matches the template
    changed nothing and is therefore not listed:

    .. code-block:: none

        # tools4vasp: template=INCAR.template sha256=483da4794b03 overrides=IBRION,NSW
        # IBRION = 11 (was -1): interactive mode walks the series in one process
        # NSW = 11 (was 0): must be at least the number of structures

    If ``run_type`` is given the *result* is validated with
    :func:`check_run_type` and nothing is written when it does not fit. In
    particular a transition-state template is refused for a single point rather
    than quietly stripped: a template with those tags in it is the wrong
    template, and silently rewriting the caller's intent is worse than stopping.

    Returns the list of changes made, each a ``"TAG: old -> new"`` string.
    """
    template = Path(template)
    if not template.is_file():
        raise VaspSetupError(f"INCAR template not found: {template}")
    overrides = normalise_overrides(overrides)

    lines = template.read_text(errors="replace").splitlines()
    lines = [ln for ln in lines
             if not _PROVENANCE_RE.match(ln) and not _REASON_RE.match(ln)]
    changes, previous, changed = [], {}, set()

    for i, line in enumerate(lines):
        code, comment = _split_code_comment(line)
        segments, touched = code.split(";"), False
        for j, segment in enumerate(segments):
            tag = _segment_tag(segment)
            if tag is None or tag not in overrides:
                continue
            old = segment.split("=", 1)[1].strip()
            new = overrides[tag][0]
            previous[tag] = old
            if old != new:
                indent = segment[:len(segment) - len(segment.lstrip())]
                trailing = segment[len(segment.rstrip()):]
                segments[j] = f"{indent}{tag} = {new}{trailing}"
                changes.append(f"{tag}: {old} -> {new}")
                changed.add(tag)
                touched = True
        if touched:
            # `comment` still carries its own leading whitespace, so the
            # template's spacing survives untouched.
            lines[i] = ";".join(segments) + comment

    for tag, (value, _) in overrides.items():
        if tag not in previous:
            previous[tag] = "absent"
            lines.append(f"{tag} = {value}")
            changes.append(f"{tag}: (absent) -> {value}")
            changed.add(tag)

    result = "\n".join(lines) + "\n"
    if run_type is not None:
        problems = check_run_type(parse_incar(result), run_type)
        if problems:
            raise VaspSetupError(
                f"INCAR template {template} does not fit run type "
                f"{run_type!r}, nothing written:\n  " + "\n  ".join(problems)
                + "\nSupply a matching template (or comment the tags out); they "
                  "are deliberately NOT stripped automatically.")

    # Only tags this render actually changed are listed. An override that
    # matches the template changed nothing, so it is not a change to record:
    # saying so would fill the header with lines about values that are simply
    # the template's own. A reason is still required for every override, since
    # the caller cannot know in advance which ones will bite.
    header = [(
        f"{PROVENANCE_PREFIX} template={template.name} "
        f"sha256={template_fingerprint(template)} "
        f"overrides={','.join(sorted(changed)) if changed else '-'}"
    )]
    for tag in sorted(changed):
        value, reason = overrides[tag]
        header.append(f"# {tag} = {value} (was {previous[tag]}): {reason}")
    if extra_comment:
        header.append(f"# {extra_comment}")
    Path(out_path).write_text("\n".join(header) + "\n" + result)
    return changes


def incar_provenance(path) -> dict | None:
    """Read back the header :func:`render_incar` wrote.

    Returns ``{"template", "sha256", "overrides": [tags], "reasons": {tag: str}}``
    or None if the INCAR carries no summary line, which means it was not built
    by this module.
    """
    found, reasons = None, {}
    for line in Path(path).read_text(errors="replace").splitlines():
        match = _PROVENANCE_RE.match(line)
        if match and found is None:
            raw = match.group("overrides") or ""
            found = {"template": match.group("template"),
                     "sha256": match.group("sha256"),
                     "overrides": [t for t in raw.split(",") if t and t != "-"]}
            continue
        reason = _REASON_RE.match(line)
        if reason:
            reasons[reason.group("tag").upper()] = reason.group("reason")
    if found is None:
        return None
    found["reasons"] = reasons
    return found


# ── POTCAR ──────────────────────────────────────────────────────────────────
def read_titels(potcar_path) -> list[str]:
    """Every ``TITEL`` string in a POTCAR, in file order. Follows symlinks."""
    titels = []
    with open(potcar_path, errors="replace") as fh:
        for line in fh:
            if "TITEL" in line:
                titels.append(line.split("=", 1)[1].strip())
    return titels


def split_potcar(potcar_path) -> list[tuple[str, str]]:
    """Split a concatenated POTCAR into ``[(element, text), ...]``.

    Blocks are delimited by the ``End of Dataset`` line that closes each
    dataset, so the split is exact rather than heuristic.
    """
    path = Path(potcar_path)
    if not path.is_file():
        raise VaspSetupError(f"POTCAR not found: {path}")
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    blocks, start = [], 0
    for i, line in enumerate(lines):
        if line.strip() == "End of Dataset":
            chunk = lines[start:i + 1]
            titels = [ln.split("=", 1)[1].strip() for ln in chunk if "TITEL" in ln]
            if len(titels) != 1:
                raise VaspSetupError(
                    f"{path}: block {len(blocks)} has {len(titels)} TITEL lines")
            blocks.append((titels[0].split()[1], "".join(chunk)))
            start = i + 1
    if not blocks:
        raise VaspSetupError(f"no 'End of Dataset' delimiters found in {path}")
    return blocks


def build_potcar_from_reference(blocks, reference, out_path, expected_titels=None):
    """Assemble a POTCAR whose block order follows ``blocks``, from ``reference``.

    ``blocks`` is :func:`poscar_blocks` output; its element sequence *including
    repeats* defines the required order. Taking the blocks from an existing
    reference POTCAR rather than from a pseudopotential library is how a new
    calculation is made provably identical to an existing dataset.

    ``expected_titels`` maps element -> TITEL and is verified if given.
    """
    wanted = [element for element, _ in blocks]
    pool: dict[str, str] = {}
    titel_of: dict[str, str] = {}
    for element, text in split_potcar(reference):
        pool.setdefault(element, text)
        titel_of.setdefault(element, read_titels_from_text(text))
    missing = [e for e in wanted if e not in pool]
    if missing:
        raise VaspSetupError(
            f"reference POTCAR {reference} has no block for {missing}; "
            f"it provides {sorted(pool)}")
    if expected_titels:
        wrong = {e: (titel_of[e], expected_titels.get(e))
                 for e in set(wanted)
                 if expected_titels.get(e) not in (None, titel_of[e])}
        if wrong:
            detail = "\n  ".join(
                f"{e}: found {found!r}, expected {want!r}" for e, (found, want) in wrong.items())
            raise VaspSetupError(f"POTCAR provenance mismatch:\n  {detail}")
    Path(out_path).write_text("".join(pool[e] for e in wanted))
    got, want = read_titels(out_path), [titel_of[e] for e in wanted]
    if got != want:
        raise VaspSetupError(
            f"POTCAR assembly check failed for {out_path}:\n  got  {got}\n  want {want}")
    return wanted


def read_titels_from_text(text: str) -> str:
    """The single TITEL string inside one POTCAR block's text."""
    for line in text.splitlines():
        if "TITEL" in line:
            return line.split("=", 1)[1].strip()
    raise VaspSetupError("POTCAR block carries no TITEL line")


def build_potcar_from_pp_path(run_dir, extension=None, timeout=120):
    """Build ``run_dir/POTCAR`` from ``VASP_PP_PATH`` for the POSCAR there.

    Delegates to ``bash_scripts/getPOTCAR.sh`` instead of reimplementing the
    element/extension table: that table encodes which pseudopotential the group
    considers recommended per element, and two copies of it would drift.

    ``extension`` selects the flavour (``""``, ``"_sv"``, ``"_pv"``, ...);
    ``None`` means the script's recommended defaults.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "POSCAR").exists():
        raise VaspSetupError(f"{run_dir} has no POSCAR; getPOTCAR.sh needs one")
    if extension not in _PP_FLAG:
        raise VaspSetupError(
            f"unknown POTCAR extension {extension!r}; known: {sorted(k for k in _PP_FLAG if k)}")
    if not os.environ.get("VASP_PP_PATH"):
        raise VaspSetupError(
            "VASP_PP_PATH is not set, so no pseudopotential library can be found")
    script = Path(__file__).parent / "bash_scripts" / "getPOTCAR.sh"
    proc = subprocess.run([str(script), _PP_FLAG[extension]], cwd=str(run_dir),
                          capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise VaspSetupError(
            f"getPOTCAR.sh failed in {run_dir} (exit {proc.returncode}):\n"
            f"{proc.stdout}{proc.stderr}")
    blocks, _ = read_poscar_blocks(run_dir / "POSCAR")
    titels = read_titels(run_dir / "POTCAR")
    if len(titels) != len(blocks):
        raise VaspSetupError(
            f"{run_dir}: POSCAR has {len(blocks)} species blocks but the built "
            f"POTCAR has {len(titels)} datasets")
    return titels


def link_potcar(reference, run_dir, name="POTCAR") -> str:
    """Point ``run_dir/POTCAR`` at a shared reference POTCAR, relatively.

    This is the layout ``replace_potcar_symlinks.sh`` produces after the fact:
    one real POTCAR per distinct species order, symlinks everywhere else. Doing
    it at build time saves the megabytes per run directory and makes it obvious
    that sibling runs share pseudopotentials.
    """
    return rel_symlink(reference, Path(run_dir) / name)


# ── interactive mode ────────────────────────────────────────────────────────
def write_interactive_stdin(atoms_list, path):
    """Write the stdin file for ``IBRION = 11``: structures 2..N.

    Per the VASP wiki each structure is given in *fractional* coordinates
    (Cartesian is not supported), blank-line separated. POSCAR supplies
    structure 1, so only ``atoms_list[1:]`` goes here, and the caller must set
    ``NSW >= len(atoms_list)``.
    """
    if len(atoms_list) < 2:
        raise VaspSetupError(
            f"interactive mode needs at least 2 structures, got {len(atoms_list)}")
    reference = atoms_list[0]
    with open(path, "w") as fh:
        for k, atoms in enumerate(atoms_list[1:], start=2):
            if len(atoms) != len(reference):
                raise VaspSetupError(
                    f"structure {k} has {len(atoms)} atoms, expected "
                    f"{len(reference)}; interactive mode requires a constant "
                    "atom count")
            fh.writelines(
                f"{pos[0]:19.16f} {pos[1]:19.16f} {pos[2]:19.16f}\n"
                for pos in atoms.get_scaled_positions(wrap=False))
            fh.write("\n")
    return len(atoms_list) - 1


def count_interactive_structures(path) -> int:
    """Number of structures in an interactive-mode stdin file.

    Structures are blank-line separated blocks of three-column coordinates, so
    the count is the number of non-empty blocks. Used to check ``NSW``, which
    must be at least this many plus one for the POSCAR.
    """
    text = Path(path).read_text(errors="replace")
    return sum(1 for block in re.split(r"\n\s*\n", text) if block.strip())


# ── job script ──────────────────────────────────────────────────────────────
def patch_runscript(template, out_path, replacements=None,
                    require=REQUIRED_SBATCH, mode=0o755):
    """Copy a job script, replacing whole lines by their prefix.

    ``replacements`` maps a line prefix (``"#SBATCH --job-name="``, ``"STDIN="``)
    to the full replacement line. Each prefix must match exactly once; zero
    matches or several mean the template is not the one the caller thinks it is,
    so nothing is written.

    ``require`` names SLURM flags that must appear as *active* directives.
    Matching is anchored to ``^#SBATCH``: these templates carry commented-out
    blocks for other machines, and a substring test would happily accept
    ``##SBATCH --licenses="SCRATCH"`` from a foreign block.
    """
    template = Path(template)
    if not template.is_file():
        raise VaspSetupError(f"job script template not found: {template}")
    text = template.read_text(errors="replace")

    for flag in require or ():
        pattern = re.compile(rf"^#SBATCH\s+{re.escape(flag)}")
        if not any(pattern.match(ln.strip()) for ln in text.splitlines()):
            raise VaspSetupError(
                f"job script template {template} has no active "
                f"'#SBATCH {flag}' line")

    replacements = dict(replacements or {})
    counts = {prefix: 0 for prefix in replacements}
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        for prefix, replacement in replacements.items():
            if stripped.startswith(prefix):
                line = replacement
                counts[prefix] += 1
                break
        out.append(line)
    wrong = {p: c for p, c in counts.items() if c != 1}
    if wrong:
        raise VaspSetupError(
            f"job script surgery on {template} matched the wrong number of "
            f"lines: {wrong} (each prefix must match exactly once)")

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n")
    dest.chmod(mode)
    return dest


# ── continuation runs ───────────────────────────────────────────────────────
def continuation_dir(src_dir, dest_dir, incar_overrides=None, run_type=None,
                     link_names=("POTCAR", "KPOINTS")):
    """Set up a restart directory whose POSCAR is a symlink to a CONTCAR.

    ``dest_dir/POSCAR`` becomes a relative symlink to ``src_dir/CONTCAR``, the
    INCAR is carried forward from ``src_dir`` with ``incar_overrides`` applied,
    and each name in ``link_names`` is linked relatively -- following the source
    link when it is itself a symlink, so a chain of continuations all point at
    the one real POTCAR instead of at each other.

    Two guards, because a CONTCAR is a live file while its job runs:

    * a CONTCAR newer than its OUTCAR, or an OUTCAR without a final timing
      block, means the source is still running or died mid-write. Refused: a
      half-written CONTCAR is a truncated geometry.
    * a source whose ionic relaxation never reached the required accuracy is
      reported as a warning, not refused. Continuing from an unconverged
      geometry is normal, but it must be said rather than discovered later.

    Returns ``{"created": [...], "warnings": [...]}``.
    """
    src, dest = Path(src_dir), Path(dest_dir)
    contcar = src / "CONTCAR"
    if not contcar.is_file():
        raise VaspSetupError(f"no CONTCAR in {src}, nothing to continue from")
    if contcar.stat().st_size == 0:
        raise VaspSetupError(f"{contcar} is empty; the source run wrote no geometry")

    from tools4vasp.outcar_convergence import _find_outcar, _read_outcar_text

    warnings = []
    outcar = _find_outcar(str(src))
    if outcar is None:
        warnings.append(f"no OUTCAR in {src}: cannot tell whether the source run finished")
    else:
        if contcar.stat().st_mtime > Path(outcar).stat().st_mtime + 1:
            raise VaspSetupError(
                f"{contcar} is newer than {outcar}: the source run looks still "
                "active, so its CONTCAR may be half written")
        text = _read_outcar_text(outcar)
        if "General timing and accounting" not in text:
            raise VaspSetupError(
                f"{outcar} has no final timing block: the source run did not "
                "finish, so its CONTCAR may be half written")
        from tools4vasp.outcar_convergence import check_ionic_convergence
        if not check_ionic_convergence(outcar):
            warnings.append(
                f"the ionic relaxation in {src} did not reach the required "
                "accuracy; this continuation starts from an unconverged geometry")

    dest.mkdir(parents=True, exist_ok=True)
    created = [str(dest / "POSCAR")]
    rel_symlink(contcar, dest / "POSCAR")
    for name in link_names:
        source = src / name
        if not source.exists():
            warnings.append(f"{source} does not exist and was not linked")
            continue
        target = source.resolve() if source.is_symlink() else source
        rel_symlink(target, dest / name)
        created.append(str(dest / name))
    src_incar = src / "INCAR"
    if not src_incar.is_file():
        raise VaspSetupError(f"no INCAR in {src} to carry forward")
    render_incar(src_incar, dest / "INCAR", overrides=incar_overrides,
                 run_type=run_type,
                 extra_comment=f"continuation of {os.path.relpath(src.resolve(), dest.resolve())}")
    created.append(str(dest / "INCAR"))
    return {"created": created, "warnings": warnings}
