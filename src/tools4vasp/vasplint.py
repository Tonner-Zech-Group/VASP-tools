#!/usr/bin/env python3
"""Check a VASP input directory *before* it is submitted.

:mod:`tools4vasp.vaspcheck` and :mod:`tools4vasp.outcar_convergence` inspect a
calculation that already ran. This module inspects one that has not, which is
where a mistake is still free: a wrong POTCAR order, a transition-state tag left
in a single-point INCAR or a missing filesystem interlock costs nothing on disk
and a whole walltime once queued.

It is deliberately usable with no arguments (``vasplint``): the run type is
inferred from the INCAR, and every check that needs something the directory does
not have reports itself as skipped rather than silently passing. Findings carry
a level, ``error`` or ``warning``; the exit status is non-zero when any error was
found, or with ``--strict`` when any warning was.

Pass ``--template`` to also verify an INCAR against the template it was built
from, using the provenance comment :func:`tools4vasp.vaspsetup.render_incar`
writes into it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tools4vasp.vaspsetup import (
    REQUIRED_SBATCH,
    SITE_SBATCH,
    VaspSetupError,
    check_run_type,
    count_interactive_structures,
    element_of_titel,
    incar_provenance,
    parse_incar,
    read_poscar_blocks,
    read_titels,
    template_fingerprint,
)

__all__ = ["infer_run_type", "lint", "main", "run"]

#: Files that, if present, are read from disk by VASP and must therefore exist
#: when the INCAR says to read them.
_DIPOLE_TAGS = ("LDIPOL", "IDIPOL", "DIPOL", "EPSILON")


def _finding(check, level, message):
    return {"check": check, "level": level, "message": message}


def infer_run_type(tags: dict) -> str:
    """Guess the run type from an INCAR's tags.

    Order matters: an interactive run also has ``NSW > 0``, and a dimer search
    also carries optimiser tags, so the most specific marker wins.
    """
    if tags.get("INTERACTIVE", "").upper().startswith(".T"):
        return "interactive"
    if "IMAGES" in tags:
        return "neb"
    if tags.get("ICHAIN", "").strip() == "2":
        return "dimer"
    ibrion = tags.get("IBRION", "").strip()
    nsw = tags.get("NSW", "0").strip()
    if ibrion == "-1" or nsw in ("0", ""):
        return "single_point"
    return "relax"


def _parse_kpoints_mesh(path):
    """The automatic mesh of a KPOINTS file, or None if it is not automatic."""
    lines = Path(path).read_text(errors="replace").splitlines()
    if len(lines) < 4:
        return None
    if not lines[2].strip()[:1].upper() in ("G", "M", "A"):
        return None
    tokens = lines[3].split()
    if len(tokens) < 3 or not all(t.lstrip("+-").isdigit() for t in tokens[:3]):
        return None
    return tuple(int(t) for t in tokens[:3])


def _sbatch_value(text, flag):
    """The value of an *active* ``#SBATCH <flag>`` directive, or None."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        body = stripped[len("#SBATCH"):].strip()
        if body.startswith(flag):
            value = body[len(flag):].strip()
            return value.split("#")[0].strip().strip('"').strip("'")
    return None


def _find_job_script(path: Path):
    """The job script in a run directory, by content rather than by name."""
    for candidate in sorted(path.glob("*")):
        if not candidate.is_file():
            continue
        if candidate.name in ("INCAR", "POSCAR", "POTCAR", "KPOINTS", "CONTCAR"):
            continue
        try:
            head = candidate.read_text(errors="replace")[:4000]
        except OSError:
            continue
        if "#SBATCH" in head:
            return candidate
    return None


def lint(path=".", template=None, run_type=None, expected_titels=None,
         require=()):
    """Check the VASP input directory ``path``; return findings and context.

    Returns ``{"path", "run_type", "findings", "skipped", "errors", "warnings"}``.
    Nothing is written and nothing is fixed: this function only reports.

    ``require`` names extra SLURM directives the job script must carry on top of
    :data:`~tools4vasp.vaspsetup.REQUIRED_SBATCH`, for site rules this package
    does not impose on everyone (see
    :data:`~tools4vasp.vaspsetup.SITE_SBATCH`).
    """
    path = Path(path)
    if not path.is_dir():
        raise VaspSetupError(f"not a directory: {path}")
    findings, skipped = [], []
    template_dir = Path(template) if template else None

    # ── INCAR, and the run type everything else is judged against ───────────
    incar_path = path / "INCAR"
    if not incar_path.exists():
        findings.append(_finding(
            "incar", "error",
            "no INCAR in this directory. Note this tool checks directories that "
            "are about to run; for a finished calculation use vaspcheck and "
            "vaspcheck-outcar instead"))
        tags = {}
    else:
        tags = parse_incar(incar_path)
        if not tags:
            findings.append(_finding("incar", "error", "INCAR assigns no tags"))
    provenance = incar_provenance(incar_path) if incar_path.exists() else None
    if provenance is not None:
        declared = {tag.upper() for tag in provenance["overrides"]}
        unexplained = sorted(declared - set(provenance.get("reasons", {})))
        if unexplained:
            findings.append(_finding(
                "template", "error",
                f"{len(unexplained)} declared override(s) have no reason line in "
                f"the INCAR header: {', '.join(unexplained)}. Every deviation from "
                "the template is a decision and has to say why, on one line under "
                "the summary line"))

    detected = run_type or (infer_run_type(tags) if tags else None)
    if detected is None:
        skipped.append("run_type: no INCAR tags, so every run-type dependent "
                       "check is skipped rather than assumed")
    elif tags:
        for problem in check_run_type(tags, detected):
            findings.append(_finding("run_type", "error", problem))

    # ── POSCAR ──────────────────────────────────────────────────────────────
    blocks, selective = None, False
    poscar = path / "POSCAR"
    if not poscar.exists():
        findings.append(_finding("poscar", "error", "no POSCAR in this directory"))
    else:
        try:
            blocks, selective = read_poscar_blocks(poscar)
        except VaspSetupError as exc:
            findings.append(_finding("poscar", "error", str(exc)))
        if selective and detected in ("single_point", "interactive"):
            findings.append(_finding(
                "poscar", "error",
                f"POSCAR carries a Selective dynamics block, but a "
                f"{detected} run does not move the atoms it would constrain; "
                "ASE's VASP reader also propagates such constraints to "
                "neighbouring structures"))

    # ── POTCAR, symlinks resolved ───────────────────────────────────────────
    potcar = path / "POTCAR"
    if not potcar.exists():
        findings.append(_finding(
            "potcar", "error",
            "no POTCAR" + (" (dangling symlink)" if potcar.is_symlink() else "")))
    elif blocks is not None:
        titels = read_titels(potcar)
        wanted = [element for element, _ in blocks]
        # Bare symbols: a POSCAR species line has no PAW suffix, so comparing
        # "Ti_sv" against "Ti" would be a false mismatch.
        got = [element_of_titel(t) for t in titels]
        if got != wanted:
            findings.append(_finding(
                "potcar", "error",
                f"POTCAR element order {got} does not match the POSCAR species "
                f"blocks {wanted}; VASP would assign the wrong pseudopotential "
                "to every atom after the first mismatch"))
        if expected_titels:
            wrong = [f"{e}: found {t!r}, expected {expected_titels[e]!r}"
                     for e, t in zip(got, titels)
                     if e in expected_titels and expected_titels[e] != t]
            if wrong:
                findings.append(_finding(
                    "potcar_provenance", "error",
                    "POTCAR pseudopotentials differ from the expected set:\n    "
                    + "\n    ".join(wrong)))
    else:
        skipped.append("potcar: POSCAR species blocks unavailable")

    # ── every symlink relative and resolving ────────────────────────────────
    for entry in sorted(path.iterdir()):
        if not entry.is_symlink():
            continue
        target = os.readlink(entry)
        if os.path.isabs(target):
            findings.append(_finding(
                "symlinks", "error",
                f"{entry.name} is an absolute symlink to {target}; it breaks as "
                "soon as the tree is copied to another machine"))
        if not entry.exists():
            findings.append(_finding(
                "symlinks", "error", f"{entry.name} is a dangling symlink to {target}"))

    # ── continuation runs ───────────────────────────────────────────────────
    if poscar.is_symlink():
        resolved = Path(os.readlink(poscar))
        if resolved.name != "CONTCAR":
            findings.append(_finding(
                "continuation", "warning",
                f"POSCAR is a symlink to {resolved.name}, not to a CONTCAR"))
        source = poscar.resolve().parent
        outcar = source / "OUTCAR"
        if outcar.exists() and poscar.resolve().stat().st_mtime > outcar.stat().st_mtime + 1:
            findings.append(_finding(
                "continuation", "error",
                f"the CONTCAR this POSCAR points at is newer than {outcar}: the "
                "source run looks still active, so the geometry may be half written"))
        elif not outcar.exists():
            skipped.append("continuation: source directory has no OUTCAR to check")

    # ── interactive mode ────────────────────────────────────────────────────
    if detected == "interactive":
        stdin_candidates = [p for p in path.glob("*interactive*") if p.is_file()]
        if not stdin_candidates:
            findings.append(_finding(
                "interactive", "error",
                "interactive mode is on but no stdin structure file was found; "
                "VASP would walk only the POSCAR"))
        else:
            n_extra = count_interactive_structures(stdin_candidates[0])
            need = n_extra + 1
            nsw = tags.get("NSW", "").strip()
            if not nsw.isdigit():
                findings.append(_finding(
                    "interactive", "error", f"NSW is {nsw or 'unset'}, expected an integer"))
            elif int(nsw) < need:
                findings.append(_finding(
                    "interactive", "error",
                    f"NSW = {nsw} but {stdin_candidates[0].name} holds {n_extra} "
                    f"structures plus the POSCAR, so NSW must be at least {need}; "
                    "VASP stops early and the missing structures are silently lost"))

    # ── charge density that must exist ──────────────────────────────────────
    icharg = tags.get("ICHARG", "").strip()
    if icharg in ("1", "11") and not (path / "CHGCAR").exists():
        findings.append(_finding(
            "restart_files", "error",
            f"ICHARG = {icharg} reads the charge density from CHGCAR, which is "
            "not in this directory; VASP fails at startup"))

    # ── job script ──────────────────────────────────────────────────────────
    script = _find_job_script(path)
    if script is None:
        skipped.append("job_script: no file containing #SBATCH directives found")
    else:
        text = script.read_text(errors="replace")
        why = {"--licenses=": "on ZIH this is the filesystem interlock; without "
                             "it a job can start while its file system is down "
                             "and lose the whole walltime",
               "--mail-user=": "job notifications",
               "--output=": "a descriptive output file name",
               "--job-name=": "a descriptive job name"}
        for flag in tuple(REQUIRED_SBATCH) + tuple(require or ()):
            if _sbatch_value(text, flag) is None:
                reason = why.get(flag, "required for this site")
                findings.append(_finding(
                    "job_script", "error",
                    f"{script.name} has no active '#SBATCH {flag}' line ({reason})"))
        ntasks = _sbatch_value(text, "--ntasks-per-node=")
        nodes = _sbatch_value(text, "--nodes=") or "1"
        ncore, kpar = tags.get("NCORE", "1").strip(), tags.get("KPAR", "1").strip()
        if ntasks and ntasks.isdigit() and nodes.isdigit() and ncore.isdigit() and kpar.isdigit():
            total = int(nodes) * int(ntasks)
            group = int(ncore) * int(kpar)
            if group and total % group:
                findings.append(_finding(
                    "parallel_layout", "warning",
                    f"NCORE({ncore}) * KPAR({kpar}) = {group} does not divide the "
                    f"{total} ranks requested ({nodes} node(s) * {ntasks}); VASP "
                    "will redistribute and the layout is not the one intended"))
        else:
            skipped.append("parallel_layout: rank count or NCORE/KPAR not both known")

    # ── KPOINTS, and KPAR against it ────────────────────────────────────────
    kpoints = path / "KPOINTS"
    mesh = None
    if not kpoints.exists():
        if "KSPACING" not in tags:
            findings.append(_finding(
                "kpoints", "error",
                "no KPOINTS file and no KSPACING tag; VASP falls back to a "
                "single k-point, which is almost never what was intended"))
    else:
        mesh = _parse_kpoints_mesh(kpoints)
        if template_dir and (template_dir / "KPOINTS").exists():
            want = _parse_kpoints_mesh(template_dir / "KPOINTS")
            if mesh and want and mesh != want:
                findings.append(_finding(
                    "kpoints", "error",
                    f"k-mesh {mesh} differs from the template's {want}"))
            elif not (mesh and want):
                skipped.append("kpoints: not an automatic mesh, meshes not compared")
    if mesh:
        kpar = tags.get("KPAR", "1").strip()
        product = mesh[0] * mesh[1] * mesh[2]
        if kpar.isdigit() and int(kpar) > product:
            findings.append(_finding(
                "kpoints", "error",
                f"KPAR = {kpar} exceeds the {product} k-points the {mesh} mesh can "
                "produce. Note this is a necessary condition only: symmetry "
                "reduction can leave fewer k-points than the mesh product, so a "
                "KPAR below this bound may still be too large"))

    # ── INCAR against its template ──────────────────────────────────────────
    if tags and template_dir:
        if provenance is None:
            findings.append(_finding(
                "template", "warning",
                "this INCAR carries no tools4vasp provenance comment, so it cannot "
                "be checked against a template; it was not built by "
                "tools4vasp.vaspsetup"))
        else:
            tmpl_path = template_dir / provenance["template"]
            if not tmpl_path.exists():
                findings.append(_finding(
                    "template", "error",
                    f"the INCAR names template {provenance['template']!r}, which is "
                    f"not in {template_dir}"))
            else:
                tmpl_tags = parse_incar(tmpl_path)
                now = template_fingerprint(tmpl_tags)
                if now != provenance["sha256"]:
                    findings.append(_finding(
                        "template", "error",
                        f"the template {tmpl_path.name} has changed since this INCAR "
                        f"was built (fingerprint {now} vs {provenance['sha256']}), so "
                        "the two are no longer comparable"))
                declared = {t.upper() for t in provenance["overrides"]}
                differing = {t for t in set(tags) | set(tmpl_tags)
                             if tags.get(t) != tmpl_tags.get(t)}
                undeclared = sorted(differing - declared)
                if undeclared:
                    detail = ", ".join(
                        f"{t}: template {tmpl_tags.get(t, '(absent)')!r} vs INCAR "
                        f"{tags.get(t, '(absent)')!r}" for t in undeclared)
                    findings.append(_finding(
                        "template", "error",
                        f"{len(undeclared)} tag(s) differ from the template without "
                        f"being declared as overrides: {detail}"))
                # Compared against the template the INCAR actually names, not
                # against a file assumed to be called "INCAR": templates are
                # conventionally "INCAR.template", and hardcoding the name made
                # this check silently do nothing.
                for tag in _DIPOLE_TAGS:
                    if (tag in tags) != (tag in tmpl_tags):
                        where = "INCAR" if tag in tags else "template"
                        findings.append(_finding(
                            "dipole", "warning",
                            f"{tag} is set in the {where} only; a dipole correction "
                            "changes the energy zero, so mixing corrected and "
                            "uncorrected runs in one comparison is not valid"))
    elif tags:
        skipped.append("template: no --template given, INCAR not compared to one")

    errors = [f for f in findings if f["level"] == "error"]
    warnings = [f for f in findings if f["level"] == "warning"]
    return {"path": str(path), "run_type": detected or "unknown",
            "findings": findings,
            "skipped": skipped, "errors": len(errors), "warnings": len(warnings)}


def run(path=".", template=None, run_type=None, expected_titels=None,
        verbose=True, strict=False, require=()):
    """Lint ``path`` and print a human-readable report. Returns the lint dict."""
    result = lint(path, template=template, run_type=run_type,
                  expected_titels=expected_titels, require=require)
    if verbose:
        print(f"* vasplint {result['path']}  (run type: {result['run_type']})")
        for finding in result["findings"]:
            mark = "ERROR  " if finding["level"] == "error" else "WARNING"
            print(f"  {mark} [{finding['check']}] {finding['message']}")
        for note in result["skipped"]:
            print(f"  skipped {note}")
        if not result["findings"]:
            print("  all checks passed")
        print(f"  {result['errors']} error(s), {result['warnings']} warning(s)")
    result["ok"] = result["errors"] == 0 and not (strict and result["warnings"])
    return result


def main():
    """CLI entry point: vasplint [path ...]."""
    parser = argparse.ArgumentParser(
        description="Check VASP input directories before submitting them.",
        epilog="Example: vasplint --site zih --template templates/ --strict run_*/")
    parser.add_argument("paths", nargs="*", default=["."],
                        help="run directories to check (default: the current one)")
    parser.add_argument("--template", default=None,
                        help="directory holding the INCAR/KPOINTS templates to compare against")
    parser.add_argument("--run-type", default=None,
                        help="override the run type inferred from the INCAR")
    sites = "; ".join(f"{name}: {' '.join(flags)}"
                      for name, flags in sorted(SITE_SBATCH.items()))
    parser.add_argument("--site", choices=sorted(SITE_SBATCH),
                        help="also require the #SBATCH directives a known site "
                             f"needs ({sites})")
    parser.add_argument("--require", action="append", default=[], metavar="FLAG",
                        help="extra #SBATCH directive the job script must carry, "
                             "repeatable. Attach the value with '=' so argparse does "
                             "not read it as an option: --require=--licenses=")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as failures too")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of a report")
    args = parser.parse_args()
    require = list(args.require) + list(SITE_SBATCH.get(args.site, ()))

    results, failed = [], False
    for path in args.paths or ["."]:
        try:
            result = run(path, template=args.template, run_type=args.run_type,
                         verbose=not args.json, strict=args.strict,
                         require=require)
        except VaspSetupError as exc:
            result = {"path": str(path), "errors": 1, "warnings": 0, "ok": False,
                      "findings": [_finding("input", "error", str(exc))], "skipped": []}
            if not args.json:
                print(f"* vasplint {path}\n  ERROR   [input] {exc}")
        results.append(result)
        failed = failed or not result["ok"]
    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
