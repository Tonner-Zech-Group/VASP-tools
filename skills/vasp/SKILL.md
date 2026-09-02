---
name: vasp
description: >
  Set up, verify and continue VASP calculations with tools4vasp. Use this
  whenever a VASP calculation is being created, edited or submitted: writing
  POSCAR/POTCAR/KPOINTS/INCAR, building a batch of single points, restarting
  from a CONTCAR, or checking a run directory before it goes to the queue
  ("set up a VASP calculation", "make the inputs for ...", "continue this
  relaxation", "is this INCAR right?", "why did VASP not start?"). It also
  covers the pre-submission gate: never submit a directory that `vasplint`
  has not passed.
---

# VASP calculations with tools4vasp

Two modules do the work. `tools4vasp.vaspsetup` builds input directories;
`tools4vasp.vasplint` (CLI: `vasplint`) checks one before it is submitted. After
a run, `vaspcheck` and `vaspcheck-outcar` take over.

## Non-negotiable rules

These are enforced by the tools, so working through them is easier than working
around them.

1. **Never submit an unchecked directory.** `vasplint <dir>` must exit 0 first.
   On ZIH machines use `vasplint --site zih <dir>`, which additionally requires
   the `--licenses` filesystem interlock.
2. **Every symlink is relative.** Absolute links break the moment the tree is
   copied to a cluster or an archive. `vaspsetup.rel_symlink()` does it right;
   `vasplint` rejects absolute ones.
3. **POTCAR block order follows the POSCAR species blocks, repeats included.** A
   POSCAR listing `Si H O C H` needs five POTCAR datasets in that order. Never
   write a POSCAR with ASE's `sort=True`: it merges the two hydrogen blocks and
   silently misaligns every atom after the first mismatch.
4. **Do not hand-edit a built INCAR.** Change the template, or pass an override.
   `vasplint --template` compares an INCAR against its template and reports
   undeclared deviations.
5. **The template's comments are never rewritten, and every override needs a
   one-line reason.** A tag's comment says what the tag does and what its
   numbered options mean; it is not the place for one job's justification. That
   goes in the header, which `render_incar` writes: a summary line naming the
   template, its fingerprint and every changed tag, then one line per change
   giving its reason. `vasplint` errors when a declared override has no reason
   line, so an undocumented deviation cannot pass the gate.
6. **Transition-state tags are never stripped automatically.** If a template
   carries `IMAGES`, `ICHAIN`, `IOPT`, `DdR` and so on, it is a band or dimer
   template. `vaspsetup` refuses it for a single point instead of quietly
   rewriting your intent.
7. **One dispersion setting per comparison.** Mixing `IVDW` on and off, or a
   dipole correction on and off, inside one energy comparison is invalid. The
   linter warns when a dipole tag appears on only one side.

## Building a calculation

```python
from ase.io import read
from tools4vasp import vaspsetup as vs

atoms = read("structure.xyz")
blocks = vs.poscar_blocks(atoms)                 # [('Si', 96), ('H', 32), ...]

vs.write_poscar(atoms, run / "POSCAR")           # order preserving, no constraints

# POTCAR: either from the pseudopotential library ...
vs.build_potcar_from_pp_path(run)                # delegates to getPOTCAR.sh, honours VASP_PP_PATH
# ... or, to reproduce an existing dataset exactly, from its own POTCAR:
vs.build_potcar_from_reference(blocks, reference_potcar, batch / "POTCAR")
vs.link_potcar(batch / "POTCAR", run)            # relative symlink, one real file per batch

# Overrides are (value, reason). The reason is mandatory: it lands in the header.
vs.render_incar(template, run / "INCAR", run_type="single_point",
                overrides={"NSW": ("0", "single structure, so no ionic steps")})

vs.patch_runscript(job_template, run / "vasp.run",
                   {"#SBATCH --job-name=": "#SBATCH --job-name=my-run"},
                   require=vs.REQUIRED_SBATCH + vs.SITE_SBATCH["zih"])
```

For a batch, write **one** real POTCAR at the batch root and `link_potcar()`
from each run directory. That is the layout
`replace_potcar_symlinks.sh` produces after the fact, so produce it directly.

Interactive mode (`IBRION = 11`) walks several structures in one VASP process,
reusing the orbitals in memory:

```python
n = vs.write_interactive_stdin(structures, run / "POSCAR.interactive")
vs.render_incar(template, run / "INCAR", run_type="interactive", overrides={
    "IBRION": ("11", "one process walks the series, reusing the orbitals"),
    "INTERACTIVE": (".TRUE.", "structures 2..N are fed on stdin"),
    # +1 because the POSCAR is structure 1
    "NSW": (str(n + 1), f"the {n + 1} structures of this walk"),
})
```

`NSW` too small is the classic interactive-mode bug: VASP stops early and the
remaining structures are silently lost. `vasplint` checks it.

## What a built INCAR looks like

```
# tools4vasp: template=INCAR.template sha256=483da4794b03 overrides=IBRION,NSW
# IBRION = 11 (was -1): one process walks the series, reusing the orbitals
# NSW = 11 (was 0): the 11 structures of this walk
System = my-system
ENCUT = 400 # plane-wave cutoff in eV
NSW = 11 # maximum ionic steps
IBRION = 11 # -1=single point, 0=MD, 2=CG, 11=interactive
```

Line 1 is machine-readable and is what `vasplint --template` uses; the lines
under it are for whoever opens the file next. The tag comments are the
template's own, unchanged. Note the fingerprint covers the template's tag/value
pairs and not its bytes, so rewriting a template's comments does not invalidate
the INCARs already rendered from it.

## Continuing a calculation

```python
result = vs.continuation_dir(previous_run, new_run,
                             incar_overrides={"NSW": "200"})
```

`POSCAR` becomes a relative symlink to the previous `CONTCAR`, and `POTCAR` and
`KPOINTS` are linked to the same real files (following the source's own links, so
a chain of continuations does not become a chain of symlinks). It refuses to run
when the source looks still active, because a CONTCAR being written is a
truncated geometry, and it warns when the source's relaxation never converged.
Read `result["warnings"]` and report them; do not discard them.

## Checking, before and after

```bash
vasplint run_dir/                        # 0 = clean
vasplint --site zih --strict run_*/      # warnings count as failures too
vasplint --template templates/ run_dir/  # where to look for the template
vasplint --outcar run_dir/               # also: did the run compute what the INCAR says?
vasplint --json run_dir/                 # for scripts and hooks
```

Findings are errors or warnings, and any check whose input is missing says it was
skipped rather than passing quietly.

**Templates resolve in three places**, in order: the run directory itself, then
any `--template` directory (repeatable), then `$VASPLINT_TEMPLATES`
(colon-separated). The INCAR names its own template in its header; these only say
where to look. Dropping a copy of the template into the run directory therefore
makes the calculation verify anywhere, including on a compute node where the
project tree is not mounted. A declared template that cannot be found is
reported, never silently skipped.

**`--outcar` closes the gap a pre-submission check cannot cover.** VASP re-prints
the parameters it actually used, which is the only witness to an INCAR edited
after linting, a value VASP overrode, or a tag it silently ignored because it was
misspelled (`ENCUTT = 520` is accepted by every pre-run check there is). It also
makes the `KPAR` check exact, because the OUTCAR states how many irreducible
k-points the run really had, where a mesh product is only an upper bound.

Two things `--outcar` deliberately does not treat as mismatches, because VASP
resolves rather than echoes them: `ISTART` is a request that drops to 0 when
there is no WAVECAR, and `GGA = --` means the functional came from the POTCAR.
Tags VASP reports nowhere are listed as skipped rather than assumed to match.

`vaspcheck` and `vaspcheck-outcar` remain the tools for whether a finished run
*converged*; `vasplint --outcar` is about whether it computed the right thing.

## Templates

`templates/INCAR.template` next to this file is the group default, in ordinary INCAR
syntax: a single point, with the transition-state and hybrid blocks commented
out. Values marked **PER SYSTEM** in its comments are decisions, not defaults to
inherit silently: cutoff, dispersion, spin, smearing and the parallel layout all
depend on what you are calculating and on what you intend to compare against.

`templates/vasp.run` is a generic job script with the universally required
directives active and the per-machine blocks commented.

There is deliberately **no default KPOINTS**. A k-mesh is a per-system
convergence decision, and shipping one would invite inheriting it unexamined.
Take the mesh from the dataset you are comparing against, or converge it.
`vasplint` errors when a directory has neither a KPOINTS file nor `KSPACING`,
because VASP would otherwise quietly run at a single k-point.

To override the default, keep your own template directory in the project and pass
it to `render_incar()` and `vasplint --template`. Projects should do exactly
that: the group default has no business carrying one project's cutoff.
