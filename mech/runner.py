"""The analysis pipeline — the only orchestrator (spec §4.11).

``analyze`` runs, in order: Kinematics (ModelError -> INVALID report) -> studies (declared, else
§2.2 defaults) -> ``studies`` filter -> ``frames`` override -> ``check_study`` (any error ->
INVALID) -> run_study -> clearance sweep -> gravity loads -> home-pose clearance -> roles ->
build_report (Δprev against the previous report.json) -> evaluate_targets -> export.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .assembly import Assembly, ModelError, Study, _suggest
from .clearance import ClearanceChecker
from .export import export_scene, read_report, write_report
from .geom import slug
from .kinematics import Kinematics
from .massprops import part_props
from .motion import check_study, default_studies, run_study
from .report import attach_targets, build_check_report, build_report, format_summary, invalid_report
from .statics import gravity_loads
from .targets import evaluate_targets

__all__ = ["analyze", "run", "check", "select_studies", "ensure_utf8_stdio", "viewer_base_url"]


def ensure_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 (the summary uses —, …, °, Δ; Windows consoles default to cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # detached or already-closed stream: leave it alone
                pass


def viewer_base_url(viewer_base: str, name: str) -> str:
    """``<viewer_base>/mech.html?m=<slug>`` — the viewer page for an exported mechanism."""
    return f"{viewer_base.rstrip('/')}/mech.html?m={slug(name)}"


def select_studies(asm: Assembly, kin: Kinematics, studies: str | Iterable[str] | None = None,
                   frames: int | None = None) -> tuple[list[Study], list[str]]:
    """(studies to run, errors): declared studies (else §2.2 defaults), filtered by name, with the
    frame override applied, then ``check_study`` on each. Also rejects targets that name a study
    which doesn't exist."""
    available = list(asm.studies) or default_studies(asm, kin)
    names = [s.name for s in available]
    errors = [f"target '{t.label}': unknown study '{t.study}'{_suggest(t.study, names)}"
              for t in asm.targets if t.study is not None and t.study not in names]
    selected = available
    if studies is not None:
        wanted = [studies] if isinstance(studies, str) else list(studies)
        errors += [f"study '{w}' (requested): no such study{_suggest(w, names)}" for w in wanted if w not in names]
        selected = [s for s in available if s.name in wanted]
    if frames is not None:
        selected = [dataclasses.replace(s, frames=int(frames)) for s in selected]
    for study in selected:
        errors += check_study(kin, study)
    return selected, errors


def analyze(asm: Assembly, *, studies=None, frames=None, export=True, out_root=Path("output"), step=False,
            viewer_base="http://localhost:3000", params=None) -> dict:
    """Analyze ``asm`` and return its report dict (also written to ``out_root/<slug>.mech/`` when
    ``export``). Model problems never raise: they come back as status INVALID (whose
    report.json is still written, so ``mech list`` and the next Δprev see it).

    ``studies`` restricts the run to these study names; ``frames`` overrides every study's frame
    count; ``params`` (the build() keyword values) is recorded in the report.
    """
    out_root = Path(out_root)
    if params is not None:
        asm.params = dict(params)
    prev = read_report(out_root, asm.name)  # before anything overwrites it

    def invalid(errors: list[str]) -> dict:
        report = invalid_report(asm.name, errors, asm.params)
        if export:
            write_report(report, out_root)
        return report

    try:
        kin = Kinematics(asm)
    except ModelError as exc:
        return invalid(exc.errors)
    selected, errors = select_studies(asm, kin, studies, frames)
    if errors:
        return invalid(errors)

    results = {s.name: run_study(asm, kin, s) for s in selected}
    checker = ClearanceChecker(asm)
    sweeps = {name: checker.sweep(res) for name, res in results.items()}
    props = {name: part_props(part) for name, part in asm.parts.items()}
    loads = {name: gravity_loads(asm, kin, res, props) for name, res in results.items()}
    home = checker.check_pose({p: np.eye(4) for p in asm.parts})
    roles = kin.roles_for(selected)
    viewer_url = viewer_base_url(viewer_base, asm.name) if export else None
    report = build_report(asm, kin, props, results, sweeps, loads, home, roles=roles, viewer_url=viewer_url,
                          prev=prev)
    attach_targets(report, evaluate_targets(asm, report, results), prev)
    if export:
        export_scene(asm, kin, props, results, sweeps, loads, report, out_root, roles=roles, step=step)
    return report


def check(asm: Assembly, *, params=None) -> dict:
    """``mech check``: validation, study checks, roles/mobility and the home-pose clearance only —
    no studies are run and nothing is exported."""
    if params is not None:
        asm.params = dict(params)
    try:
        kin = Kinematics(asm)
    except ModelError as exc:
        return invalid_report(asm.name, exc.errors, asm.params)
    selected, errors = select_studies(asm, kin)
    if errors:
        return invalid_report(asm.name, errors, asm.params)
    home = ClearanceChecker(asm).check_pose({p: np.eye(4) for p in asm.parts})
    props = {name: part_props(part) for name, part in asm.parts.items()}
    return build_check_report(asm, kin, props, selected, home, roles=kin.roles_for(selected))


def run(asm: Assembly, **kw) -> dict:
    """``analyze`` + print the summary (``verbose=True`` lifts the 15-line cap). Returns the report."""
    verbose = bool(kw.pop("verbose", False))
    report = analyze(asm, **kw)
    ensure_utf8_stdio()
    print(format_summary(report, verbose=verbose))
    return report
