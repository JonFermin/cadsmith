"""The analysis pipeline — the only orchestrator (spec §4.11).

``analyze`` runs, in order: Kinematics (ModelError -> INVALID report) -> studies (declared, else
§2.2 defaults) -> ``studies`` filter -> ``frames`` override -> ``check_study`` (any error ->
INVALID) -> run_study -> clearance sweep -> gravity loads -> home-pose clearance -> roles ->
build_report (Δprev against the previous report) -> evaluate_targets -> export.

A run narrowed by ``studies`` or ``frames`` is *partial*: the report says so (``partial``), its
targets never guess about skipped studies, and it never replaces ``report.json``/``scene.json``
(the last full run, which ``mech list``, ``mech shot``, the viewer and the next full run's Δprev
read); it writes ``report.partial.json`` instead.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .assembly import Assembly, ModelError, Study, _suggest
from .clearance import ClearanceChecker
from .export import clear_scene, export_scene, read_report, write_report
from .geom import slug
from .kinematics import Kinematics
from .massprops import part_props
from .motion import check_study, default_studies, run_study
from .report import attach_targets, build_check_report, build_report, format_summary, invalid_report
from .statics import gravity_loads
from .targets import evaluate_targets

__all__ = ["analyze", "run", "check", "select_studies", "available_studies", "ensure_utf8_stdio",
           "viewer_base_url", "OUTPUT_DIR"]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"  # <repo>/output: what the viewer and CLI use


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


def available_studies(asm: Assembly, kin: Kinematics) -> list[Study]:
    """The studies a full run analyzes: the declared ones, else the §2.2 defaults."""
    return list(asm.studies) or default_studies(asm, kin)


def select_studies(asm: Assembly, kin: Kinematics, studies: str | Iterable[str] | None = None,
                   frames: int | None = None) -> tuple[list[Study], list[str]]:
    """(studies to run, errors): declared studies (else §2.2 defaults), filtered by name, with the
    frame override applied, then ``check_study`` on each. Also rejects targets that name a study
    which doesn't exist."""
    available = available_studies(asm, kin)
    names = [s.name for s in available]
    errors = [f"target '{t.label}': unknown study '{t.study}'{_suggest(t.study, names)}"
              for t in asm.targets if t.study is not None and t.study not in names]
    selected = available
    if studies is not None:
        wanted = _names(studies)
        errors += [f"study '{w}' (requested): no such study{_suggest(w, names)}" for w in wanted if w not in names]
        selected = [s for s in available if s.name in wanted]
    if frames is not None:
        selected = [dataclasses.replace(s, frames=int(frames)) for s in selected]
    for study in selected:
        errors += check_study(kin, study)
    return selected, errors


def _names(studies: str | Iterable[str]) -> list[str]:
    return [studies] if isinstance(studies, str) else list(studies)


def _scope(report: dict | None) -> dict:
    return {s["name"]: s.get("frames") for s in (report or {}).get("studies") or []}


def analyze(asm: Assembly, *, studies=None, frames=None, export=True, out_root=None, step=False,
            viewer_base="http://localhost:3000", params=None, progress=None) -> dict:
    """Analyze ``asm`` and return its report dict (also written to ``out_root/<slug>.mech/`` when
    ``export``; ``out_root`` defaults to ``<repo>/output``, where the viewer and ``mech shot``
    look). Model problems never raise: they come back as status INVALID (whose report is still
    written, so ``mech list`` and the next Δprev see it).

    ``studies`` restricts the run to these study names and ``frames`` overrides every study's
    frame count — either makes the run partial (see the module docstring); ``params`` (the
    build() keyword values) is recorded in the report. ``progress`` prints one line per study on
    stderr (``study pan: 69 frames … 41.0 s``); None = only when stderr is a terminal.
    """
    out_root = OUTPUT_DIR if out_root is None else Path(out_root)
    if params is not None:
        asm.params = dict(params)
    partial_run = studies is not None or frames is not None
    prev = read_report(out_root, asm.name)  # the last full run, read before anything overwrites it
    prev_partial = read_report(out_root, asm.name, partial=True) if partial_run else None

    def invalid(errors: list[str], partial: dict | None = None) -> dict:
        report = invalid_report(asm.name, errors, asm.params, partial)
        if export:
            write_report(report, out_root, partial=partial_run)
            if not partial_run:  # the old scene no longer matches the model
                clear_scene(out_root, asm.name)
        return report

    request = {"studies": None if studies is None else _names(studies), "skipped": [],
               "frames": None if frames is None else int(frames)} if partial_run else None
    try:
        kin = Kinematics(asm)
    except ModelError as exc:
        return invalid(exc.errors, request)
    available = available_studies(asm, kin)
    selected, errors = select_studies(asm, kin, studies, frames)
    partial = None
    if partial_run:
        ran = [s.name for s in selected]
        partial = {"studies": ran, "skipped": [s.name for s in available if s.name not in ran],
                   "frames": None if frames is None else int(frames)}
    if errors:
        return invalid(errors, partial)
    if partial_run and _scope(prev_partial) == {s.name: s.frames for s in selected}:
        prev = prev_partial  # the same partial run as last time: compare with it in full

    show = sys.stderr.isatty() if progress is None else bool(progress)
    checker = ClearanceChecker(asm)
    props = {name: part_props(part) for name, part in asm.parts.items()}
    results, sweeps, loads = {}, {}, {}
    for s in selected:
        if show:
            print(f"study {s.name}: {s.frames} frames …", end="", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        results[s.name] = run_study(asm, kin, s)
        sweeps[s.name] = checker.sweep(results[s.name])
        loads[s.name] = gravity_loads(asm, kin, results[s.name], props)
        if show:
            print(f" {time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)
    home = checker.check_pose({p: np.eye(4) for p in asm.parts})
    roles = kin.roles_for(available)  # a joint driven by a skipped study is still a driver
    full_export = export and not partial_run
    viewer_url = viewer_base_url(viewer_base, asm.name) if full_export else None
    out_dir = None
    if full_export and out_root.resolve() != OUTPUT_DIR.resolve():
        out_dir = str(out_root.resolve())
    report = build_report(asm, kin, props, results, sweeps, loads, home, roles=roles, viewer_url=viewer_url,
                          prev=prev, partial=partial, out_dir=out_dir)
    attach_targets(report, evaluate_targets(asm, report, results), prev)
    if full_export:
        export_scene(asm, kin, props, results, sweeps, loads, report, out_root, roles=roles, step=step)
    elif export:
        write_report(report, out_root, partial=True)
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
