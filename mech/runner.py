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
import inspect
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
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
           "viewer_base_url", "call_with_progress", "StudyProgress", "StepStatus", "OUTPUT_DIR"]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"  # <repo>/output: what the viewer and CLI use
_TICKS = 10  # non-terminal progress: one dot per 10 % of a stage


def call_with_progress(fn: Callable, *args, progress: Callable | None):
    """``fn(*args, progress=progress)`` when ``fn`` accepts a ``progress`` keyword (or ``**kw``),
    else ``fn(*args)`` — the analysis stages gain the callback independently of this module."""
    if progress is not None:
        try:
            params = inspect.signature(fn).parameters
            accepts = "progress" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            accepts = False
        if accepts:
            try:
                return fn(*args, progress=progress)
            except TypeError as exc:  # a signature that lies (C extension, decorator): fall through
                if "progress" not in str(exc):
                    raise
    return fn(*args)


class StudyProgress:
    """stderr progress of one study, fed ``progress(stage, done, total)`` by the solver and the
    clearance sweep (one call per frame): on a terminal a live ``study walk: 73 frames …
    clearance 12/73`` line rewritten in place, else one dot per 10 % of each stage on the one line
    per study. The finished line adds where the time went:
    ``study walk: 73 frames … solve ·········· clearance ·········· 96.2 s (solve 12.1 s,
    clearance 84.0 s, loads 0.1 s)``. ``begin(stage)`` starts timing a stage before its first callback (or for one
    that reports none, like the gravity loads)."""

    def __init__(self, name: str, frames: int, stream=None, clock: Callable[[], float] = time.perf_counter):
        self.stream = sys.stderr if stream is None else stream
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.head = f"study {name}: {frames} frames …"
        self.clock = clock
        self.stage: str | None = None
        self.shown: str | None = None  # the stage whose label the dots follow (non-terminal)
        self.times: dict[str, float] = {}
        self.ticks = 0
        self.width = 0
        self._since = clock()
        self._emit(self.head)

    def _emit(self, text: str, end: str = "") -> None:
        print(text, end=end, file=self.stream, flush=True)

    def _rewrite(self, text: str, end: str = "") -> None:
        pad = " " * max(0, self.width - len(text))
        self.width = len(text)
        self._emit(f"\r{text}{pad}", end)

    def begin(self, stage: str) -> None:
        """Close the running stage's time and start ``stage``'s."""
        if stage == self.stage:
            return
        now = self.clock()
        if self.stage is not None:
            self.times[self.stage] = self.times.get(self.stage, 0.0) + now - self._since
        self.stage, self._since, self.ticks = stage, now, 0

    def __call__(self, stage: str, done: int, total: int) -> None:
        self.begin(stage)
        total = max(int(total), 1)
        done = min(max(int(done), 0), total)
        if self.tty:
            self._rewrite(f"{self.head} {stage} {done}/{total}")
            return
        want = done * _TICKS // total
        if want > self.ticks:
            if self.shown != stage:
                self.shown = stage
                self._emit(f" {stage} ")
            self._emit("·" * (want - self.ticks))
            self.ticks = want

    def finish(self, seconds: float) -> None:
        self.begin("")  # closes the last stage
        self.times.pop("", None)
        split = ", ".join(f"{k} {v:.1f} s" for k, v in self.times.items() if k)
        text = f"{seconds:.1f} s" + (f" ({split})" if len(self.times) > 1 else "")
        if self.tty:
            self._rewrite(f"{self.head} {text}", "\n")
        else:
            self._emit(f" {text}", "\n")


class StepStatus:
    """stderr status of the steps around the studies (clearance setup, home pose): one
    ``setup … 20.3 s`` line per step; on a terminal with ``keep=False`` (``mech check``) the line
    is rewritten in place and cleared by ``close()``, so only the summary stays."""

    def __init__(self, title: str = "", show: bool = True, stream=None, keep: bool = True,
                 clock: Callable[[], float] = time.perf_counter):
        self.stream = sys.stderr if stream is None else stream
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.title, self.show, self.keep, self.clock = title, show, keep or not self.tty, clock
        self.width = 0

    def _emit(self, text: str, end: str = "") -> None:
        print(text, end=end, file=self.stream, flush=True)

    def _rewrite(self, text: str, end: str = "") -> None:
        self._emit(f"\r{text}{' ' * max(0, self.width - len(text))}", end)
        self.width = 0 if end else len(text)

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        if not self.show:
            yield
            return
        t0 = self.clock()
        text = f"{self.title}{label} …"
        self._rewrite(text) if self.tty else self._emit(text)
        try:
            yield
        finally:
            took = f" {self.clock() - t0:.1f} s"
            if not self.tty:
                self._emit(took, "\n")
            elif self.keep:
                self._rewrite(text + took, "\n")

    def close(self) -> None:
        """Clear a rewritten (``keep=False``) terminal line."""
        if self.show and self.tty and not self.keep and self.width:
            self._rewrite("")
            self._emit("\r")
            self.width = 0


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
    build() keyword values) is recorded in the report. ``progress`` prints the progress on stderr
    (None = only when stderr is a terminal): ``setup … 0.4 s``, then one line per study fed per
    frame by the solver and the clearance sweep (``study pan: 69 frames … solve ·········· clearance
    ·········· 41.0 s (solve 2.1 s, clearance 38.8 s, loads 0.1 s)``; live counts on a terminal,
    see ``StudyProgress``), then ``home pose clearance … 0.2 s``.
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
    steps = StepStatus(show=show)
    with steps.step("setup (clearance pairs, mass)"):
        checker = ClearanceChecker(asm)
        props = {name: part_props(part) for name, part in asm.parts.items()}
    results, sweeps, loads = {}, {}, {}
    for s in selected:
        meter = StudyProgress(s.name, s.frames) if show else None

        def stage(name: str) -> None:
            if meter is not None:
                meter.begin(name)

        t0 = time.perf_counter()
        stage("solve")
        results[s.name] = call_with_progress(run_study, asm, kin, s, progress=meter)
        stage("clearance")
        sweeps[s.name] = call_with_progress(checker.sweep, results[s.name], progress=meter)
        stage("loads")
        loads[s.name] = gravity_loads(asm, kin, results[s.name], props)
        if meter is not None:
            meter.finish(time.perf_counter() - t0)
    with steps.step("home pose clearance"):
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


def check(asm: Assembly, *, params=None, progress=None) -> dict:
    """``mech check``: validation, study checks, roles/mobility and the home-pose clearance only —
    no studies are run and nothing is exported. ``progress`` (None = when stderr is a terminal)
    shows what runs on stderr: on a terminal one status line rewritten in place and cleared at
    the end; with ``progress=True`` elsewhere (``mech check --verbose``) one line per step."""
    if params is not None:
        asm.params = dict(params)
    show = sys.stderr.isatty() if progress is None else bool(progress)
    steps = StepStatus(f"check {asm.name}: ", show=show, keep=False)
    try:
        with steps.step("validating"):
            try:
                kin = Kinematics(asm)
            except ModelError as exc:
                return invalid_report(asm.name, exc.errors, asm.params)
            selected, errors = select_studies(asm, kin)
        if errors:
            return invalid_report(asm.name, errors, asm.params)
        with steps.step("clearance setup"):
            checker = ClearanceChecker(asm)
        with steps.step("home pose clearance"):
            home = checker.check_pose({p: np.eye(4) for p in asm.parts})
        props = {name: part_props(part) for name, part in asm.parts.items()}
        return build_check_report(asm, kin, props, selected, home, roles=kin.roles_for(selected))
    finally:
        steps.close()


def run(asm: Assembly, **kw) -> dict:
    """``analyze`` + print the summary (``verbose=True`` lifts the 15-line cap). Returns the report."""
    verbose = bool(kw.pop("verbose", False))
    report = analyze(asm, **kw)
    ensure_utf8_stdio()
    print(format_summary(report, verbose=verbose))
    return report
