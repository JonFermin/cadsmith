"""The ``mech`` console script (spec §4.11).

    mech run   <script.py> [-p k=v ...] [--study NAME ...] [--frames N] [--no-export] [--step] [--verbose] [--json]
    mech check <script.py> [-p k=v ...]
    mech sweep <script.py> k=a:b:step k=v1,v2 ... [--frames N] [--export-best]
    mech shot  <name> [--issue i] [--view iso|top|…] [--ghost N] [--layout quad] [--frame N] [-o out.png]
    mech list

Exit codes: PASS 0, WARN 1, FAIL 2, INVALID 3 (also: script errors and bad arguments).
``mech shot`` / ``mech list`` exit 0 on success, 3 on error. A model script defines
``build(**params) -> Assembly`` with every tunable as a keyword default; it is imported by path
with its own directory and the repo root on ``sys.path``.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from .assembly import Assembly
from .geom import slug
from .runner import analyze, check, ensure_utf8_stdio
from .report import format_check, format_summary
from .shot import VIEWS, ShotError, shot
from .sweep import best_variant, format_table, grid, parse_axis, parse_value, run_sweep

__all__ = ["main", "EXIT_CODES"]

REPO_ROOT = Path(__file__).resolve().parents[1]
EXIT_CODES = {"PASS": 0, "WARN": 1, "FAIL": 2, "INVALID": 3}
EXIT_ERROR = 3
_TRACE_LINES = 6  # a trimmed script traceback is at most this many lines


class CliError(Exception):
    """A user-facing error: print the message, exit 3."""


# ------------------------------------------------------------------------------ script loading


def _trimmed_traceback(exc: BaseException, script: Path, verbose: bool) -> str:
    """The script's own frames + the exception (≤ 6 lines); the full traceback with ``verbose``.

    Without a script frame (a failure inside mech itself) the innermost frame is shown instead;
    a SyntaxError already names its file and line.
    """
    if verbose:
        return "".join(traceback.format_exception(exc)).rstrip()
    target = os.path.normcase(str(script.resolve()))
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f for f in tb if os.path.normcase(os.path.abspath(f.filename)) == target]
    if not frames and not isinstance(exc, SyntaxError):
        frames = tb[-1:]
    body = []
    for f in frames:
        body.append(f'  File "{f.filename}", line {f.lineno}, in {f.name}')
        if f.line:
            body.append(f"    {f.line.strip()}")
    budget = _TRACE_LINES - 1  # the caller prints a header line
    tail = "".join(traceback.format_exception_only(exc)).rstrip().splitlines()[:budget]
    keep = budget - len(tail)  # the innermost script frames are kept
    body = body[len(body) - keep:] if keep > 0 else []
    return "\n".join(body + tail)


def _load_build(script: Path, verbose: bool) -> Callable[..., Assembly]:
    """Import ``script`` by path and return its ``build``; CliError with a trimmed traceback on failure."""
    script = script.resolve()
    if not script.is_file():
        raise CliError(f"no such script: {script}")
    for p in (str(REPO_ROOT), str(script.parent)):  # script dir ends up first
        if p not in sys.path:
            sys.path.insert(0, p)
    name = f"_mech_script_{slug(script.stem)}"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise CliError(f"can't import {script} (not a .py file?)")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling look the module up by name
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # the user's script: show its frames only
        raise CliError(f"error importing {script.name}:\n{_trimmed_traceback(exc, script, verbose)}") from None
    build = getattr(module, "build", None)
    if not callable(build):
        raise CliError(f"{script.name} defines no build() function (expected `def build(**params) -> Assembly`)")
    return build


def _check_keys(build: Callable, keys) -> dict:
    """build()'s keyword defaults; CliError naming the valid params if one of ``keys`` is unknown."""
    sig = inspect.signature(build)
    params = [n for n, p in sig.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    defaults = {n: sig.parameters[n].default for n in params
                if sig.parameters[n].default is not inspect.Parameter.empty}
    var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    unknown = [k for k in keys if k not in params]
    if unknown and not var_kw:
        valid = ", ".join(f"{n}={defaults[n]!r}" if n in defaults else n for n in params) or "(none)"
        plural = "s" if len(unknown) > 1 else ""
        raise CliError(f"unknown param{plural} {', '.join(unknown)} — build() accepts: {valid}")
    return defaults


def _parse_params(pairs: list[str]) -> dict:
    """``["k=v", ...]`` -> {k: literal_eval(v) or the string}."""
    out = {}
    for item in pairs:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise CliError(f"bad -p '{item}': expected key=value")
        out[key.strip()] = parse_value(value.strip())
    return out


def _build(build: Callable[..., Assembly], given: dict, script: Path, verbose: bool) -> tuple[Assembly, dict]:
    """(assembly, full params = defaults overlaid with ``given``)."""
    defaults = _check_keys(build, given)
    try:
        asm = build(**given)
    except Exception as exc:  # the user's build(): show its frames only
        raise CliError(f"error in {script.name} build():\n{_trimmed_traceback(exc, script, verbose)}") from None
    if not isinstance(asm, Assembly):
        raise CliError(f"build() must return a mech.Assembly (got {type(asm).__name__})")
    return asm, {**defaults, **given}


def _guarded(fn: Callable[[], dict], script: Path, verbose: bool) -> dict:
    """Run an analysis; an unexpected exception (e.g. from a drive callable) becomes a CliError."""
    try:
        return fn()
    except Exception as exc:
        raise CliError(f"analysis failed:\n{_trimmed_traceback(exc, script, verbose)}") from None


# ------------------------------------------------------------------------------ commands


def _cmd_run(args) -> int:
    script = Path(args.script)
    build = _load_build(script, args.verbose)
    asm, params = _build(build, _parse_params(args.p), script, args.verbose)
    report = _guarded(lambda: analyze(asm, studies=args.study, frames=args.frames, export=not args.no_export,
                                      out_root=args.output_dir, step=args.step, params=params), script, args.verbose)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False))
    else:
        print(format_summary(report, verbose=args.verbose))
    return EXIT_CODES[report["status"]]


def _cmd_check(args) -> int:
    script = Path(args.script)
    build = _load_build(script, args.verbose)
    asm, params = _build(build, _parse_params(args.p), script, args.verbose)
    report = _guarded(lambda: check(asm, params=params), script, args.verbose)
    print(format_check(report, verbose=args.verbose))
    return EXIT_CODES[report["status"]]


def _cmd_sweep(args) -> int:
    script = Path(args.script)
    build = _load_build(script, args.verbose)
    try:
        axes = [parse_axis(a) for a in args.axes]
        variants = grid(axes)
    except ValueError as exc:
        raise CliError(str(exc)) from None
    defaults = _check_keys(build, [k for k, _ in axes])
    rows = _guarded(lambda: run_sweep(build, variants, base=defaults, frames=args.frames, out_root=args.output_dir),
                    script, args.verbose)
    print(f"mech sweep {script.name} — {len(rows)} variant{'s' if len(rows) != 1 else ''}")
    print(format_table(rows))
    best = best_variant(rows)
    chosen = " ".join(f"{k}={v!r}" for k, v in best.params.items())
    if args.export_best and best.status != "INVALID":
        asm, params = _build(build, best.params, script, args.verbose)
        report = _guarded(lambda: analyze(asm, frames=args.frames, out_root=args.output_dir, params=params), script,
                          args.verbose)
        print(f"best {chosen} — {report['status']} · exported · view {report['viewer_url']}")
    else:
        print(f"best {chosen} — {best.status}" + ("" if args.export_best else " (--export-best to export it)"))
    return EXIT_CODES[best.status]


def _cmd_shot(args) -> int:
    try:
        path = shot(args.name, issue=args.issue, view=args.view, ghost=args.ghost, layout=args.layout,
                    frame=args.frame, out=args.o, output_dir=args.output_dir)
    except ShotError as exc:
        raise CliError(str(exc)) from None
    print(path)
    return 0


def _cmd_list(args) -> int:
    root = Path(args.output_dir)
    rows = []
    for d in root.glob("*.mech"):
        if not d.is_dir():
            continue
        report_path = d / "report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            status = report.get("status", "?")
            sevs = [i.get("severity") for i in report.get("issues", [])]
            detail = f"{sevs.count('FAIL')} FAIL · {sevs.count('WARN')} WARN"
            mtime = report_path.stat().st_mtime
        except (OSError, ValueError, AttributeError):
            status, detail, mtime = "?", "no readable report.json", d.stat().st_mtime
        rows.append((mtime, d.name[: -len(".mech")], status, detail))
    if not rows:
        print(f"no mechanisms in {root} — run `uv run mech run <script.py>`")
        return 0
    width = max(len(r[1]) for r in rows)
    for mtime, name, status, detail in sorted(rows, reverse=True):
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        print(f"{name.ljust(width)}  {status:<7}  {detail}  {stamp}")
    return 0


# ------------------------------------------------------------------------------ argument parsing


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors, which would read as FAIL; bad arguments exit 3."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def _parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="mech", epilog="exit codes: PASS 0, WARN 1, FAIL 2, INVALID/error 3",
                 description="Mechanism analysis: kinematics, clearances, mass properties, gravity loads, "
                             "targets, viewer export.")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{run,check,sweep,shot,list}")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output",
                       help="where <name>.mech/ folders live (default: <repo>/output)")

    def script_args(p: argparse.ArgumentParser, params: bool = True) -> None:
        p.add_argument("script", help="model script defining build(**params) -> Assembly")
        if params:
            p.add_argument("-p", action="extend", nargs="+", default=[], metavar="K=V",
                           help="build() parameter (Python literal, else string); repeatable")
        p.add_argument("--verbose", action="store_true", help="no line cap; full tracebacks")
        common(p)

    p = sub.add_parser("run", help="analyze a model script and print the summary")
    script_args(p)
    p.add_argument("--study", action="append", metavar="NAME", help="run only this study (repeatable)")
    p.add_argument("--frames", type=int, help="override every study's frame count")
    p.add_argument("--no-export", action="store_true", help="don't write output/<name>.mech/")
    p.add_argument("--step", action="store_true", help="also export assembly.step")
    p.add_argument("--json", action="store_true", help="print report.json instead of the summary")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("check", help="validate + roles/mobility + home-pose clearance (no studies)")
    script_args(p)
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("sweep", help="run a parameter grid (≤ 50 variants), one row per variant")
    script_args(p, params=False)
    p.add_argument("axes", nargs="+", metavar="K=A:B:STEP|K=V1,V2", help="grid axis")
    p.add_argument("--frames", type=int, help="override every study's frame count")
    p.add_argument("--export-best", action="store_true", help="export the best variant for the viewer")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("shot", help="headless screenshot of output/<name>.mech (needs playwright)")
    p.add_argument("name", help="mechanism name (output/<name>.mech)")
    p.add_argument("--issue", type=int, help="focus report.issues[i]")
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--ghost", type=int, metavar="N", help="N translucent poses of the study")
    p.add_argument("--layout", choices=("quad",))
    p.add_argument("--frame", type=int, metavar="N")
    p.add_argument("-o", metavar="OUT.png", help="output file (default output/<name>.mech/shot_*.png)")
    common(p)
    p.set_defaults(func=_cmd_shot)

    p = sub.add_parser("list", help="list output/*.mech with status")
    common(p)
    p.set_defaults(func=_cmd_list)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``mech`` console script; returns the exit code."""
    ensure_utf8_stdio()
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"mech: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
