"""The ``mech`` console script (spec §4.11).

    mech run   <script.py> [-p k=v ...] [--study NAME ...] [--frames N] [--no-export] [--step] [--verbose] [--json]
    mech check <script.py> [-p k=v ...]
    mech sweep <script.py> k=a:b:step k=v1,v2 ... [--frames N] [--export-best]
    mech shot  <name> [--issue i] [--study S] [--frame N|home] [--q j:v] [--view iso|top|…] [--cam az,el]
               [--zoom F] [--section x:10] [--focus P] [--isolate P] [--hide P] [--explode F] [--axes]
               [--ghost N] [--paths 0|1] [--layout quad] [--param k=v] [-o out.png]
    mech list

Exit codes: PASS 0, WARN 1, FAIL 2, INVALID 3 (also: script errors and bad arguments).
``mech shot`` / ``mech list`` exit 0 on success, 3 on error. A model script defines
``build(**params) -> Assembly`` with every tunable as a keyword default; it is imported by path
with its own directory and the repo root on ``sys.path``. ``--study``/``--frames`` make a run
partial: it never replaces the last full run's report.json/scene.json (see ``mech.runner``).
``mech shot`` takes every viewer URL parameter as a flag (``mech.shot.OPTIONS``; values that start
with ``-`` work as written: ``--section -x:10``, ``--cam -30,20``) plus raw ``--param k=v``.
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
from .export import ExportError
from .geom import slug
from .runner import analyze, check, ensure_utf8_stdio
from .report import format_check, format_summary
from .shot import OPTIONS, VIEWS, ShotError, shot
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
    """Run an analysis; an unexpected exception (e.g. from a drive callable) becomes a CliError,
    an export failure (a part's STL could not be written) a one-line one."""
    try:
        return fn()
    except ExportError as exc:
        raise CliError(f"export failed: {exc}") from None
    except Exception as exc:
        raise CliError(f"analysis failed:\n{_trimmed_traceback(exc, script, verbose)}") from None


# ------------------------------------------------------------------------------ commands


def _cmd_run(args) -> int:
    script = Path(args.script)
    build = _load_build(script, args.verbose)
    asm, params = _build(build, _parse_params(args.p), script, args.verbose)
    report = _guarded(lambda: analyze(asm, studies=args.study, frames=args.frames, export=not args.no_export,
                                      out_root=args.output_dir, step=args.step, params=params,
                                      progress=True if args.verbose else None), script, args.verbose)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False))
    else:
        print(format_summary(report, verbose=args.verbose))
    return EXIT_CODES[report["status"]]


def _cmd_check(args) -> int:
    script = Path(args.script)
    build = _load_build(script, args.verbose)
    asm, params = _build(build, _parse_params(args.p), script, args.verbose)
    report = _guarded(lambda: check(asm, params=params, progress=True if args.verbose else None), script,
                      args.verbose)
    print(format_check(report, verbose=args.verbose, command=_run_command(args)))
    return EXIT_CODES[report["status"]]


def _quoted(word: str) -> str:
    """``word`` as one shell word (double-quoted when it holds a space or a quote)."""
    if word and not any(c in word for c in " 	\"'"):
        return word
    return '"' + word.replace('"', '\\"') + '"'


def _run_command(args) -> str:
    """The ``uv run mech run`` command line for the script, -p params and --output-dir of a
    ``mech check`` invocation — its footer's next step, ready to copy."""
    words = ["uv", "run", "mech", "run", _quoted(str(args.script))]
    if args.p:
        words += ["-p", *(_quoted(kv) for kv in args.p)]
    if Path(args.output_dir).resolve() != (REPO_ROOT / "output").resolve():
        words += ["--output-dir", _quoted(str(args.output_dir))]
    return " ".join(words)


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
        # exported as a full run (declared frame counts), so the scene/report match `mech run`
        asm, params = _build(build, best.params, script, args.verbose)
        report = _guarded(lambda: analyze(asm, out_root=args.output_dir, params=params), script, args.verbose)
        full = " (full run at the declared frames)" if args.frames is not None else ""
        where = f"view {report['viewer_url']}" if not report.get("out_dir") else f"in {report['out_dir']}"
        print(f"best {chosen} — {report['status']} · exported{full} · {where}")
        return EXIT_CODES[report["status"]]
    print(f"best {chosen} — {best.status}" + ("" if args.export_best else " (--export-best to export it)"))
    return EXIT_CODES[best.status]


def _shot_options(args) -> tuple[dict, dict]:
    """(viewer options, raw ``--param k=v`` passthrough) from the parsed ``mech shot`` arguments."""
    options = {k: getattr(args, k, None) for k in OPTIONS}
    for key in ("q", "focus", "isolate", "hide"):  # repeatable comma lists
        if options.get(key):
            options[key] = ",".join(options[key])
    params = {}
    for item in args.param or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise CliError(f"bad --param '{item}': expected key=value")
        params[key.strip()] = value.strip()
    return options, params


def _cmd_shot(args) -> int:
    options, params = _shot_options(args)
    try:
        path = shot(args.name, out=args.o, output_dir=args.output_dir, params=params, **options)
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
        partial_only = not report_path.exists() and (d / "report.partial.json").exists()
        if partial_only:  # only --study/--frames runs so far: say so, never pass it off as the full run
            report_path = d / "report.partial.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            status = report.get("status", "?")
            sevs = [i.get("severity") for i in report.get("issues", [])]
            detail = f"{sevs.count('FAIL')} FAIL · {sevs.count('WARN')} WARN"
            detail += " · partial run only" if partial_only else ""
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
        p.add_argument("--verbose", action="store_true",
                       help="no line/width cap, sweep stats; per-frame progress on stderr; full tracebacks")
        common(p)

    p = sub.add_parser("run", help="analyze a model script and print the summary")
    script_args(p)
    p.add_argument("--study", action="append", metavar="NAME",
                   help="run only this study (repeatable); partial run: not exported")
    p.add_argument("--frames", type=int, help="override every study's frame count; partial run: not exported")
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
    p.add_argument("--export-best", action="store_true",
                   help="export the best variant for the viewer (a full run at the declared frames)")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("shot", help="headless screenshot of output/<name>.mech (needs playwright)",
                       description="Screenshot the viewer. Without options the report's targeted view is shot "
                                   "(first FAIL/WARN issue, else a ghosted quad overview); any option replaces "
                                   "that. The file name encodes the options: shot_<study>_f12_top.png.")
    p.add_argument("name", help="mechanism name (output/<name>.mech)")
    p.add_argument("--issue", type=int, metavar="I", help="focus report.issues[I]")
    p.add_argument("--study", metavar="NAME", help="the study to pose (default: the first)")
    p.add_argument("--frame", metavar="N|home", help="frame N of the chosen study, or 'home' for the drawn pose")
    p.add_argument("--q", action="append", metavar="JOINT:VALUE[,…]",
                   help="pose: the frame nearest these joint values (repeatable; not with --frame)")
    p.add_argument("--view", choices=VIEWS, help="world-plane view: front = the XZ plane, right = the YZ plane")
    p.add_argument("--cam", metavar="AZ,EL", help="camera azimuth (from +X toward +Y) and elevation, degrees")
    p.add_argument("--zoom", type=float, metavar="F", help="zoom factor on the auto-fit (2 = twice as close)")
    p.add_argument("--section", metavar="[-]x|y|z[:OFFSET]", help="section plane; -x keeps the other side")
    p.add_argument("--focus", action="append", metavar="PART[,…]", help="frame these parts (repeatable)")
    p.add_argument("--isolate", action="append", metavar="PART[,…]",
                   help="emphasise these parts: the others are drawn faint, as context (repeatable; --hide "
                        "removes parts)")
    p.add_argument("--hide", action="append", metavar="PART[,…]", help="hide these parts (repeatable)")
    p.add_argument("--explode", type=float, metavar="F", help="explode the parts apart (0…1 and beyond)")
    p.add_argument("--axes", nargs="?", const="1", metavar="0|1", help="draw the joint axes")
    p.add_argument("--ghost", type=int, metavar="N", help="N translucent poses of the study")
    p.add_argument("--paths", metavar="0|1", help="draw probe paths (paths=0 hides them)")
    p.add_argument("--layout", choices=("quad",))
    p.add_argument("--param", action="append", metavar="K=V", help="any other viewer URL parameter (repeatable)")
    p.add_argument("-o", metavar="OUT.png", help="output file (default output/<name>.mech/shot_<options>.png)")
    common(p)
    p.set_defaults(func=_cmd_shot)

    p = sub.add_parser("list", help="list output/*.mech with status")
    common(p)
    p.set_defaults(func=_cmd_list)
    return ap


_DASH_VALUES = ("--section", "--cam")  # values that may start with '-': `--section -x:10`, `--cam -30,20`


def _attach_dash_values(argv: list[str]) -> list[str]:
    """``--section -x:10`` → ``--section=-x:10`` (argparse would read ``-x:10`` as an option)."""
    out, i = [], 0
    while i < len(argv):
        token = argv[i]
        if token in _DASH_VALUES and i + 1 < len(argv) and argv[i + 1].startswith("-") and len(argv[i + 1]) > 1:
            out.append(f"{token}={argv[i + 1]}")
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``mech`` console script; returns the exit code."""
    ensure_utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(_attach_dash_values(argv))
    try:
        return args.func(args)
    except CliError as exc:
        print(f"mech: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
