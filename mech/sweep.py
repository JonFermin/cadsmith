"""Variant sweeps (spec §4.11 ``mech sweep``): run a model over a parameter grid, one table row each.

Axes are ``k=a:b:step`` (inclusive numeric range) or ``k=v1,v2,...`` / ``k=v`` (literal values).
The grid is the product of the axes, capped at 50 variants. Variants are analyzed without
export; the table shows params | status | #FAIL/#WARN | min clearance | worst SF | each target.
"""

from __future__ import annotations

import ast
import itertools
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .assembly import Assembly
from .report import sig
from .runner import analyze

__all__ = ["MAX_VARIANTS", "MAX_ROWS", "Variant", "parse_value", "parse_axis", "grid", "run_sweep",
           "best_variant", "format_table"]

MAX_VARIANTS = 50
MAX_ROWS = 20
_STATUS_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2, "INVALID": 3}


def parse_value(text: str):
    """A ``-p``/axis value: a Python literal (``3``, ``2.5``, ``'x'``, ``(1, 2)``), else the string."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _num(text: str) -> float | int:
    v = parse_value(text)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"'{text}' is not a finite number")
    return v


def parse_axis(spec: str) -> tuple[str, list]:
    """``k=a:b:step`` -> (k, [a, a+step, ..., ≤ b]); ``k=v1,v2`` -> (k, [v1, v2]); ``k=v`` -> (k, [v]).

    Integer range bounds and step give ints; ``ValueError`` on malformed specs.
    """
    key, sep, rhs = spec.partition("=")
    key, rhs = key.strip(), rhs.strip()
    if not sep or not key or not rhs:
        raise ValueError(f"bad axis '{spec}': expected k=a:b:step or k=v1,v2,...")
    if rhs.count(":") == 2:
        try:
            a, b, step = (_num(x) for x in rhs.split(":"))
        except ValueError as exc:
            raise ValueError(f"bad axis '{spec}': {exc}") from None
        if step <= 0 or b < a:
            raise ValueError(f"bad axis '{spec}': need step > 0 and a ≤ b")
        n = math.floor((b - a) / step + 1e-9) + 1
        if n > MAX_VARIANTS:
            raise ValueError(f"axis '{key}' alone has {n} values (cap {MAX_VARIANTS})")
        if all(isinstance(x, int) for x in (a, b, step)):
            return key, [a + i * step for i in range(n)]
        return key, [round(a + i * step, 12) for i in range(n)]  # 12 digits: hide float drift (0.30000000000000004)
    return key, [parse_value(v.strip()) for v in rhs.split(",")]


def grid(axes: list[tuple[str, list]]) -> list[dict]:
    """Product of the axes as param dicts (first axis varies slowest); ValueError above 50 variants."""
    n = math.prod(len(values) for _, values in axes)
    if n > MAX_VARIANTS:
        sizes = " × ".join(f"{k}:{len(v)}" for k, v in axes)
        raise ValueError(f"{n} variants ({sizes}) exceeds the cap of {MAX_VARIANTS}; coarsen the steps")
    keys = [k for k, _ in axes]
    return [dict(zip(keys, combo)) for combo in itertools.product(*(values for _, values in axes))]


@dataclass
class Variant:
    params: dict  # the swept values
    status: str
    fails: int = 0
    warns: int = 0
    min_clearance: float | None = None
    worst_sf: float | None = None
    targets: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None  # build() raised
    report: dict | None = None


def _summarize(params: dict, report: dict) -> Variant:
    sevs = [i["severity"] for i in report["issues"]]
    clear = [s["min_clearance"]["value"] for s in report["studies"]
             if s.get("min_clearance") and s["min_clearance"].get("value") is not None]
    sfs = [e["sf"] for s in report["studies"] for e in (s.get("loads") or {}).values() if e.get("sf") is not None]
    return Variant(params, report["status"], sevs.count("FAIL"), sevs.count("WARN"), min(clear, default=None),
                   min(sfs, default=None), {t["label"]: t["value"] for t in report["targets"]}, None, report)


def run_sweep(build: Callable[..., Assembly], variants: list[dict], *, base: dict | None = None,
              frames: int | None = None, out_root: Path = Path("output")) -> list[Variant]:
    """Build and analyze (no export) every variant; a build() exception becomes an INVALID row.

    ``base`` holds the fixed build() keyword values; each variant's values override it.
    """
    rows = []
    for n, params in enumerate(variants, 1):
        if sys.stderr.isatty():
            print(f"\rvariant {n}/{len(variants)}", end="", file=sys.stderr, flush=True)
        full = {**(base or {}), **params}
        try:
            asm = build(**full)
        except Exception as exc:  # user code: one bad variant must not stop the sweep
            rows.append(Variant(params, "INVALID", 1, error=f"{type(exc).__name__}: {exc}"))
            continue
        report = analyze(asm, frames=frames, export=False, out_root=out_root, params=full)
        rows.append(_summarize(params, report))
    if sys.stderr.isatty():
        print("\r" + " " * 24 + "\r", end="", file=sys.stderr, flush=True)
    return rows


def _rank(v: Variant) -> tuple:
    clearance = v.min_clearance if v.min_clearance is not None else math.inf
    sf = v.worst_sf if v.worst_sf is not None else math.inf
    return (_STATUS_RANK.get(v.status, 3), v.fails, v.warns, -clearance, -sf)


def best_variant(rows: list[Variant]) -> Variant:
    """Best status, then fewest FAIL/WARN, then largest min clearance, then largest worst SF."""
    return min(rows, key=_rank)


def _cell(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return sig(v)
    return str(v)


def format_table(rows: list[Variant]) -> str:
    """Aligned table, grid order; above 20 variants only the 20 best are listed."""
    shown = rows if len(rows) <= MAX_ROWS else sorted(rows, key=_rank)[:MAX_ROWS]
    keys = list(rows[0].params) if rows else []
    labels = list(dict.fromkeys(label for r in rows for label in r.targets))
    head = keys + ["status", "F/W", "min clr", "worst SF"] + [lbl[:14] for lbl in labels]
    body = []
    for r in shown:
        cells = [_cell(r.params.get(k)) for k in keys] + [r.status, f"{r.fails}/{r.warns}", _cell(r.min_clearance),
                                                          _cell(r.worst_sf)]
        cells += [_cell(r.targets.get(lbl)) for lbl in labels]
        if r.error:
            cells.append(r.error[:60])
        body.append(cells)
    widths = [max(len(row[i]) for row in [head] + body if i < len(row)) for i in range(len(head))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(head, widths)).rstrip()]
    for cells in body:
        fixed = "  ".join(c.ljust(w) for c, w in zip(cells, widths))
        lines.append((fixed + ("  " + cells[-1] if len(cells) > len(head) else "")).rstrip())
    if len(rows) > len(shown):
        lines.append(f"(+{len(rows) - len(shown)} more variants not shown — the {MAX_ROWS} best are listed)")
    return "\n".join(lines)
