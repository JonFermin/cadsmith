"""Headless screenshots of the mech viewer (spec §4.11 ``mech shot``).

Serves the static viewer build (``previewer/dist``, built with ``npm run build`` when missing or
older than its sources) at ``/`` and ``output/`` at ``/output/`` from a threaded ``http.server``
on a free localhost port, opens ``mech.html?m=<slug>&…`` in headless Chromium (playwright),
waits for ``window.__mechReady`` — failing fast on ``window.__mechError`` — and saves the canvas
to ``output/<slug>.mech/shot_<suffix>.png``, the suffix encoding the options. playwright is
optional (dev dependency): without it, ``shot`` raises ``ShotError`` with the viewer URL and
install instructions.

Every viewer URL parameter is an option (``OPTIONS``): ``study``, ``frame`` (of that study, or
``home``), ``q`` (``joint:value`` list — the nearest frame; not with ``frame``), ``cam`` (``az,el`` degrees),
``view``, ``section`` (``x|y|z[:offset]``, ``-x`` keeps the other side), ``focus`` / ``isolate``
/ ``hide`` (part lists), ``explode``, ``axes``, ``ghost``, ``layout``, ``issue``, ``zoom``,
``paths``; list options (``q``, ``focus``, ``isolate``, ``hide``) take a comma string or a
list, and ``params`` passes raw ``k=v`` pairs through untouched. Values are checked here the way
``previewer/src/mech/params.js`` checks them, and the study / frame / issue against report.json
(``check_query``), so a typo fails before a browser starts.
Aborted connections from the browser (it cancels requests when the page is done) are swallowed
by the server, and a page load that fails is retried once.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

from .geom import slug

__all__ = ["ShotError", "shot", "resolve_name", "build_query", "shot_suffix", "validate_options", "check_query",
           "check_scene", "dist_is_current", "ensure_dist", "serve", "VIEWS", "OPTIONS"]

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("iso", "top", "front", "right", "left", "back", "bottom")
# viewer URL parameters `mech shot` can set, in the order the file suffix encodes them
OPTIONS = ("issue", "study", "frame", "q", "view", "cam", "zoom", "section", "focus", "isolate", "hide", "explode",
           "axes", "ghost", "paths", "layout")
TIMEOUT_S = 60.0
VIEWPORT = {"width": 1280, "height": 900}
_DIST_SOURCES = ("mech.html", "index.html", "vite.config.js", "package.json", "src")
_SUFFIX_MAX = 48  # longer suffixes are cut and made unique with a hash of the full query
_CONNECTION_ERRORS = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)
_LOAD_ATTEMPTS = 2  # the page load is retried once (the viewer's fetch of a mesh can fail cold)


class ShotError(RuntimeError):
    """A screenshot could not be taken (message says why and what to do)."""


def resolve_name(name: str, output_dir: Path) -> tuple[str, Path]:
    """(slug, ``output/<slug>.mech``) from ``four_bar``, ``four_bar.mech`` or a path to that folder."""
    base = Path(str(name).rstrip("/\\")).name
    if base.endswith(".mech"):
        base = base[: -len(".mech")]
    s = slug(base)
    return s, Path(output_dir) / f"{s}.mech"


# ------------------------------------------------------------------------------ options -> URL


def _text(key: str, value) -> str:
    """The URL text of an option value: lists joined with commas, booleans as 1/0, floats short."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(_text(key, v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).strip()


def _number(key: str, text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        v = float("nan")
    if text == "" or v != v or v in (float("inf"), float("-inf")):
        raise ShotError(f"bad {key}={text}: expected a number")
    return v


def validate_options(options: dict[str, str]) -> None:
    """``ShotError`` for a value the viewer would reject (mirrors ``params.js``), and for ``q`` with
    ``frame``: both pick the frame (the viewer would apply ``frame`` and drop the ``q`` pose)."""
    if options.get("q") and options.get("frame") is not None:
        raise ShotError(f"q={options['q']} and frame={options['frame']} both pick the frame — q poses the frame "
                        f"nearest those joint values: drop frame (or drop q to show frame {options['frame']})")
    for key, raw in options.items():
        if key == "view" and raw not in VIEWS:
            raise ShotError(f"bad view '{raw}': expected one of {', '.join(VIEWS)}")
        elif key == "layout" and raw not in ("single", "quad"):
            raise ShotError(f"bad layout '{raw}': expected quad (or single)")
        elif key in ("issue", "ghost") and not re.fullmatch(r"\d+", raw):
            raise ShotError(f"bad {key}={raw}: expected an integer ≥ 0")
        elif key == "frame" and raw != "home" and not re.fullmatch(r"\d+", raw):
            raise ShotError(f"bad frame={raw}: expected a frame index ≥ 0 or 'home'")
        elif key == "q":
            for item in raw.split(","):
                joint, sep, value = item.rpartition(":")
                if not sep or not joint:
                    raise ShotError(f"bad q={item}: expected joint:value")
                _number("q", value)
        elif key == "cam":
            parts = raw.split(",")
            if len(parts) != 2:
                raise ShotError(f"bad cam={raw}: expected az,el (degrees)")
            if abs(_number("cam", parts[1])) > 90:
                raise ShotError(f"bad cam={raw}: elevation must be within ±90°")
            _number("cam", parts[0])
        elif key == "section":
            m = re.fullmatch(r"(-?)([xyzXYZ])(?::(.+))?", raw)
            if not m:
                raise ShotError(f"bad section={raw}: expected x|y|z[:offset_mm] (-x keeps the other side)")
            if m.group(3) is not None:
                _number("section", m.group(3))
        elif key == "explode" and _number("explode", raw) < 0:
            raise ShotError(f"bad explode={raw}: must be ≥ 0")
        elif key == "zoom" and _number("zoom", raw) <= 0:
            raise ShotError(f"bad zoom={raw}: must be > 0")
        elif key in ("axes", "paths") and raw.lower() not in ("0", "1", "true", "false", "on", "off", "yes", "no"):
            raise ShotError(f"bad {key}={raw}: expected 0 or 1")


def build_query(slug_: str, report: dict | None, *, params: dict | None = None, **options) -> dict[str, str]:
    """Viewer URL params: the explicit options (``OPTIONS`` keywords, None = unset) plus raw
    ``params``, else the report's targeted view (first issue, or a ghosted quad overview).
    Always ``ui=0`` (canvas only). ``ShotError`` on an unknown or malformed option."""
    unknown = [k for k in options if k not in OPTIONS]
    if unknown:
        raise ShotError(f"unknown shot option{'s' if len(unknown) > 1 else ''} {', '.join(unknown)} — "
                        f"expected {', '.join(OPTIONS)} (or params={{k: v}} for anything else)")
    explicit = {k: _text(k, options[k]) for k in OPTIONS if options.get(k) is not None}
    validate_options(explicit)
    extra = {str(k).strip(): _text(str(k), v) for k, v in (params or {}).items() if str(k).strip() not in ("m", "ui")}
    chosen = {**explicit, **extra}
    if not chosen and report and report.get("viewer_url"):
        chosen = {k: v for k, v in parse_qsl(urlsplit(report["viewer_url"]).query) if k != "m"}
    chosen.pop("ui", None)
    return {"m": slug_, **chosen, "ui": "0"}


def check_query(query: dict[str, str], report: dict | None) -> None:
    """``ShotError`` when the query names what the report does not have — an unknown study, a frame
    past the chosen study's last (``frame`` applies to ``study``, else to the first study) or an
    issue index past the list — so the shot fails before a browser starts."""
    if not report:
        return
    studies = {s.get("name"): s.get("frames") for s in report.get("studies") or []}
    study = query.get("study")
    if study is not None and study not in studies:
        raise ShotError(f"unknown study '{study}' — this run has {', '.join(map(str, studies)) or 'no studies'}")
    frame = query.get("frame")
    if frame not in (None, "home") and frame.isdigit() and studies:
        name = study if study is not None else next(iter(studies))
        n = studies.get(name)
        if isinstance(n, int) and int(frame) >= n:
            raise ShotError(f"frame {frame} is past the last frame of study '{name}' (frames 0…{n - 1})")
    issue = query.get("issue")
    if issue is not None and issue.isdigit() and int(issue) >= len(report.get("issues") or []):
        raise ShotError(f"issue {issue} does not exist (the report lists {len(report.get('issues') or [])})")


_SUFFIX_FORMS = {"issue": "issue{}", "frame": "f{}", "cam": "cam{}", "zoom": "z{}", "section": "sec{}",
                 "focus": "focus_{}", "isolate": "only_{}", "hide": "hide_{}", "explode": "x{}", "ghost": "ghost{}",
                 "paths": "paths{}", "q": "q_{}"}


def shot_suffix(query: dict[str, str]) -> str:
    """``shot_<suffix>.png`` name from the params, in ``OPTIONS`` order: ``issue0``, ``ghost6_quad``,
    ``f12_top``, ``dig_f40_cam30_20_z1_5``; raw ``k=v`` passthroughs as ``k_v``. Sanitised by
    ``geom.slug``; longer than ``_SUFFIX_MAX`` characters it is cut and ends in a short hash of the
    full query, so different option sets never share a file."""
    bits = []
    for key in OPTIONS:
        v = query.get(key)
        if v is None:
            continue
        if key == "axes":
            bits.append("axes" if v.lower() not in ("0", "false", "off", "no") else "noaxes")
        else:
            bits.append(_SUFFIX_FORMS.get(key, "{}").format(v))
    bits += [f"{k}_{v}" for k, v in query.items() if k not in OPTIONS and k not in ("m", "ui")]
    if not bits:
        return "default"
    text = slug("_".join(bits))
    if len(text) > _SUFFIX_MAX:
        digest = hashlib.sha1(urlencode(sorted(query.items())).encode("utf-8")).hexdigest()[:6]
        text = f"{text[:_SUFFIX_MAX - 7].rstrip('_')}_{digest}"
    return text


# ------------------------------------------------------------------------------ viewer build & server


def dist_is_current(previewer: Path = REPO_ROOT / "previewer") -> bool:
    """True when ``previewer/dist/mech.html`` exists and is newer than every viewer source."""
    built = previewer / "dist" / "mech.html"
    if not built.is_file():
        return False
    sources = []
    for name in _DIST_SOURCES:
        p = previewer / name
        sources += list(p.rglob("*")) if p.is_dir() else [p]
    newest = max((p.stat().st_mtime for p in sources if p.is_file()), default=0.0)
    return built.stat().st_mtime >= newest


def ensure_dist(previewer: Path = REPO_ROOT / "previewer") -> Path:
    """``previewer/dist``, rebuilt with ``npm run build`` when missing or older than its sources."""
    dist = previewer / "dist"
    built = dist / "mech.html"
    if dist_is_current(previewer):
        return dist
    if not (previewer / "node_modules").is_dir():
        raise ShotError(f"viewer build missing and {previewer / 'node_modules'} not installed — run "
                        f"`cd previewer && npm install` once")
    # npm is a .cmd shim on Windows, which only the shell resolves
    proc = subprocess.run("npm run build", cwd=previewer, shell=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not built.is_file():
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
        raise ShotError(f"`npm run build` in {previewer} failed:\n{tail}")
    return dist


class _Handler(SimpleHTTPRequestHandler):
    """Static files: ``/output/…`` from the output dir, everything else from the viewer build.
    A connection the browser drops mid-response (it aborts fetches once the page is done) is not
    an error worth a traceback: ``handle`` swallows it."""

    # explicit types: Windows' registry may map .js to text/plain, which browsers refuse for modules
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript",
                      ".css": "text/css", ".html": "text/html", ".json": "application/json",
                      ".stl": "application/octet-stream", ".png": "image/png", ".svg": "image/svg+xml"}

    def __init__(self, *args, dist: Path, output: Path, **kw):
        self._roots = (dist, output)
        super().__init__(*args, directory=str(dist), **kw)

    def handle(self) -> None:
        try:
            super().handle()
        except _CONNECTION_ERRORS:
            self.close_connection = True

    def translate_path(self, path: str) -> str:
        dist, output = self._roots
        rel = unquote(urlsplit(path).path)
        root = dist
        if rel == "/output" or rel.startswith("/output/"):
            root, rel = output, rel[len("/output"):]
        parts = [p for p in posixpath.normpath(rel).split("/") if p not in ("", ".", "..")]
        if any(":" in p or "\\" in p for p in parts):  # drive letters / backslashes: outside the roots
            return str(root / "__forbidden__")
        return str(root.joinpath(*parts))

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # re-runs overwrite output files in place
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - base-class signature
        pass  # keep the CLI output clean


class _Server(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` that keeps quiet about connections the client aborted (WinError
    10053/10054 while the viewer page tears down) instead of printing a traceback per request."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, _CONNECTION_ERRORS):
            return
        super().handle_error(request, client_address)


@contextmanager
def serve(dist: Path, output: Path) -> Iterator[str]:
    """Serve the viewer build and output dir on a free 127.0.0.1 port; yields the base URL."""
    server = _Server(("127.0.0.1", 0), partial(_Handler, dist=dist, output=output))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------------------ the shot


def _read_report(mech: Path) -> dict | None:
    try:
        return json.loads((mech / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def check_scene(mech: Path, report: dict | None) -> None:
    """``ShotError`` unless ``mech/scene.json`` shows the last run: the last run was INVALID (no
    scene), there is no scene, or scene.json embeds another report than report.json (stale)."""
    if report is not None and report.get("status") == "INVALID":
        first = next((i.get("message") for i in report.get("issues") or []), "")
        raise ShotError(f"the last run of {mech.name} was INVALID ({first}) — nothing to render; fix the model "
                        f"script and run it again with `uv run mech run`")
    scene_path = mech / "scene.json"
    if not scene_path.is_file():
        raise ShotError(f"no scene at {mech} — run the model script with `uv run mech run` first (`uv run mech list` "
                        f"shows what exists)")
    if report is not None:
        try:
            embedded = json.loads(scene_path.read_text(encoding="utf-8")).get("report")
        except (OSError, ValueError, AttributeError):
            embedded = None
        if embedded != report:
            raise ShotError(f"{scene_path} is stale: it does not match report.json (status "
                            f"{(embedded or {}).get('status', '?')} vs {report.get('status', '?')}) — run the model "
                            f"script again with `uv run mech run`")


def _load_viewer(page, url: str, timeout: float, PlaywrightError) -> None:
    """Open ``url`` and wait for the viewer to be ready; one retry when the load or the viewer's
    own data fetch fails (a cold mesh fetch can abort). ``ShotError`` with the page errors."""
    last = None
    for attempt in range(1, _LOAD_ATTEMPTS + 1):
        page_errors: list[str] = []
        handler = lambda exc: page_errors.append(str(exc))  # noqa: E731
        page.on("pageerror", handler)
        try:
            page.goto(url, wait_until="load", timeout=timeout * 1000)
            try:
                page.wait_for_function("window.__mechReady === true || window.__mechError !== undefined",
                                       timeout=timeout * 1000)
            except PlaywrightError:
                detail = f" (page errors: {'; '.join(page_errors[:3])})" if page_errors else ""
                last = ShotError(f"viewer not ready after {timeout:g} s{detail}: {url}")
                continue
            error = page.evaluate("window.__mechError")
            if error is None:
                return
            last = ShotError(f"viewer error: {error} ({url})")
            if "fetch" not in str(error).lower() and "load" not in str(error).lower():
                break  # a bad parameter or model: retrying cannot help
        except PlaywrightError as exc:
            last = ShotError(f"playwright failed: {str(exc).strip().splitlines()[0] if str(exc).strip() else exc}")
        finally:
            page.remove_listener("pageerror", handler)
    raise last if last is not None else ShotError(f"viewer did not load: {url}")


def shot(name: str, *, out: Path | str | None = None, output_dir: Path = REPO_ROOT / "output",
         previewer: Path = REPO_ROOT / "previewer", timeout: float = TIMEOUT_S, params: dict | None = None,
         **options) -> Path:
    """Screenshot the viewer for ``output/<slug>.mech``; returns the PNG path (``ShotError`` on failure).

    ``options`` are the viewer parameters in ``OPTIONS`` (``study=``, ``frame=``, ``view=``, …;
    see the module docstring), ``params`` raw extra ``k=v`` pairs. Without any, the report's
    targeted view is shot. The PNG is ``<mech>/shot_<suffix>.png`` unless ``out`` is given.
    """
    s, mech = resolve_name(name, output_dir)
    report = _read_report(mech)
    query = build_query(s, report, params=params, **options)  # option errors before any I/O
    check_scene(mech, report)
    check_query(query, report)
    path = Path(out) if out is not None else mech / f"shot_{shot_suffix(query)}.png"
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ShotError(f"playwright is not installed; open http://localhost:3000/mech.html?{urlencode(query)} with "
                        f"`cd previewer && npm run dev`, or install it: `uv sync --group dev && uv run playwright "
                        f"install chromium`") from None
    dist = ensure_dist(previewer)
    with serve(dist, Path(output_dir)) as base:
        url = f"{base}/mech.html?{urlencode(query, safe=',:')}"
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    page = browser.new_page(viewport=VIEWPORT)
                    _load_viewer(page, url, timeout, PlaywrightError)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    canvas = page.locator("canvas").first
                    if canvas.count():
                        canvas.screenshot(path=str(path))
                    else:
                        page.screenshot(path=str(path))
                finally:
                    browser.close()
        except PlaywrightError as exc:
            msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            hint = " — run `uv run playwright install chromium`" if "Executable doesn't exist" in str(exc) else ""
            raise ShotError(f"playwright failed: {msg}{hint}") from None
    return Path(os.path.abspath(path))
