"""Headless screenshots of the mech viewer (spec §4.11 ``mech shot``).

Serves the static viewer build (``previewer/dist``, built with ``npm run build`` when missing or
older than its sources) at ``/`` and ``output/`` at ``/output/`` from a threaded ``http.server``
on a free localhost port, opens ``mech.html?m=<slug>&…`` in headless Chromium (playwright),
waits for ``window.__mechReady`` — failing fast on ``window.__mechError`` — and saves the canvas
to ``output/<slug>.mech/shot_<suffix>.png``. playwright is optional (dev dependency): without
it, ``shot`` raises ``ShotError`` with the viewer URL and install instructions.
"""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

from .geom import slug

__all__ = ["ShotError", "shot", "resolve_name", "build_query", "check_scene", "dist_is_current", "ensure_dist", "serve",
           "VIEWS"]

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("iso", "top", "front", "right", "left", "back", "bottom")
TIMEOUT_S = 60.0
VIEWPORT = {"width": 1280, "height": 900}
_DIST_SOURCES = ("mech.html", "index.html", "vite.config.js", "package.json", "src")


class ShotError(RuntimeError):
    """A screenshot could not be taken (message says why and what to do)."""


def resolve_name(name: str, output_dir: Path) -> tuple[str, Path]:
    """(slug, ``output/<slug>.mech``) from ``four_bar``, ``four_bar.mech`` or a path to that folder."""
    base = Path(str(name).rstrip("/\\")).name
    if base.endswith(".mech"):
        base = base[: -len(".mech")]
    s = slug(base)
    return s, Path(output_dir) / f"{s}.mech"


def build_query(slug_: str, report: dict | None, *, issue: int | None = None, view: str | None = None,
                ghost: int | None = None, layout: str | None = None, frame: int | None = None) -> dict[str, str]:
    """Viewer URL params: the explicit options, else the report's targeted view (first issue, or
    a ghosted quad overview). Always ``ui=0`` (canvas only)."""
    explicit = {"issue": issue, "view": view, "ghost": ghost, "layout": layout, "frame": frame}
    params = {k: str(v) for k, v in explicit.items() if v is not None}
    if not params and report and report.get("viewer_url"):
        params = {k: v for k, v in parse_qsl(urlsplit(report["viewer_url"]).query) if k != "m"}
    params.pop("ui", None)
    return {"m": slug_, **params, "ui": "0"}


def _suffix(query: dict[str, str]) -> str:
    """``shot_<suffix>.png`` name from the params, e.g. ``issue0``, ``ghost6_quad``, ``top_f12``."""
    bits = []
    for key in ("issue", "view", "ghost", "layout", "frame"):
        v = query.get(key)
        if v is not None:
            bits.append({"issue": f"issue{v}", "ghost": f"ghost{v}", "frame": f"f{v}"}.get(key, v))
    return slug("_".join(bits)) if bits else "default"


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
    """Static files: ``/output/…`` from the output dir, everything else from the viewer build."""

    # explicit types: Windows' registry may map .js to text/plain, which browsers refuse for modules
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript",
                      ".css": "text/css", ".html": "text/html", ".json": "application/json",
                      ".stl": "application/octet-stream", ".png": "image/png", ".svg": "image/svg+xml"}

    def __init__(self, *args, dist: Path, output: Path, **kw):
        self._roots = (dist, output)
        super().__init__(*args, directory=str(dist), **kw)

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


@contextmanager
def serve(dist: Path, output: Path) -> Iterator[str]:
    """Serve the viewer build and output dir on a free 127.0.0.1 port; yields the base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Handler, dist=dist, output=output))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


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
        raise ShotError(f"the last run of {mech.name} was INVALID ({first}) — nothing to render; fix the model and "
                        f"re-run `uv run mech run <script.py>`")
    scene_path = mech / "scene.json"
    if not scene_path.is_file():
        raise ShotError(f"no scene at {mech} — run `uv run mech run <script.py>` first (`uv run mech list` shows "
                        f"what exists)")
    if report is not None:
        try:
            embedded = json.loads(scene_path.read_text(encoding="utf-8")).get("report")
        except (OSError, ValueError, AttributeError):
            embedded = None
        if embedded != report:
            raise ShotError(f"{scene_path} is stale: it does not match report.json (status "
                            f"{(embedded or {}).get('status', '?')} vs {report.get('status', '?')}) — re-run "
                            f"`uv run mech run <script.py>`")


def shot(name: str, *, issue: int | None = None, view: str | None = None, ghost: int | None = None,
         layout: str | None = None, frame: int | None = None, out: Path | str | None = None,
         output_dir: Path = REPO_ROOT / "output", previewer: Path = REPO_ROOT / "previewer",
         timeout: float = TIMEOUT_S) -> Path:
    """Screenshot the viewer for ``output/<slug>.mech``; returns the PNG path (``ShotError`` on failure)."""
    if view is not None and view not in VIEWS:
        raise ShotError(f"bad view '{view}': expected one of {', '.join(VIEWS)}")
    if layout is not None and layout != "quad":
        raise ShotError(f"bad layout '{layout}': only 'quad' is supported")
    s, mech = resolve_name(name, output_dir)
    report = _read_report(mech)
    check_scene(mech, report)
    query = build_query(s, report, issue=issue, view=view, ghost=ghost, layout=layout, frame=frame)
    path = Path(out) if out is not None else mech / f"shot_{_suffix(query)}.png"
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
                    page_errors: list[str] = []
                    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                    page.goto(url, wait_until="load", timeout=timeout * 1000)
                    try:
                        page.wait_for_function("window.__mechReady === true || window.__mechError !== undefined",
                                               timeout=timeout * 1000)
                    except PlaywrightError:
                        detail = f" (page errors: {'; '.join(page_errors[:3])})" if page_errors else ""
                        raise ShotError(f"viewer not ready after {timeout:g} s{detail}: {url}") from None
                    error = page.evaluate("window.__mechError")
                    if error is not None:
                        raise ShotError(f"viewer error: {error} ({url})")
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
