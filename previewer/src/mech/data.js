// Loading of exported mech scenes (MECH_SPEC §4.10): scene.json, report.json and part STLs.
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

/**
 * URL of a file under output/, resolved against the page so the viewer works both from the
 * vite dev server and from a static server that mounts output/ next to dist/.
 */
function outputUrl(rel) {
  return new URL(`output/${rel}`, document.baseURI).href;
}

async function fetchOk(url, what) {
  let res;
  try {
    res = await fetch(url, { cache: 'no-store' });
  } catch (err) {
    throw new Error(`${what}: ${err.message} (${url})`);
  }
  if (!res.ok) throw new Error(`${what}: HTTP ${res.status} (${url})`);
  // The vite dev server answers unknown paths with index.html (200) — treat that as missing.
  if ((res.headers.get('content-type') || '').includes('text/html')) {
    throw new Error(`${what}: no such file (${url})`);
  }
  return res;
}

/** report.json of a mech, or null when it is missing/unreadable. */
async function tryReport(base) {
  try {
    return await (await fetchOk(outputUrl(`${base}report.json`), 'report.json')).json();
  } catch {
    return null;
  }
}

/**
 * Load scene.json (+ its report). The scene must show the last run: after an INVALID run there
 * is none (report.json names the error), and a scene whose embedded report differs from
 * report.json is stale — both are errors rather than an old model shown as current.
 * @returns {Promise<{scene: object, report: object, base: string}>}
 */
export async function loadMech(slug) {
  if (!/^[A-Za-z0-9_.-]+$/.test(slug)) throw new Error(`bad m=${slug}: expected a mech slug`);
  const base = `${slug}.mech/`;
  let res;
  try {
    res = await fetchOk(outputUrl(`${base}scene.json`), `mech '${slug}' not found`);
  } catch (err) {
    const last = await tryReport(base);
    if (last?.status === 'INVALID') {
      const first = last.issues?.[0]?.message || 'invalid model';
      throw new Error(`mech '${slug}': the last run was INVALID (${first}) — fix the model and re-run`);
    }
    throw err;
  }
  const scene = await res.json();
  if (scene.version !== 1) throw new Error(`unsupported scene.json version ${scene.version}`);
  const saved = await tryReport(base);
  if (saved && scene.report && JSON.stringify(saved) !== JSON.stringify(scene.report)) {
    throw new Error(`mech '${slug}': scene.json is stale (report.json says ${saved.status}, the scene `
      + `${scene.report.status}) — re-run \`uv run mech run\``);
  }
  const report = scene.report || saved;
  if (!report) throw new Error(`mech '${slug}': report.json missing`);
  return { scene, report, base };
}

/** Load every part's binary STL (paths come from parts[].mesh only). Returns id → geometry. */
export async function loadGeometries(scene, base) {
  const loader = new STLLoader();
  const entries = await Promise.all(scene.parts.map(async part => {
    const res = await fetchOk(outputUrl(base + part.mesh), `mesh for part '${part.id}'`);
    const buf = await res.arrayBuffer();
    if (buf.byteLength < 84) throw new Error(`${part.mesh}: truncated STL (${buf.byteLength} bytes)`);
    const n = new DataView(buf).getUint32(80, true);
    if (buf.byteLength !== 84 + 50 * n) {
      throw new Error(`${part.mesh}: not a binary STL (${buf.byteLength} bytes for ${n} triangles)`);
    }
    const geom = loader.parse(buf);
    geom.computeBoundingBox();
    return [part.id, geom];
  }));
  return new Map(entries);
}

/** Mechs listed by the dev server's /api/mechs; [] under a plain static server. */
export async function listMechs() {
  try {
    const res = await fetch(new URL('api/mechs', document.baseURI).href, { cache: 'no-store' });
    if (!res.ok || !(res.headers.get('content-type') || '').includes('json')) return [];
    const list = await res.json();
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}
