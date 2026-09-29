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

/** @returns {Promise<{scene: object, report: object, base: string}>} */
export async function loadMech(slug) {
  if (!/^[A-Za-z0-9_.-]+$/.test(slug)) throw new Error(`bad m=${slug}: expected a mech slug`);
  const base = `${slug}.mech/`;
  const res = await fetchOk(outputUrl(`${base}scene.json`), `mech '${slug}' not found`);
  const scene = await res.json();
  if (scene.version !== 1) throw new Error(`unsupported scene.json version ${scene.version}`);
  let report = scene.report;
  if (!report) {
    report = await (await fetchOk(outputUrl(`${base}report.json`), 'report.json')).json();
  }
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
