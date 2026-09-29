// mech.html entry: load output/<m>.mech/, build the scene and UI, apply URL params, then
// signal window.__mechReady (or window.__mechError) for headless screenshots (`mech shot`).
import './mech.css';
import { Raycaster, Vector2 } from 'three';
import { parseParams } from './params.js';
import { listMechs, loadGeometries, loadMech } from './data.js';
import { MechModel, fmt } from './model.js';
import { Viewer } from './viewer.js';
import { MechApp } from './app.js';
import { buildDisplay, buildIsolateChip, buildIssues, buildParts, buildTargets, buildTopbar } from './panels.js';
import { buildHud, buildTimeline } from './timeline.js';

const $ = id => document.getElementById(id);

function showMessage(text, isError = false) {
  const m = $('message');
  m.hidden = false;
  m.textContent = text;
  m.classList.toggle('error', isError);
}

function fail(err) {
  const msg = (err && err.message) || String(err);
  if (window.__mechError === undefined) window.__mechError = msg;
  showMessage(`Could not show mechanism:\n${msg}`, true);
}

window.addEventListener('error', e => fail(e.error || e.message));
window.addEventListener('unhandledrejection', e => fail(e.reason));

const nextFrame = () => new Promise(resolve => requestAnimationFrame(() => resolve()));

function installKeys(app) {
  window.addEventListener('keydown', e => {
    if (e.target.closest('input, select, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
    const s = app.state;
    switch (e.key) {
      case ' ': app.setPlaying(!s.playing); break;
      case 'ArrowRight': app.step(1); break;
      case 'ArrowLeft': app.step(-1); break;
      case 'h': case 'H': app.setPlaying(false); app.setFrame(null); break;
      case 'f': case 'F': app.fit(); break;
      case 'Escape': app.clearIsolate(); break;
      default: return;
    }
    e.preventDefault();
  });
}

/** Hover a part to see its name, material and mass (iso pane only). */
function installTooltip(app, viewer) {
  const tip = $('tooltip');
  const ray = new Raycaster();
  const ndc = new Vector2();
  let pending = null;
  const pick = () => {
    const ev = pending;
    pending = null;
    const rect = viewer.canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const pane = viewer.panes()[0];
    const inside = x >= pane.x && x < pane.x + pane.w && y >= pane.y && y < pane.y + pane.h;
    let hit = null;
    if (inside && ev.buttons === 0) {
      ndc.set(((x - pane.x) / pane.w) * 2 - 1, -((y - pane.y) / pane.h) * 2 + 1);
      ray.setFromCamera(ndc, pane.cam);
      const meshes = app.parts.meshes().filter(m => m.visible && m.material.opacity > 0.5);
      hit = ray.intersectObjects(meshes, false)[0] || null;
    }
    if (!hit) {
      tip.hidden = true;
      return;
    }
    const meta = app.model.parts.find(p => p.id === hit.object.userData.partId);
    tip.replaceChildren(meta.id);
    const sub = document.createElement('div');
    sub.className = 'sub';
    sub.textContent = [meta.material, meta.mass_g !== null && meta.mass_g !== undefined ? `${fmt(meta.mass_g)} g` : null,
      meta.ground ? 'ground' : null, meta.bom].filter(Boolean).join(' · ');
    tip.append(sub);
    tip.style.left = `${x + 14}px`;
    tip.style.top = `${y + 12}px`;
    tip.hidden = false;
  };
  viewer.canvas.addEventListener('pointermove', ev => {
    if (!pending) requestAnimationFrame(pick);
    pending = ev;
  });
  viewer.canvas.addEventListener('pointerleave', () => { tip.hidden = true; });
}

async function main() {
  const params = parseParams();
  if (!params.ui) document.body.classList.add('bare');

  // /api/mechs only exists on the vite server; a static build (mech shot) relies on ?m=.
  const mechs = (params.ui && import.meta.env.DEV) || !params.m ? await listMechs() : [];
  const slug = params.m || mechs[0]?.name;
  if (!slug) throw new Error('no mechanism: pass ?m=<name> (output/<name>.mech/)');
  showMessage(`Loading ${slug}…`);

  const { scene, report, base } = await loadMech(slug);
  const geometries = await loadGeometries(scene, base);
  const model = new MechModel(scene, report);
  const viewer = new Viewer($('stage'), $('viewport'));
  const app = new MechApp(viewer, model, geometries);
  app.init(params);

  if (params.ui) {
    buildTopbar(app, slug, mechs);
    buildIssues(app);
    buildTargets(app);
    buildParts(app);
    buildDisplay(app);
    buildIsolateChip(app);
    buildTimeline(app);
    buildHud(app);
    installKeys(app);
    installTooltip(app, viewer);
  }

  $('message').hidden = true;
  viewer.render();
  viewer.start(dt => app.tick(dt));
  await nextFrame();
  await nextFrame();
  if (window.__mechError === undefined) window.__mechReady = true;
  window.__mech = app; // debugging handle
}

main().catch(fail);
