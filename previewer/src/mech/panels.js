// Side panels and top bar: issues, targets, parts, display controls, view buttons.
import { buildQuery } from './params.js';
import { fmt } from './model.js';
import { niceStep } from './viewer.js';

const $ = id => document.getElementById(id);

export function h(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : String(c));
  }
  return e;
}

function issueWhere(iss) {
  const bits = [];
  if (iss.study) bits.push(iss.study);
  bits.push(iss.frame === null || iss.frame === undefined ? 'home' : `f${iss.frame}`);
  if (iss.parts?.length) bits.push(iss.parts.join(' / '));
  return bits.join(' · ');
}

/** Top bar: mech picker, status, summary, view buttons, quad toggle, copy link. */
export function buildTopbar(app, slug, mechs) {
  const { model } = app;
  const report = model.report;
  const select = $('mech-select');
  const names = mechs.length ? mechs : [{ name: slug, status: report.status }];
  if (!names.some(x => x.name === slug)) names.unshift({ name: slug, status: report.status });
  for (const m of names) {
    select.append(h('option', { value: m.name, selected: m.name === slug }, m.status ? `${m.name}  ·  ${m.status}` : m.name));
  }
  select.addEventListener('change', () => {
    window.location.search = `?m=${encodeURIComponent(select.value)}`;
  });

  const badge = $('status-badge');
  badge.textContent = report.status || '?';
  badge.classList.add(report.status);
  const mass = report.mass?.total_g;
  const joints = model.scene.joints.filter(j => j.kind !== 'fixed').length;
  $('summary').textContent = [
    `${model.parts.length} parts`, `${joints} joints`,
    mass !== undefined ? `${fmt(mass)} g` : null,
    `${model.studies.length} ${model.studies.length === 1 ? 'study' : 'studies'}`,
  ].filter(Boolean).join(' · ');
  document.title = `${model.scene.name} — mech ${report.status || ''}`.trim();

  for (const b of document.querySelectorAll('#view-buttons [data-view]')) {
    b.addEventListener('click', () => app.setView(b.dataset.view));
  }
  $('btn-fit').addEventListener('click', () => app.fit());
  const quad = $('btn-quad');
  quad.addEventListener('click', () => app.setLayout(app.state.layout === 'quad' ? 'single' : 'quad'));
  $('btn-link').addEventListener('click', async () => {
    const s = app.state;
    const q = buildQuery({
      m: slug,
      study: s.si === null ? null : model.studies[s.si].name,
      frame: s.frame === null ? 'home' : s.frame,
      hide: [...s.hidden], isolate: [...s.isolate], section: s.section, explode: s.explode,
      axes: s.axes, view: s.view, ghost: s.ghost, layout: s.layout,
    });
    const url = `${location.origin}${location.pathname}?${q}`;
    try {
      await navigator.clipboard.writeText(url);
      flash($('btn-link'), 'Copied');
    } catch {
      window.prompt('Viewer link', url);
    }
  });
  app.onChange(what => {
    if (what === 'layout') quad.classList.toggle('on', app.state.layout === 'quad');
  });
  quad.classList.toggle('on', app.state.layout === 'quad');
}

function flash(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => { btn.textContent = old; }, 1200);
}

/** Issues list from report.issues; click → study + frame + isolate + focus. */
export function buildIssues(app) {
  const list = $('issues');
  const issues = app.model.issues;
  $('issue-count').textContent = issues.length ? issues.length : '';
  if (!issues.length) {
    list.append(h('li', { class: 'empty' }, 'No issues'));
    return;
  }
  const rows = issues.map((iss, i) => {
    const li = h('li', { class: 'issue', title: iss.message, onclick: () => app.selectIssue(i) },
      h('span', { class: `sev ${iss.severity}` }, iss.severity),
      h('span', { class: 'code' }, iss.code),
      h('span', { class: 'msg' }, iss.message),
      h('span', { class: 'where' }, `#${i} · ${issueWhere(iss)}`));
    list.append(li);
    return li;
  });
  const sync = () => rows.forEach((li, i) => li.classList.toggle('selected', app.state.issue === i));
  app.onChange(what => { if (what === 'isolate') sync(); });
  sync();
}

export function buildTargets(app) {
  const targets = app.model.report.targets || [];
  if (!targets.length) {
    $('targets-section').hidden = true;
    return;
  }
  const list = $('targets');
  for (const t of targets) {
    const bound = [t.min !== null && t.min !== undefined ? `≥ ${fmt(t.min)}` : null,
      t.max !== null && t.max !== undefined ? `≤ ${fmt(t.max)}` : null].filter(Boolean).join(' ');
    list.append(h('li', { class: `target ${t.met ? 'met' : 'miss'}`, title: `${t.metric} (${t.severity})` },
      h('span', { class: 'mark' }, t.met ? '✓' : '✗'),
      h('span', {}, t.label),
      h('span', { class: 'val' }, `${fmt(t.value)} ${bound}`)));
  }
}

/** Part list: colour, name (click = focus, double-click = isolate), mass, visibility. */
export function buildParts(app) {
  const list = $('parts');
  const { model } = app;
  $('part-count').textContent = model.parts.length;
  const rows = new Map();
  for (const p of model.parts) {
    const eye = h('button', { class: 'icon-btn', title: 'Show / hide' });
    const row = h('li', { class: 'part' },
      h('span', { class: 'swatch', style: `background:${p.color || '#9aa4b2'}` }),
      h('span', {
        class: 'name', title: 'Click: focus · double-click: isolate',
        onclick: () => app.fit([p.id]),
        ondblclick: () => { app.setIsolate([p.id]); app.fit([p.id]); },
      }, p.id),
      p.ground ? h('span', { class: 'ground' }, 'ground') : null,
      h('span', { class: 'mass', title: p.material || '' }, p.mass_g === null || p.mass_g === undefined ? '' : `${fmt(p.mass_g)} g`),
      eye);
    eye.addEventListener('click', () => app.setHidden(p.id, !app.state.hidden.has(p.id)));
    rows.set(p.id, { row, eye });
    list.append(row);
  }
  const sync = () => {
    const flags = new Map();
    for (const e of model.pairIssuesAt(app.state.si, app.state.frame)) {
      if (e.status !== 'interference' && e.status !== 'tight') continue;
      for (const id of [e.a, e.b]) if (flags.get(id) !== 'interference') flags.set(id, e.status);
    }
    for (const [id, { row, eye }] of rows) {
      const off = app.state.hidden.has(id);
      row.classList.toggle('off', off);
      eye.textContent = off ? '○' : '●';
      row.classList.toggle('flag-interference', flags.get(id) === 'interference');
      row.classList.toggle('flag-tight', flags.get(id) === 'tight');
    }
  };
  app.onChange(what => { if (what === 'pose') sync(); });
  sync();
}

/** Display controls: axes, probes, explode, ghosts, section plane. */
export function buildDisplay(app) {
  const s = app.state;
  const axes = $('chk-axes');
  const probes = $('chk-probes');
  axes.checked = s.axes;
  probes.checked = s.probes;
  axes.addEventListener('change', () => app.setAxes(axes.checked));
  probes.addEventListener('change', () => app.setProbes(probes.checked));

  const explode = $('rng-explode');
  const outExplode = $('out-explode');
  explode.value = String(Math.min(s.explode, 1));
  outExplode.textContent = fmt(s.explode, 2);
  explode.addEventListener('input', () => {
    outExplode.textContent = fmt(+explode.value, 2);
    app.setExplode(+explode.value);
  });

  const ghost = $('rng-ghost');
  const outGhost = $('out-ghost');
  ghost.max = String(Math.max(12, s.ghost));
  ghost.value = String(s.ghost);
  outGhost.textContent = s.ghost || 'off';
  ghost.addEventListener('input', () => {
    outGhost.textContent = +ghost.value || 'off';
    app.setGhost(+ghost.value);
  });

  const axisSel = $('sel-section');
  const flip = $('chk-flip');
  const offset = $('rng-section');
  const outOffset = $('out-section');
  const offsetRow = $('section-offset-row');
  const b = app.bounds;
  const rangeFor = axis => {
    // Round step and ends so the slider lands on tidy millimetre values.
    const step = niceStep(Math.max((b.max[axis] - b.min[axis]) / 400, 1e-3));
    offset.min = String(Math.floor(b.min[axis] / step) * step);
    offset.max = String(Math.ceil(b.max[axis] / step) * step);
    offset.step = String(step);
  };
  const sync = () => {
    const sec = s.section;
    axisSel.value = sec ? sec.axis : '';
    flip.checked = !!sec?.flip;
    offsetRow.hidden = !sec;
    if (sec) {
      rangeFor(sec.axis);
      offset.value = String(sec.offset);
      outOffset.textContent = `${fmt(sec.offset, 4)} mm`;
    }
  };
  const apply = keepOffset => {
    const axis = axisSel.value;
    if (!axis) {
      app.setSection(null);
    } else {
      const off = keepOffset && s.section?.axis === axis ? +offset.value : null;
      app.setSection({ axis, offset: off, flip: flip.checked });
    }
    sync();
  };
  axisSel.addEventListener('change', () => apply(false));
  flip.addEventListener('change', () => apply(true));
  offset.addEventListener('input', () => apply(true));
  sync();
}

/** "Isolated: a, b  [Show all]" chip over the canvas. */
export function buildIsolateChip(app) {
  const chip = $('isolate-chip');
  const sync = () => {
    const ids = [...app.state.isolate];
    chip.hidden = ids.length === 0;
    chip.querySelector('span').textContent = `Isolated: ${ids.join(', ')}`;
  };
  $('btn-show-all').addEventListener('click', () => app.clearIsolate());
  app.onChange(what => { if (what === 'isolate') sync(); });
  sync();
}
