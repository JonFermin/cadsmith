// Timeline (study picker, play/pause, scrubber with issue ticks, speed) and the pose HUD.
import { fmt, formatJointValue } from './model.js';
import { h } from './panels.js';

const $ = id => document.getElementById(id);

export function buildTimeline(app) {
  const { model } = app;
  const s = app.state;
  const bar = $('timeline');
  const play = $('btn-play');
  const study = $('study-select');
  const scrub = $('scrubber');
  const label = $('frame-label');
  const ticks = $('ticks');

  model.studies.forEach((st, i) => study.append(h('option', { value: i }, `${st.name} (${st.frames})`)));
  if (!model.studies.length) {
    study.append(h('option', {}, 'no studies'));
    for (const e of [study, play, scrub]) e.disabled = true;
  }
  study.addEventListener('change', () => app.setStudy(+study.value));
  play.addEventListener('click', () => app.setPlaying(!s.playing));
  $('btn-home').addEventListener('click', () => {
    app.setPlaying(false);
    app.setFrame(null);
  });
  scrub.addEventListener('input', () => {
    app.setPlaying(false);
    app.setFrame(+scrub.value);
  });
  const speed = $('speed');
  speed.addEventListener('change', () => app.setSpeed(+speed.value));

  /** Red / orange tick marks on the scrubber for frames with interference / tight pairs. */
  const drawTicks = () => {
    ticks.replaceChildren();
    if (s.si === null) return;
    const st = model.studies[s.si];
    const worst = new Map();
    for (const e of st.issues || []) {
      if (e.frame === null || (e.status !== 'interference' && e.status !== 'tight')) continue;
      if (worst.get(e.frame) !== 'interference') worst.set(e.frame, e.status);
    }
    const n = Math.max(st.frames - 1, 1);
    for (const [f, status] of worst) {
      ticks.append(h('i', { class: status, style: `left:${(100 * f) / n}%`, title: `f${f}: ${status}` }));
    }
  };

  let shownStudy;
  const sync = () => {
    if (s.si !== shownStudy) {
      shownStudy = s.si;
      study.value = s.si === null ? '' : String(s.si);
      scrub.max = String(Math.max(model.frameCount(s.si) - 1, 0));
      drawTicks();
    }
    play.textContent = s.playing ? '❚❚' : '▶';
    bar.classList.toggle('home', s.frame === null);
    if (s.frame !== null) scrub.value = String(s.frame);
    const n = model.frameCount(s.si);
    if (s.frame === null) label.textContent = 'home pose';
    else {
      const t = model.studies[s.si].t?.[s.frame];
      label.textContent = `f${s.frame} / ${n - 1} · ${fmt(t ?? 0, 3)} s`;
    }
  };
  app.onChange(sync);
  sync();
}

/** Pose HUD: frame/time, joint values with roles, gravity loads vs capacity, loop residual. */
export function buildHud(app) {
  const { model } = app;
  const s = app.state;
  const hud = $('hud');
  const roles = model.report.roles || {};

  const render = () => {
    const rows = [];
    const st = s.si === null ? null : model.studies[s.si];
    rows.push(h('tr', {}, h('td', { class: 'sub' }, 'study'), h('td', {}, st ? st.name : '—')));
    rows.push(h('tr', {}, h('td', { class: 'sub' }, 'pose'),
      h('td', {}, s.frame === null ? 'home' : `f${s.frame} · ${fmt(st.t?.[s.frame] ?? 0, 3)} s`)));

    rows.push(h('tr', { class: 'grp' }, h('td', { colspan: 2 }, 'joints')));
    for (const { joint, value } of model.jointValuesAt(s.si, s.frame)) {
      const lim = joint.limits;
      const outside = lim && value !== null && (value < lim[0] - 1e-6 || value > lim[1] + 1e-6);
      rows.push(h('tr', {},
        h('td', {}, joint.name, h('span', { class: 'role' }, roles[joint.name] || joint.role || '')),
        h('td', { class: outside ? 'flag' : null, title: lim ? `limits ${lim[0]} … ${lim[1]}` : null },
          formatJointValue(joint, value))));
    }

    const loads = model.loadsAt(s.si, s.frame);
    if (loads.length) {
      rows.push(h('tr', { class: 'grp' }, h('td', { colspan: 2 }, 'holding loads')));
      for (const l of loads) {
        const over = l.capacity !== null && l.value !== null && Math.abs(l.value) > l.capacity;
        rows.push(h('tr', {}, h('td', {}, l.name),
          h('td', { class: over ? 'over' : null, title: l.capacity !== null ? `capacity ${l.capacity} ${l.unit}` : null },
            l.value === null ? '—' : `${fmt(l.value, 3)} ${l.unit}`)));
      }
    }

    const pairs = model.pairIssuesAt(s.si, s.frame).filter(e => e.status === 'interference' || e.status === 'tight');
    if (pairs.length) {
      rows.push(h('tr', { class: 'grp' }, h('td', { colspan: 2 }, 'clearance')));
      for (const e of pairs) {
        rows.push(h('tr', {}, h('td', { class: e.status === 'interference' ? 'over' : 'flag' }, `${e.a} / ${e.b}`),
          h('td', {}, e.status === 'interference' ? `${fmt(e.volume)} mm³` : `${fmt(e.distance)} mm`)));
      }
    }
    const res = st && s.frame !== null ? st.residual?.[s.frame] : null;
    if (res !== null && res !== undefined) {
      rows.push(h('tr', { class: 'grp' }, h('td', { colspan: 2 }, 'loop')));
      rows.push(h('tr', {}, h('td', { class: 'sub' }, 'residual'),
        h('td', { class: res > 1e-4 ? 'over' : null }, `${fmt(res, 2)} mm`)));
    }
    hud.replaceChildren(h('table', {}, rows));
  };
  app.onChange(what => { if (what === 'pose' || what === 'study') render(); });
  render();
}
