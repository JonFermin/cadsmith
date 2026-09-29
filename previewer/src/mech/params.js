// URL parameter parsing for mech.html (MECH_SPEC §7). Every malformed value throws, so an
// automated screenshot (`mech shot`) fails fast via window.__mechError instead of silently
// rendering the wrong state.

export const VIEWS = ['iso', 'top', 'front', 'right', 'left', 'back', 'bottom'];

const list = v => (v ? v.split(',').map(s => s.trim()).filter(Boolean) : []);

function num(name, raw) {
  const v = Number(raw);
  if (raw === '' || !Number.isFinite(v)) throw new Error(`bad ${name}=${raw}: expected a number`);
  return v;
}

function int(name, raw, min = 0) {
  const v = num(name, raw);
  if (!Number.isInteger(v) || v < min) throw new Error(`bad ${name}=${raw}: expected an integer ≥ ${min}`);
  return v;
}

/**
 * @typedef {object} MechParams
 * @property {string|null} m            mech slug (output/<m>.mech/)
 * @property {string|null} study
 * @property {number|'home'|null} frame  0-based frame index, or 'home' for the drawn pose
 * @property {{joint: string, value: number}[]} q   pick the frame nearest these joint values
 * @property {string[]} hide
 * @property {string[]} isolate
 * @property {string[]} focus
 * @property {number|null} issue        index into report.issues
 * @property {{axis: 'x'|'y'|'z', offset: number|null, flip: boolean}|null} section
 * @property {number} explode            0..1 (values above 1 are allowed, just further apart)
 * @property {boolean} axes
 * @property {string|null} view
 * @property {{az: number, el: number}|null} cam   degrees; az from +X toward +Y, el above XY
 * @property {number} ghost
 * @property {'single'|'quad'} layout
 * @property {boolean} ui
 */

/** @returns {MechParams} */
export function parseParams(search = window.location.search) {
  const p = new URLSearchParams(search);
  const get = k => {
    const v = p.get(k);
    return v === null ? null : v.trim();
  };

  const frameRaw = get('frame');
  let frame = null;
  if (frameRaw !== null) frame = frameRaw === 'home' ? 'home' : int('frame', frameRaw);

  const q = list(get('q')).map(item => {
    const i = item.lastIndexOf(':');
    if (i <= 0) throw new Error(`bad q=${item}: expected joint:value`);
    return { joint: item.slice(0, i), value: num('q', item.slice(i + 1)) };
  });

  // section=x:10 | x (bbox centre) | -x:10 (keep the other side)
  let section = null;
  const sec = get('section');
  if (sec) {
    const m = /^(-?)([xyz])(?::(.+))?$/i.exec(sec);
    if (!m) throw new Error(`bad section=${sec}: expected x|y|z[:offset_mm]`);
    section = {
      axis: m[2].toLowerCase(),
      offset: m[3] === undefined ? null : num('section', m[3]),
      flip: m[1] === '-',
    };
  }

  const view = get('view');
  if (view && !VIEWS.includes(view)) throw new Error(`bad view=${view}: expected ${VIEWS.join('|')}`);

  let cam = null;
  const camRaw = get('cam');
  if (camRaw) {
    const parts = camRaw.split(',');
    if (parts.length !== 2) throw new Error(`bad cam=${camRaw}: expected az,el (degrees)`);
    cam = { az: num('cam', parts[0]), el: num('cam', parts[1]) };
    if (Math.abs(cam.el) > 90) throw new Error(`bad cam=${camRaw}: elevation must be within ±90°`);
  }

  const layout = get('layout') || 'single';
  if (!['single', 'quad'].includes(layout)) throw new Error(`bad layout=${layout}: expected quad`);

  const explode = get('explode') === null ? 0 : num('explode', get('explode'));
  if (explode < 0) throw new Error(`bad explode=${explode}: must be ≥ 0`);

  const flag = k => {
    const v = get(k);
    return v !== null && !['0', 'false', 'off', 'no'].includes(v.toLowerCase());
  };

  return {
    m: get('m') || null,
    study: get('study') || null,
    frame,
    q,
    hide: list(get('hide')),
    isolate: list(get('isolate')),
    focus: list(get('focus')),
    issue: get('issue') === null ? null : int('issue', get('issue')),
    section,
    explode,
    axes: flag('axes'),
    view,
    cam,
    ghost: get('ghost') === null ? 0 : int('ghost', get('ghost')),
    layout,
    ui: get('ui') === null ? true : flag('ui'),
  };
}

/** Serialize the viewer state back into a shareable query string (only non-defaults). */
export function buildQuery(s) {
  const p = new URLSearchParams();
  p.set('m', s.m);
  if (s.study) p.set('study', s.study);
  if (s.frame !== null && s.frame !== undefined) p.set('frame', String(s.frame));
  if (s.hide.length) p.set('hide', s.hide.join(','));
  if (s.isolate.length) p.set('isolate', s.isolate.join(','));
  if (s.section) {
    const off = s.section.offset === null ? '' : `:${+s.section.offset.toFixed(3)}`;
    p.set('section', `${s.section.flip ? '-' : ''}${s.section.axis}${off}`);
  }
  if (s.explode > 0) p.set('explode', String(+s.explode.toFixed(3)));
  if (s.axes) p.set('axes', '1');
  if (s.view) p.set('view', s.view);
  if (s.ghost > 0) p.set('ghost', String(s.ghost));
  if (s.layout === 'quad') p.set('layout', 'quad');
  return p.toString().replace(/%2C/g, ',').replace(/%3A/g, ':');
}
