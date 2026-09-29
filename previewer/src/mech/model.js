// Read-only view over scene.json: per-frame transforms, issues, joint values and loads.
import { Matrix4 } from 'three';

const IDENTITY = Object.freeze(new Matrix4());

/** Levenshtein distance (names are short, so the O(n·m) table is fine). */
function editDistance(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++) {
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = row;
  }
  return prev[b.length];
}

/** "did you mean" helper for name errors: substring matches or edits ≤ ⅓ of the length. */
function unknown(kind, name, options) {
  const lower = name.toLowerCase();
  const close = options.filter(o => {
    const opt = o.toLowerCase();
    return opt.includes(lower) || lower.includes(opt) || editDistance(lower, opt) <= Math.max(1, Math.floor(opt.length / 3));
  });
  const hint = close.length ? ` — did you mean ${close.join(', ')}?` : '';
  return new Error(`unknown ${kind} '${name}'${hint} (have: ${options.join(', ') || 'none'})`);
}

export class MechModel {
  constructor(scene, report) {
    this.scene = scene;
    this.report = report;
    this.parts = scene.parts;
    this.partIds = scene.parts.map(p => p.id);
    this.joints = new Map(scene.joints.map(j => [j.name, j]));
    this.studies = scene.studies || [];
    this.issues = report.issues || [];
    // Pre-build Matrix4 arrays once; the timeline scrubs through them every frame.
    this._matrices = this.studies.map(st => {
      const out = new Map();
      for (const [id, arr] of Object.entries(st.transforms || {})) {
        out.set(id, arr.map(m => new Matrix4().fromArray(m)));
      }
      return out;
    });
  }

  studyIndex(name) {
    const i = this.studies.findIndex(s => s.name === name);
    if (i < 0) throw unknown('study', name, this.studies.map(s => s.name));
    return i;
  }

  checkParts(names, param) {
    for (const n of names) {
      if (!this.partIds.includes(n)) throw unknown(`part in ${param}=`, n, this.partIds);
    }
    return names;
  }

  frameCount(si) {
    return si === null || !this.studies[si] ? 0 : this.studies[si].frames;
  }

  /** Part transform at (study, frame); frame === null is the home pose (identity). */
  transform(si, frame, id) {
    if (si === null || frame === null) return IDENTITY;
    const seq = this._matrices[si].get(id);
    return seq ? seq[frame] : IDENTITY;
  }

  /** Scene clearance entries at the current pose (frame null = home pose entries). */
  pairIssuesAt(si, frame) {
    if (si === null) return [];
    return (this.studies[si].issues || []).filter(e => (e.frame ?? null) === frame);
  }

  /** Joint values (deg / mm) at a frame; home values at the home pose. */
  jointValuesAt(si, frame) {
    const out = [];
    for (const j of this.joints.values()) {
      if (j.kind === 'fixed') continue;
      const series = si !== null && frame !== null ? this.studies[si].joints?.[j.name] : null;
      out.push({ joint: j, value: series ? series[frame] : j.home });
    }
    return out;
  }

  loadsAt(si, frame) {
    if (si === null || frame === null) return [];
    const loads = this.studies[si].loads || {};
    const meta = (this.report.studies || []).find(s => s.name === this.studies[si].name)?.loads || {};
    return Object.entries(loads).map(([name, series]) => ({
      name,
      value: series[frame],
      unit: meta[name]?.unit || (this.joints.get(name)?.kind === 'prismatic' ? 'N' : 'N·m'),
      capacity: meta[name]?.capacity ?? null,
    }));
  }

  /** Probe polylines and the current probe positions for a study. */
  probes(si) {
    if (si === null) return [];
    const st = this.studies[si];
    return Object.entries(st.probes || {}).map(([name, pts]) => ({ name, pts }));
  }

  /**
   * Frame whose joint values are nearest the requested ones. Revolute joints compare
   * modulo 360° so q=j:-90 finds 270° in a full turn. If no study is pinned, the first study
   * in which every requested joint actually moves is used.
   */
  nearestFrame(q, si) {
    for (const { joint } of q) if (!this.joints.has(joint)) throw unknown('joint in q=', joint, [...this.joints.keys()]);
    const moves = st => q.every(({ joint }) => {
      const s = st.joints?.[joint];
      return s && Math.max(...s) - Math.min(...s) > 1e-9;
    });
    let study = si;
    if (study === null) {
      study = this.studies.findIndex(moves);
      if (study < 0) throw new Error(`no study moves ${q.map(x => x.joint).join(', ')}`);
    }
    const st = this.studies[study];
    let best = 0;
    let bestErr = Infinity;
    for (let f = 0; f < st.frames; f++) {
      let err = 0;
      for (const { joint, value } of q) {
        const s = st.joints?.[joint];
        const v = s ? s[f] : this.joints.get(joint).home;
        let d = v - value;
        if (this.joints.get(joint).kind === 'revolute') d = ((d % 360) + 540) % 360 - 180;
        err += d * d;
      }
      if (err < bestErr - 1e-12) {
        bestErr = err;
        best = f;
      }
    }
    return { study, frame: best };
  }

  /** N frames evenly spaced over the study (inclusive of both ends). */
  ghostFrames(si, n) {
    const count = this.frameCount(si);
    if (!n || count < 2) return [];
    if (n === 1) return [0];
    const set = new Set();
    for (let i = 0; i < n; i++) set.add(Math.round((i * (count - 1)) / (n - 1)));
    return [...set];
  }

  /** Parts that move in a study (ghosting static parts only adds overdraw). */
  movingParts(si) {
    if (si === null) return [];
    return [...this._matrices[si].keys()];
  }
}

export function formatJointValue(joint, v) {
  if (v === null || v === undefined) return '—';
  return joint.kind === 'revolute' ? `${fmt(v, 4)}°` : `${fmt(v, 4)} mm`;
}

/** Compact number formatting (≈ significant figures, no exponent for normal ranges). */
export function fmt(v, sig = 3) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e6) {
    const digits = Math.max(0, sig - 1 - Math.floor(Math.log10(a)));
    return v.toFixed(Math.min(digits, 6)).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  }
  return v.toExponential(sig - 1);
}
