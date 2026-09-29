// Viewer state and everything derived from it: pose, highlighting, overlays, camera.
import { Box3, Plane, Vector3 } from 'three';
import { PartSet } from './parts.js';
import { IssueMarkers, JointAxes, ProbePaths, SectionPlane } from './overlays.js';
import { camDirection, viewDirection } from './viewer.js';

const SECTION_KEEP = { x: -1, y: 1, z: -1 }; // default kept side faces the iso/front/top camera
const ISSUE_STATUS = { interference: 'interference', static_interference: 'interference', tight_clearance: 'tight' };

export class MechApp {
  /**
   * @param {import('./viewer.js').Viewer} viewer
   * @param {import('./model.js').MechModel} model
   * @param {Map<string, import('three').BufferGeometry>} geometries
   */
  constructor(viewer, model, geometries) {
    this.viewer = viewer;
    this.model = model;
    this.parts = new PartSet(model, geometries);
    viewer.content.add(this.parts.group, this.parts.ghostGroup);
    this.issueMarkers = new IssueMarkers(viewer);
    this.probePaths = new ProbePaths(viewer);
    this.jointAxes = new JointAxes(viewer);
    this.sectionPlane = new SectionPlane(viewer);
    this.probePart = new Map((model.scene.probes || []).map(p => [p.name, p.part]));
    this.listeners = [];

    this.state = {
      si: model.studies.length ? 0 : null,
      frame: model.studies.length ? 0 : null, // null = home pose
      hidden: new Set(),
      isolate: new Set(),
      issue: null,
      section: null,
      explode: 0,
      axes: false,
      probes: true,
      ghost: 0,
      layout: 'single',
      view: null,
      playing: false,
      speed: 1,
      time: 0,
    };

    // Bounds over the home pose and every study's motion envelope: grid, lights, section range.
    this.bounds = this.parts.envelope(null, model.partIds);
    model.studies.forEach((_, si) => this.bounds.union(this.parts.envelope(si, model.partIds)));
    viewer.setBounds(this.bounds);
    this.size = this.bounds.getSize(new Vector3()).length();
  }

  onChange(fn) {
    this.listeners.push(fn);
  }

  _emit(what) {
    for (const fn of this.listeners) fn(what, this.state);
  }

  // ------------------------------------------------------------------ initial URL state

  /** Apply parsed URL params (explicit params override what `issue=` implies). */
  init(p) {
    const s = this.state;
    const m = this.model;
    let focus = [];
    if (p.issue !== null) {
      if (p.issue >= m.issues.length) throw new Error(`issue=${p.issue} out of range (report has ${m.issues.length} issues)`);
      const iss = m.issues[p.issue];
      this._selectIssue(p.issue);
      focus = this._issueParts(iss);
    }
    if (p.study) {
      s.si = m.studyIndex(p.study);
      if (p.issue === null) s.frame = 0;
    }
    if (p.q.length) {
      const hit = m.nearestFrame(p.q, p.study ? s.si : null);
      s.si = hit.study;
      s.frame = hit.frame;
    }
    if (p.frame !== null) {
      if (p.frame === 'home') s.frame = null;
      else {
        if (s.si === null) throw new Error('frame= given but the scene has no studies');
        const n = m.frameCount(s.si);
        if (p.frame >= n) throw new Error(`frame=${p.frame} out of range 0..${n - 1} for study '${m.studies[s.si].name}'`);
        s.frame = p.frame;
      }
    }
    s.hidden = new Set(m.checkParts(p.hide, 'hide'));
    if (p.isolate.length) s.isolate = new Set(m.checkParts(p.isolate, 'isolate'));
    if (p.focus.length) focus = m.checkParts(p.focus, 'focus');
    s.explode = p.explode;
    s.axes = p.axes;
    s.ghost = p.ghost;
    s.layout = p.layout;
    s.view = p.view;
    if (p.section) this._setSection(p.section);
    if (s.si !== null) s.time = m.studies[s.si].t?.[s.frame ?? 0] ?? 0;

    this.viewer.setLayout(s.layout);
    this.refresh({ ghosts: true });
    const dir = p.cam ? camDirection(p.cam.az, p.cam.el) : viewDirection(p.view || 'iso');
    this.viewer.setMainLabel(p.cam ? `az ${p.cam.az}° el ${p.cam.el}°` : p.view || 'iso');
    this.fit(focus.length ? focus : null, dir);
  }

  _issueParts(iss) {
    return (iss.parts || []).filter(id => this.model.partIds.includes(id));
  }

  // ------------------------------------------------------------------ derived state

  /** Recompute pose, styling and overlays for the current state; `ghosts` rebuilds ghosts. */
  refresh({ ghosts = false } = {}) {
    const s = this.state;
    const { model, parts } = this;
    parts.setPose(s.si, s.frame, s.explode);

    const entries = model.pairIssuesAt(s.si, s.frame);
    const highlight = new Map();
    for (const e of entries) {
      if (e.status !== 'interference' && e.status !== 'tight') continue;
      for (const id of [e.a, e.b]) {
        if (highlight.get(id) !== 'interference') highlight.set(id, e.status);
      }
    }
    const selected = this._selectedMarker(entries);
    // A selected interference is shown x-ray style so the overlap inside the parts is visible.
    const xray = new Set(selected?.status === 'interference' ? selected.parts : []);
    parts.applyStyle({ hidden: s.hidden, isolate: s.isolate, highlight, xray, section: !!s.section });
    if (ghosts) parts.setGhosts(s.si, model.ghostFrames(s.si, s.ghost), s.hidden);

    const offsetOf = id => parts.offsetOf(id);
    this.issueMarkers.set(entries, { offsetOf, size: this.size, selected });
    this.probePaths.set(model.probes(s.si), s.frame, {
      offsetOf, partOf: n => this.probePart.get(n), size: this.size, visible: s.probes,
    });
    this.jointAxes.set([...model.joints.values()], {
      transformOf: id => model.transform(s.si, s.frame, id), offsetOf, size: this.size, visible: s.axes,
    });
    this.sectionPlane.set(s.section, this.bounds);
    this.viewer.requestRender();
    this._emit('pose');
  }

  /** Marker details for the selected report issue when the current pose is the issue's pose. */
  _selectedMarker(entries) {
    const s = this.state;
    if (s.issue === null) return null;
    const iss = this.model.issues[s.issue];
    const si = iss.study ? this.model.studies.findIndex(st => st.name === iss.study) : s.si;
    if (si !== s.si || (iss.frame ?? null) !== s.frame) return null;
    const parts = this._issueParts(iss);
    const entry = entries.find(e => (e.status === 'interference' || e.status === 'tight')
      && parts.includes(e.a) && parts.includes(e.b)) || null;
    let box = null;
    const a = parts[0];
    if (iss.location && a && (iss.extent || !entry)) {
      // location/extent are in part a's HOME frame; its current matrix carries them along.
      box = {
        center: new Vector3(...iss.location),
        size: new Vector3(...(iss.extent || [0, 0, 0])),
        matrix: this.parts.items.get(a).mesh.matrix,
      };
    }
    const status = ISSUE_STATUS[iss.code] || null;
    return { entry, box, status, parts, text: `${iss.code} · ${parts.join(' / ')}` };
  }

  // ------------------------------------------------------------------ actions (UI + keyboard)

  setStudy(si) {
    const s = this.state;
    s.si = si;
    s.frame = 0;
    s.time = 0;
    this.refresh({ ghosts: true });
    this._emit('study');
  }

  /** Jump to a frame index, or null for the home pose. */
  setFrame(frame) {
    const s = this.state;
    if (frame !== null) {
      const n = this.model.frameCount(s.si);
      if (!n) return;
      frame = ((frame % n) + n) % n;
      s.time = this.model.studies[s.si].t?.[frame] ?? 0;
    }
    s.frame = frame;
    this.refresh();
  }

  step(d) {
    const s = this.state;
    this.setPlaying(false);
    this.setFrame((s.frame ?? 0) + d);
  }

  setPlaying(on) {
    const s = this.state;
    if (on && (s.si === null || this.model.frameCount(s.si) < 2)) return;
    s.playing = on;
    if (on && s.frame === null) s.frame = 0;
    this._emit('play');
  }

  setSpeed(speed) {
    this.state.speed = speed;
    this._emit('play');
  }

  /** Advance playback by dt seconds of wall time; wraps at the end of the study. */
  tick(dt) {
    const s = this.state;
    if (!s.playing) return;
    const st = this.model.studies[s.si];
    const t = st.t;
    const n = st.frames;
    // One frame interval past the last sample, so `once` studies dwell on the final pose and
    // pingpong studies (pre-expanded, t_N would equal t_0 + duration) wrap seamlessly.
    const period = t[n - 1] + (t[n - 1] - t[0]) / (n - 1);
    s.time = (s.time + dt * s.speed) % period;
    let f = 0;
    while (f + 1 < n && t[f + 1] <= s.time) f++;
    if (f !== s.frame) {
      s.frame = f;
      this.refresh();
    }
  }

  selectIssue(i) {
    const iss = this.model.issues[i];
    this.setPlaying(false);
    this._selectIssue(i);
    this.refresh({ ghosts: true });
    const parts = this._issueParts(iss);
    this.fit(parts.length ? parts : null);
    this._emit('isolate');
  }

  _selectIssue(i) {
    const s = this.state;
    const iss = this.model.issues[i];
    s.issue = i;
    if (iss.study) s.si = this.model.studyIndex(iss.study);
    s.frame = iss.frame ?? null;
    const parts = this._issueParts(iss);
    s.isolate = new Set(parts);
  }

  clearIsolate() {
    this.state.isolate = new Set();
    this.state.issue = null;
    this.refresh();
    this._emit('isolate');
  }

  setIsolate(ids) {
    this.state.isolate = new Set(ids);
    this.refresh();
    this._emit('isolate');
  }

  setHidden(id, hidden) {
    if (hidden) this.state.hidden.add(id);
    else this.state.hidden.delete(id);
    this.refresh({ ghosts: true });
  }

  setExplode(x) {
    this.state.explode = x;
    this.refresh();
  }

  setAxes(on) {
    this.state.axes = on;
    this.refresh();
  }

  setProbes(on) {
    this.state.probes = on;
    this.refresh();
  }

  setGhost(n) {
    this.state.ghost = n;
    this.refresh({ ghosts: true });
  }

  setLayout(layout) {
    this.state.layout = layout;
    this.viewer.setLayout(layout);
    this.viewer.requestRender();
    this._emit('layout');
  }

  /** section: {axis, offset|null, flip} or null. */
  setSection(section) {
    this._setSection(section);
    this.refresh({ ghosts: true });
  }

  _setSection(section) {
    const s = this.state;
    if (!section) {
      s.section = null;
      this.parts.setClipping([]);
      return;
    }
    const center = this.bounds.getCenter(new Vector3());
    const offset = section.offset ?? center[section.axis];
    s.section = { axis: section.axis, offset, flip: !!section.flip };
    const sign = SECTION_KEEP[section.axis] * (section.flip ? -1 : 1);
    const normal = new Vector3();
    normal[section.axis] = sign;
    // Plane keeps points with normal·p + constant ≥ 0, i.e. sign·(p_axis − offset) ≥ 0.
    this.parts.setClipping([new Plane(normal, -sign * offset)]);
  }

  setView(name) {
    this.state.view = name;
    this.viewer.setMainLabel(name);
    this.fit(this._focusIds(), viewDirection(name));
  }

  _focusIds() {
    return this.state.isolate.size ? [...this.state.isolate] : null;
  }

  /**
   * Fit the camera: explicit ids → their current bbox; otherwise the visible parts' motion
   * envelope over the current study plus their current (possibly exploded) boxes.
   */
  fit(ids = this._focusIds(), dir = null) {
    const { parts, state: s } = this;
    let box;
    if (ids && ids.length) {
      box = parts.currentBox(ids);
    } else {
      box = parts.envelope(s.si).union(parts.currentBox());
      if (box.isEmpty()) box = this.bounds.clone();
    }
    if (box.isEmpty()) box = new Box3(new Vector3(-10, -10, -10), new Vector3(10, 10, 10));
    this.viewer.setOrthoBox(box);
    this.viewer.frame(box, dir);
  }
}
