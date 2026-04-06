import * as THREE from 'three';

// --- Error Display ---
function showError(message) {
  const toast = document.getElementById('error-toast');
  toast.textContent = message;
  toast.style.display = 'block';
  clearTimeout(showError._timer);
  showError._timer = setTimeout(() => { toast.style.display = 'none'; }, 5000);
}

// --- State ---
let scene, camera, renderer, modelGroup;
let manifest = null;

// Orbit control state
const orbit = {
  isDragging: false,
  isPanning: false,
  prevX: 0,
  prevY: 0,
  theta: Math.PI / 4,   // horizontal angle
  phi: Math.PI / 6,     // vertical angle (from top)
  distance: 150,
  target: new THREE.Vector3(0, 0, 0),
  minDistance: 5,
  maxDistance: 2000,
};

// --- Init ---
function init() {
  const canvas = document.getElementById('viewport');
  const container = document.getElementById('canvas-container');

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x1a1a2e);
  renderer.shadowMap.enabled = true;

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(50, 1, 0.1, 10000);
  updateCameraPosition();

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambient);

  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(50, 100, 50);
  dirLight.castShadow = true;
  scene.add(dirLight);

  const dirLight2 = new THREE.DirectionalLight(0x8888ff, 0.3);
  dirLight2.position.set(-50, 30, -50);
  scene.add(dirLight2);

  // Grid ground plane
  const grid = new THREE.GridHelper(200, 20, 0x0f3460, 0x0a0a20);
  scene.add(grid);

  modelGroup = new THREE.Group();
  scene.add(modelGroup);

  // Event listeners
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });

  // Touch support
  canvas.addEventListener('touchstart', onTouchStart, { passive: false });
  canvas.addEventListener('touchmove', onTouchMove, { passive: false });
  canvas.addEventListener('touchend', onTouchEnd);

  // Manifest dropdown
  const select = document.getElementById('manifest-select');
  select.addEventListener('change', onManifestSelect);
  loadManifestList();

  // Drag and drop
  const dropTarget = container;
  dropTarget.addEventListener('dragover', (e) => {
    e.preventDefault();
    document.getElementById('drop-overlay').classList.add('active');
  });
  dropTarget.addEventListener('dragleave', () => {
    document.getElementById('drop-overlay').classList.remove('active');
  });
  dropTarget.addEventListener('drop', onDrop);

  // Resize
  window.addEventListener('resize', onResize);
  onResize();

  animate();
}

// --- Orbit Controls (manual) ---
function updateCameraPosition() {
  const sinPhi = Math.sin(orbit.phi);
  const cosPhi = Math.cos(orbit.phi);
  const sinTheta = Math.sin(orbit.theta);
  const cosTheta = Math.cos(orbit.theta);

  camera.position.set(
    orbit.target.x + orbit.distance * cosPhi * sinTheta,
    orbit.target.y + orbit.distance * sinPhi,
    orbit.target.z + orbit.distance * cosPhi * cosTheta
  );
  camera.lookAt(orbit.target);
}

function onPointerDown(e) {
  orbit.isDragging = true;
  orbit.isPanning = e.shiftKey;
  orbit.prevX = e.clientX;
  orbit.prevY = e.clientY;
  e.target.setPointerCapture(e.pointerId);
}

function onPointerMove(e) {
  if (!orbit.isDragging) return;
  const dx = e.clientX - orbit.prevX;
  const dy = e.clientY - orbit.prevY;
  orbit.prevX = e.clientX;
  orbit.prevY = e.clientY;

  if (orbit.isPanning) {
    // Pan: move target in camera-local XY plane
    const panSpeed = orbit.distance * 0.002;
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    camera.getWorldDirection(new THREE.Vector3());
    right.crossVectors(camera.up, new THREE.Vector3().subVectors(camera.position, orbit.target).normalize()).normalize();
    up.crossVectors(new THREE.Vector3().subVectors(camera.position, orbit.target).normalize(), right).normalize();
    orbit.target.add(right.multiplyScalar(-dx * panSpeed));
    orbit.target.add(up.multiplyScalar(dy * panSpeed));
  } else {
    // Rotate
    orbit.theta -= dx * 0.005;
    orbit.phi += dy * 0.005;
    orbit.phi = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, orbit.phi));
  }
  updateCameraPosition();
}

function onPointerUp(e) {
  orbit.isDragging = false;
  orbit.isPanning = false;
}

function onWheel(e) {
  e.preventDefault();
  orbit.distance *= 1 + e.deltaY * 0.001;
  orbit.distance = Math.max(orbit.minDistance, Math.min(orbit.maxDistance, orbit.distance));
  updateCameraPosition();
}

// Touch state for pinch zoom
let touchStartDist = 0;
let touchStartOrbitDist = 0;

function onTouchStart(e) {
  if (e.touches.length === 1) {
    orbit.isDragging = true;
    orbit.prevX = e.touches[0].clientX;
    orbit.prevY = e.touches[0].clientY;
  } else if (e.touches.length === 2) {
    e.preventDefault();
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    touchStartDist = Math.sqrt(dx * dx + dy * dy);
    touchStartOrbitDist = orbit.distance;
  }
}

function onTouchMove(e) {
  if (e.touches.length === 1 && orbit.isDragging) {
    const dx = e.touches[0].clientX - orbit.prevX;
    const dy = e.touches[0].clientY - orbit.prevY;
    orbit.prevX = e.touches[0].clientX;
    orbit.prevY = e.touches[0].clientY;
    orbit.theta -= dx * 0.005;
    orbit.phi += dy * 0.005;
    orbit.phi = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, orbit.phi));
    updateCameraPosition();
  } else if (e.touches.length === 2) {
    e.preventDefault();
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (touchStartDist > 0) {
      orbit.distance = touchStartOrbitDist * (touchStartDist / dist);
      orbit.distance = Math.max(orbit.minDistance, Math.min(orbit.maxDistance, orbit.distance));
      updateCameraPosition();
    }
  }
}

function onTouchEnd() {
  orbit.isDragging = false;
  touchStartDist = 0;
}

// --- Resize ---
function onResize() {
  const container = document.getElementById('canvas-container');
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

// --- Manifest List Loading ---
function loadManifestList() {
  fetch('/api/manifests')
    .then(r => r.json())
    .then(files => {
      const select = document.getElementById('manifest-select');
      files.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = f.replace('_manifest.json', '').replace(/_/g, ' ');
        select.appendChild(opt);
      });
      // Auto-select if URL param matches or only one manifest
      const params = new URLSearchParams(window.location.search);
      const manifestParam = params.get('manifest');
      if (manifestParam) {
        const match = files.find(f => manifestParam.includes(f));
        if (match) {
          select.value = match;
          fetchAndLoadManifest(match);
        }
      } else if (files.length === 1) {
        select.value = files[0];
        fetchAndLoadManifest(files[0]);
      }
    })
    .catch(() => {});
}

function onManifestSelect(e) {
  const file = e.target.value;
  if (!file) return;
  fetchAndLoadManifest(file);
}

function fetchAndLoadManifest(filename) {
  fetch(`/output/${filename}`)
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then(data => loadManifest(data))
    .catch(err => {
      console.error('Failed to load manifest:', err);
      showError(`Failed to load ${filename}`);
    });
}

function onDrop(e) {
  e.preventDefault();
  document.getElementById('drop-overlay').classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.json')) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        loadManifest(data);
      } catch (err) {
        console.error('Failed to parse manifest:', err);
        showError('Failed to parse manifest file. Check that it is valid JSON.');
      }
    };
    reader.readAsText(file);
  }
}

// --- Manifest → Scene ---
function loadManifest(data) {
  manifest = data;

  // Clear existing model
  while (modelGroup.children.length > 0) {
    const child = modelGroup.children[0];
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
    modelGroup.remove(child);
  }

  // Hide empty state
  document.getElementById('empty-state').style.display = 'none';

  const legendParts = [];

  for (const part of data.parts) {
    const mesh = createPartMesh(part);
    if (!mesh) continue;
    modelGroup.add(mesh);
    legendParts.push(part);
  }

  // Auto-frame camera
  autoFrame();

  // Build legend
  buildLegend(legendParts);

  // Update grid to match model scale
  updateGrid();
}

function createPartMesh(part) {
  let geometry;
  const p = part.params || {};

  switch (part.type) {
    case 'box':
      geometry = new THREE.BoxGeometry(
        p.width || 10, p.height || 10, p.depth || 10
      );
      break;

    case 'sphere':
      geometry = new THREE.SphereGeometry(
        p.radius || 5, p.segments || 32, p.segments || 32
      );
      break;

    case 'cylinder':
      geometry = new THREE.CylinderGeometry(
        p.radiusTop != null ? p.radiusTop : (p.radius || 5),
        p.radiusBottom != null ? p.radiusBottom : (p.radius || 5),
        p.height || 10,
        p.segments || 32
      );
      break;

    case 'extrude': {
      if (!p.shape || p.shape.length < 3) return null;
      const shape = new THREE.Shape();
      shape.moveTo(p.shape[0][0], p.shape[0][1]);
      for (let i = 1; i < p.shape.length; i++) {
        shape.lineTo(p.shape[i][0], p.shape[i][1]);
      }
      shape.closePath();
      geometry = new THREE.ExtrudeGeometry(shape, {
        depth: p.height || 10,
        bevelEnabled: false,
      });
      break;
    }

    case 'lathe': {
      if (!p.points || p.points.length < 2) return null;
      const lathePoints = p.points.map(pt => new THREE.Vector2(pt[0], pt[1]));
      geometry = new THREE.LatheGeometry(lathePoints, p.segments || 32);
      break;
    }

    default:
      console.warn(`Unknown part type: ${part.type}`);
      return null;
  }

  const color = new THREE.Color(part.color || '#4488cc');
  const isDifference = part.operation === 'difference';

  let material;
  if (isDifference || part.wireframe) {
    material = new THREE.MeshBasicMaterial({
      color,
      wireframe: true,
      transparent: true,
      opacity: part.opacity != null ? part.opacity : 0.3,
    });
  } else {
    material = new THREE.MeshStandardMaterial({
      color,
      transparent: part.opacity != null && part.opacity < 1,
      opacity: part.opacity != null ? part.opacity : 1.0,
      metalness: 0.1,
      roughness: 0.6,
    });
  }

  const mesh = new THREE.Mesh(geometry, material);

  if (part.position) {
    mesh.position.set(part.position[0], part.position[1], part.position[2]);
  }
  if (part.rotation) {
    mesh.rotation.set(
      part.rotation[0] * Math.PI / 180,
      part.rotation[1] * Math.PI / 180,
      part.rotation[2] * Math.PI / 180
    );
  }
  if (part.scale) {
    mesh.scale.set(part.scale[0], part.scale[1], part.scale[2]);
  }

  mesh.userData = { partId: part.id, label: part.label };
  return mesh;
}

function autoFrame() {
  const box = new THREE.Box3().setFromObject(modelGroup);
  if (box.isEmpty()) return;

  const center = new THREE.Vector3();
  box.getCenter(center);
  orbit.target.copy(center);

  const size = new THREE.Vector3();
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);

  orbit.distance = maxDim * 2.5;
  orbit.theta = Math.PI / 4;
  orbit.phi = Math.PI / 6;
  updateCameraPosition();
}

function updateGrid() {
  // Replace grid to match model scale
  const existing = scene.children.find(c => c.type === 'GridHelper');
  if (existing) scene.remove(existing);

  const box = new THREE.Box3().setFromObject(modelGroup);
  if (box.isEmpty()) return;

  const size = new THREE.Vector3();
  box.getSize(size);
  const gridSize = Math.max(size.x, size.z, 200) * 2;
  const grid = new THREE.GridHelper(gridSize, 20, 0x0f3460, 0x0a0a20);
  scene.add(grid);
}

function buildLegend(parts) {
  const legend = document.getElementById('legend');
  if (parts.length === 0) {
    legend.innerHTML = '';
    return;
  }

  let html = '<h3>Parts</h3>';
  for (const part of parts) {
    const isDiff = part.operation === 'difference';
    html += `<div class="legend-item">
      <div class="legend-swatch" style="background:${part.color || '#4488cc'};${isDiff ? 'border:1px dashed #ff4444;background:transparent;' : ''}"></div>
      <span class="legend-label">${part.label || part.id || 'unnamed'}</span>
      ${isDiff ? '<span class="legend-diff">(subtracted)</span>' : ''}
    </div>`;
  }
  legend.innerHTML = html;
}

// --- Animate ---
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

// --- Start ---
init();
