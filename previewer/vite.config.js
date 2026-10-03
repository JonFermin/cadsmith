import { defineConfig, loadEnv } from 'vite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const outputDir = path.resolve(here, '../output');

const CONTENT_TYPES = {
  '.json': 'application/json; charset=utf-8',
  '.stl': 'application/octet-stream',
  '.step': 'application/octet-stream',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
  '.scad': 'text/plain; charset=utf-8',
  '.py': 'text/plain; charset=utf-8',
};

function sendJson(res, value) {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(value));
}

/** `output/*.mech/` directories, newest first, with the status from their report.json. */
function listMechs() {
  let entries;
  try {
    entries = fs.readdirSync(outputDir, { withFileTypes: true });
  } catch {
    return [];
  }
  return entries
    .filter(e => e.isDirectory() && e.name.endsWith('.mech'))
    .map(e => {
      const dir = path.join(outputDir, e.name);
      const item = { name: e.name.slice(0, -'.mech'.length), status: null, mtime: 0 };
      try {
        const reportPath = path.join(dir, 'report.json');
        item.mtime = fs.statSync(reportPath).mtimeMs;
        item.status = JSON.parse(fs.readFileSync(reportPath, 'utf-8')).status ?? null;
      } catch {
        item.mtime = fs.statSync(dir).mtimeMs;
      }
      return item;
    })
    .sort((a, b) => b.mtime - a.mtime);
}

/** Static file server for ../output mounted at /output, with a path-traversal guard. */
function serveOutput(req, res, next) {
  let rel;
  try {
    rel = decodeURIComponent((req.url || '/').split('?')[0]);
  } catch {
    res.statusCode = 400;
    res.end('Bad Request');
    return;
  }
  const filePath = path.resolve(path.join(outputDir, rel));
  if (!filePath.startsWith(outputDir + path.sep)) {
    res.statusCode = 403;
    res.end('Forbidden');
    return;
  }
  fs.stat(filePath, (err, stat) => {
    if (err || !stat.isFile()) {
      next();
      return;
    }
    const type = CONTENT_TYPES[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
    res.setHeader('Content-Type', type);
    res.setHeader('Content-Length', stat.size);
    res.setHeader('Cache-Control', 'no-store'); // re-runs overwrite files in place
    fs.createReadStream(filePath).pipe(res);
  });
}

function outputApiPlugin() {
  const install = server => {
    server.middlewares.use('/api/manifests', (_req, res) => {
      try {
        sendJson(res, fs.readdirSync(outputDir).filter(f => f.endsWith('_manifest.json')).sort());
      } catch {
        sendJson(res, []);
      }
    });
    server.middlewares.use('/api/mechs', (_req, res) => sendJson(res, listMechs()));
    server.middlewares.use('/output', serveOutput);
  };
  return { name: 'cadsmith-output', configureServer: install, configurePreviewServer: install };
}

export default defineConfig(({ mode, command }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    // Relative asset URLs so dist/ works from any static server (mech shot mounts it at /).
    base: command === 'build' ? './' : '/',
    server: {
      open: true,
      port: 3000,
      host: true,
      allowedHosts: env.VITE_ALLOWED_HOST ? [env.VITE_ALLOWED_HOST] : [],
    },
    build: {
      chunkSizeWarningLimit: 700, // three.module is ~510 kB on its own
      rollupOptions: {
        input: {
          index: path.resolve(here, 'index.html'),
          mech: path.resolve(here, 'mech.html'),
        },
      },
    },
    plugins: [outputApiPlugin()],
  };
});
