import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';

function manifestListPlugin() {
  const outputDir = path.resolve(__dirname, '../output');
  return {
    name: 'manifest-list',
    configureServer(server) {
      server.middlewares.use('/api/manifests', (_req, res) => {
        try {
          const files = fs.readdirSync(outputDir)
            .filter(f => f.endsWith('_manifest.json'))
            .sort();
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(files));
        } catch {
          res.setHeader('Content-Type', 'application/json');
          res.end('[]');
        }
      });
      // Serve output files at /output/
      server.middlewares.use('/output', (req, res, next) => {
        const requested = path.join(outputDir, decodeURIComponent(req.url));
        const filePath = path.resolve(requested);
        if (!filePath.startsWith(outputDir + path.sep) && filePath !== outputDir) {
          res.statusCode = 403;
          res.end('Forbidden');
          return;
        }
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
          res.setHeader('Content-Type', 'application/json');
          res.end(fs.readFileSync(filePath, 'utf-8'));
        } else {
          next();
        }
      });
    },
  };
}

export default defineConfig({
  server: {
    open: true,
    port: 3000,
  },
  plugins: [manifestListPlugin()],
});
