import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomBytes } from 'node:crypto';
import { paperCacheStatus, cleanPaperCache } from './paper-cache.mjs';
const root = fileURLToPath(new URL('../dist/', import.meta.url));
const port = Number(process.env.PORT || 4173);
const base = (process.env.BASE_PATH || '').replace(/\/$/, '');
const workbench = process.env.PAPER_WORKBENCH === '1';
const cacheToken = randomBytes(24).toString('hex');
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json', '.svg': 'image/svg+xml', '.md': 'text/plain', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf' };
const server = http.createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (base && !pathname.startsWith(base + '/')) { res.writeHead(404).end(); return; }
    const subpath = pathname.slice(base.length);
    if (subpath.startsWith('/api/paper-cache')) {
      const origin = `http://${req.headers.host}`;
      const localHost = new URL(origin).hostname;
      if (!workbench) { res.writeHead(404).end(); return; }
      if (!['localhost', '127.0.0.1', '[::1]'].includes(localHost) || (req.headers.origin && req.headers.origin !== origin)) { res.writeHead(403).end(); return; }
      res.setHeader('Content-Type', 'application/json'); res.setHeader('Cache-Control', 'no-store');
      if (subpath === '/api/paper-cache' && req.method === 'GET') { res.end(JSON.stringify({ ...await paperCacheStatus(), token: cacheToken })); return; }
      if (subpath === '/api/paper-cache/clean' && req.method === 'POST' && req.headers['x-paper-cache-token'] === cacheToken) { res.end(JSON.stringify(await cleanPaperCache())); return; }
      res.writeHead(403).end(JSON.stringify({ error: '请求无效' })); return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405).end(); return; }
    const file = resolve(root, `.${subpath === '/' ? '/index.html' : subpath}`);
    if (!file.startsWith(resolve(root) + sep)) { res.writeHead(403).end(); return; }
    const data = await readFile(file);
    res.writeHead(200, { 'Content-Type': `${mime[extname(file)] || 'application/octet-stream'}; charset=utf-8`, 'Cache-Control': 'no-store' });
    res.end(data);
  } catch { res.writeHead(404).end('Not found. Run npm run build first.'); }
}).listen(port, '127.0.0.1', () => console.log(`Preview: http://127.0.0.1:${server.address().port}${base}/ (Ctrl+C 停止)`));
