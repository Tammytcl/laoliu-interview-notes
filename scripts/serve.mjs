import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../dist/', import.meta.url));
const port = Number(process.env.PORT || 4173);
const base = (process.env.BASE_PATH || '').replace(/\/$/, '');
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json', '.svg': 'image/svg+xml', '.md': 'text/plain' };
http.createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (base && !pathname.startsWith(base + '/')) { res.writeHead(404).end(); return; }
    const subpath = pathname.slice(base.length);
    const file = resolve(root, `.${subpath === '/' ? '/index.html' : subpath}`);
    if (!file.startsWith(resolve(root) + sep)) { res.writeHead(403).end(); return; }
    const data = await readFile(file);
    res.writeHead(200, { 'Content-Type': `${mime[extname(file)] || 'application/octet-stream'}; charset=utf-8`, 'Cache-Control': 'no-store' });
    res.end(data);
  } catch { res.writeHead(404).end('Not found. Run npm run build first.'); }
}).listen(port, '127.0.0.1', () => console.log(`Preview: http://127.0.0.1:${port}${base}/ (Ctrl+C 停止)`));
