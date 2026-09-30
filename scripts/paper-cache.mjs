import { lstat, readdir, rm, mkdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
export async function paperCacheStatus(project = root) {
  const cache = join(project, '.paper-cache');
  async function walk(path) {
    let stat; try { stat = await lstat(path); } catch (e) { if (e.code === 'ENOENT') return { bytes: 0, files: 0 }; throw e; }
    if (stat.isSymbolicLink()) throw new Error('缓存中含有符号链接，请手动检查');
    if (stat.isFile()) return { bytes: stat.size, files: 1 };
    if (!stat.isDirectory()) throw new Error('缓存包含非普通文件');
    const result = { bytes: 0, files: 0 };
    for (const name of await readdir(path)) { const next = await walk(join(path, name)); result.bytes += next.bytes; result.files += next.files; }
    return result;
  }
  return { ...(await walk(cache)), path: '.paper-cache', preserved: ['content/papers', 'content/reports', 'assets/papers'] };
}
export async function cleanPaperCache(project = root) {
  const before = await paperCacheStatus(project);
  await rm(join(project, '.paper-cache'), { recursive: true, force: true });
  await mkdir(join(project, '.paper-cache'), { recursive: true });
  return { freedBytes: before.bytes, deletedFiles: before.files, preserved: before.preserved };
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = process.argv.includes('--clean') ? await cleanPaperCache() : await paperCacheStatus();
  console.log(JSON.stringify(result, null, 2));
}
