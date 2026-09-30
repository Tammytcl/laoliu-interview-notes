import { readdir, readFile, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parsePaperContent } from './papers-content.mjs';

export const normalizedTitle = title => title.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
export function matchOpenAlex(paper, work) {
  if (normalizedTitle(work.title || '') !== normalizedTitle(paper.paperTitle)) throw new Error('title-mismatch');
  if (!/^https:\/\/openalex\.org\/W\d+$/.test(work.id || '')) throw new Error('invalid-work-id');
  if (!Number.isInteger(work.cited_by_count) || work.cited_by_count < 0) throw new Error('invalid-count');
  const surname = normalizedTitle(paper.authors[0].split(' ').at(-1));
  if (!work.authorships?.some(a => normalizedTitle(a.author?.display_name || '').includes(surname))) throw new Error('author-mismatch');
  return { count: work.cited_by_count, url: work.id, matchedTitle: work.title };
}
export function matchScholar(paper, data) {
  const results = (data.organic_results || []).filter(r => normalizedTitle(r.title || '') === normalizedTitle(paper.paperTitle));
  if (results.length !== 1) throw new Error('ambiguous-or-missing-title');
  const result = results[0];
  const surname = normalizedTitle(paper.authors[0].split(' ').at(-1));
  if (!normalizedTitle(result.publication_info?.summary || '').includes(surname)) throw new Error('author-mismatch');
  const citing = result.inline_links?.cited_by;
  if (!Number.isInteger(citing?.total) || citing.total < 0) throw new Error('missing-citation-count');
  const url = new URL(citing.link);
  if (url.protocol !== 'https:' || url.hostname !== 'scholar.google.com' || url.username || url.password) throw new Error('invalid-citation-url');
  return { count: citing.total, url: url.href, matchedTitle: result.title };
}
async function jsonRequest(url, fetcher) {
  const response = await fetcher(url, { signal: AbortSignal.timeout(25000), headers: { 'User-Agent': 'laoliu-paper-library/1.0' } });
  if (!response.ok) throw new Error(`http-${response.status}`);
  return response.json();
}
export async function updateCitations(papers, previous, { fetcher = fetch, now = new Date(), openalexKey = '', scholarKey = '', force = false } = {}) {
  const output = structuredClone(previous);
  for (const paper of papers) {
    const entry = output[paper.id] ||= {};
    const providers = ['openalex', ...(scholarKey ? ['google-scholar'] : [])];
    for (const provider of providers) {
      const checked = entry.attempts?.[provider]?.checkedAt;
      if (!force && checked && now - new Date(checked) < 7 * 86400000) continue;
      const attempted = { checkedAt: now.toISOString(), status: 'ok' };
      try {
        let metric;
        if (provider === 'openalex') {
          const arxiv = new URL(paper.paperUrl).pathname.match(/\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})/);
          const workId = entry.openalex?.url?.split('/').at(-1) || paper.openalexId || (arxiv ? `https://doi.org/10.48550/arxiv.${arxiv[1]}` : null);
          if (!workId) throw new Error('missing-work-identifier');
          const url = new URL(`https://api.openalex.org/works/${workId}`);
          if (openalexKey) url.searchParams.set('api_key', openalexKey);
          metric = matchOpenAlex(paper, await jsonRequest(url, fetcher));
        } else {
          const url = new URL('https://serpapi.com/search.json');
          url.search = new URLSearchParams({ engine: 'google_scholar', q: `"${paper.paperTitle}"`, api_key: scholarKey, num: '5', hl: 'en' });
          metric = matchScholar(paper, await jsonRequest(url, fetcher));
        }
        entry[provider] = { ...metric, fetchedAt: now.toISOString() };
      } catch (error) {
        attempted.status = 'failed';
        // Do not log exception messages or request URLs: they may contain API keys.
        attempted.reason = /^(http-\d+|title-mismatch|author-mismatch|invalid-work-id|invalid-count|missing-work-identifier|ambiguous-or-missing-title|missing-citation-count|invalid-citation-url)$/.test(error.message) ? error.message : 'network-or-response-error';
      }
      (entry.attempts ||= {})[provider] = attempted;
    }
  }
  return output;
}
async function walk(dir) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walk(path));
    else if (entry.name.endsWith('.md')) files.push(path);
  }
  return files.sort();
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const papers = [];
  for (const file of await walk(join(root, 'content/papers'))) {
    const paper = parsePaperContent(await readFile(file, 'utf8'), file, 'papers');
    if (!paper.draft) papers.push(paper);
  }
  const path = join(root, 'content/metadata/citations.json');
  let previous = {};
  try { previous = JSON.parse(await readFile(path, 'utf8')); } catch (e) { if (e.code !== 'ENOENT') throw e; }
  const updated = await updateCitations(papers, previous, { openalexKey: process.env.OPENALEX_API_KEY, scholarKey: process.env.SERPAPI_API_KEY, force: process.argv.includes('--force') });
  await mkdir(join(root, 'content/metadata'), { recursive: true });
  await writeFile(path, JSON.stringify(updated, null, 2) + '\n');
  for (const paper of papers) for (const [provider, result] of Object.entries(updated[paper.id].attempts || {})) console.log(`${paper.id} · ${provider}: ${result.status}${result.reason ? ` (${result.reason})` : ''}`);
}
