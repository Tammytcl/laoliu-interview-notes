import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const safeReason = error => /^http-\d+$/.test(error.message) ? error.message : 'network-or-response-error';

// An observation never changes the reviewed baseline or the article.
export async function observeSources(manifest, previous = {}, request = fetch, now = new Date().toISOString()) {
  const observations = { ...previous.observations };
  const jobs = manifest.sources.filter(source => source.watchUrl || source.repository);
  const repositoryRequests = new Map();
  const json = async url => {
    const headers = { Accept: 'application/vnd.github+json', 'User-Agent': 'laoliu-topic-source-watch' };
    if (process.env.GH_TOKEN && new URL(url).hostname === 'api.github.com') headers.Authorization = `Bearer ${process.env.GH_TOKEN}`;
    const response = await request(url, { headers, signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error(`http-${response.status}`);
    return response.json();
  };
  const observe = async source => {
    const prior = observations[source.id] || {};
    try {
      let revision, sha256;
      if (source.repository) {
        if (!repositoryRequests.has(source.repository)) {
          repositoryRequests.set(source.repository, json(`https://api.github.com/repos/${source.repository}/commits/HEAD`));
        }
        const result = await repositoryRequests.get(source.repository);
        if (!/^[a-f0-9]{40}$/.test(result.sha || '')) throw new Error('invalid-revision');
        revision = result.sha;
      }
      if (source.watchUrl) {
        let bytes;
        if (source.repository && source.watchPath) {
          const path = source.watchPath.split('/').map(encodeURIComponent).join('/');
          const file = await json(`https://api.github.com/repos/${source.repository}/contents/${path}?ref=${revision}`);
          if (file.encoding !== 'base64' || typeof file.content !== 'string') throw new Error('invalid-body');
          bytes = Buffer.from(file.content, 'base64');
        } else {
          const response = await request(source.watchUrl, { signal: AbortSignal.timeout(15000) });
          if (!response.ok) throw new Error(`http-${response.status}`);
          bytes = new Uint8Array(await response.arrayBuffer());
        }
        if (!bytes.length || bytes.length > 5_000_000) throw new Error('invalid-body');
        sha256 = hash(bytes);
      }
      const pendingReview = source.watchUrl
        ? sha256 !== source.reviewedSha256
        : revision !== source.revision;
      observations[source.id] = { checkedAt: now, lastSuccessAt: now, status: 'ok', revision, sha256, pendingReview };
    } catch (error) {
      // Preserve the last successful observation and pending status; a failed fetch is not evidence of no changes.
      observations[source.id] = { ...prior, checkedAt: now, status: 'error', reason: safeReason(error) };
    }
  };
  for (let i = 0; i < jobs.length; i += 4) await Promise.all(jobs.slice(i, i + 4).map(observe));
  return { version: 1, topicId: manifest.topicId, checkedAt: now, observations };
}

export function sourceReviewReport(manifest, snapshot) {
  const watched = manifest.sources.filter(s => s.watchUrl || s.repository);
  const pending = watched.filter(s => snapshot.observations[s.id]?.pendingReview);
  const failed = watched.filter(s => snapshot.observations[s.id]?.status === 'error');
  const lines = [`# ${manifest.topicId} 来源观察记录`, '', `自动检测时间：${snapshot.checkedAt}`, '',
    '这是来源变化的候选账本。正文核对日期、固定版本和结论不会由检测脚本覆盖。仓库 commit 更新可能与本文无关；需要阅读差异后再判断。', '',
    `观察 ${watched.length} 个来源条目；待核对 ${pending.length} 项；本轮失败 ${failed.length} 项。未观察的网页仍需在整理任务中核查。`, '',
    '## 待核对变化', ''];
  if (!pending.length) lines.push('当前成功观察没有新增待核对项；这不代表所有外部链接和事实永远有效。');
  for (const source of pending) {
    const current = snapshot.observations[source.id];
    const diff = source.repository && source.revision && current.revision
      ? `https://github.com/${source.repository}/compare/${source.revision}...${current.revision}` : source.url;
    lines.push(`- [${source.title}](${diff})：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。`);
  }
  lines.push('', '## 抓取失败', '');
  if (!failed.length) lines.push('本轮观察请求均成功。');
  for (const source of failed) lines.push(`- [${source.title}](${source.url})：${snapshot.observations[source.id].reason}；保留上次成功值，不能判断本轮是否有变化。`);
  lines.push('', '## 完成核对的方法', '',
    '按 docs/topic-workflow.md 读取来源差异，修订正文、引用、图解与题目，人工核对后更新 manifest 的 reviewedAt / revision / reviewedSha256。运行 topics:quality、测试、构建和浏览器验收，再提交推送；候选状态将在下一次观察中重新计算。', '');
  return lines.join('\n');
}

async function main() {
  const topics = JSON.parse(await readFile(join(root, 'content/research/topics.json'), 'utf8'));
  for (const topic of topics) {
    const manifest = JSON.parse(await readFile(join(root, topic.sources), 'utf8'));
    const path = join(root, `content/research/${topic.id}-observations.json`);
    let previous = {};
    try { previous = JSON.parse(await readFile(path, 'utf8')); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const snapshot = await observeSources(manifest, previous);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(snapshot, null, 2) + '\n');
    await writeFile(join(root, `content/research/${topic.id}-pending.md`), sourceReviewReport(manifest, snapshot));
    const results = Object.values(snapshot.observations);
    console.log(`${topic.id}: ${results.length} observed, ${results.filter(r => r.pendingReview).length} pending, ${results.filter(r => r.status === 'error').length} errors`);
  }
}
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) await main();
