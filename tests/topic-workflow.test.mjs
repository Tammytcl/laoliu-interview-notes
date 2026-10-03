import test from 'node:test';
import assert from 'node:assert/strict';
import { observeSources, sourceReviewReport } from '../scripts/topic-sources.mjs';
const sha = 'a'.repeat(40), newer = 'b'.repeat(40);
const manifest = { topicId: 'example', sources: [{ id: 'repo', title: 'Repository', url: 'https://github.com/example/repo', kind: 'official-code', repository: 'example/repo', revision: sha, reviewedAt: '2026-10-03' }] };
test('观察上游变化只生成候选，不修改已核对的来源版本', async () => {
  const before = JSON.stringify(manifest);
  const snapshot = await observeSources(manifest, {}, async () => new Response(JSON.stringify({ sha: newer })), '2026-10-04T00:00:00Z');
  assert.equal(snapshot.observations.repo.pendingReview, true);
  assert.equal(JSON.stringify(manifest), before);
  assert.match(sourceReviewReport(manifest, snapshot), new RegExp(`compare/${sha}\\.\\.\\.${newer}`));
});
test('失败保留上次成功值、待核状态，并隐藏异常中的敏感细节', async () => {
  const previous = { observations: { repo: { status: 'ok', revision: newer, pendingReview: true, lastSuccessAt: '2026-10-04T00:00:00Z' } } };
  const snapshot = await observeSources(manifest, previous, async () => { throw new Error('request failed: secret-token'); }, '2026-10-05T00:00:00Z');
  assert.equal(snapshot.observations.repo.revision, newer);
  assert.equal(snapshot.observations.repo.pendingReview, true);
  assert.equal(snapshot.observations.repo.lastSuccessAt, previous.observations.repo.lastSuccessAt);
  assert.equal(snapshot.observations.repo.status, 'error');
  assert.doesNotMatch(JSON.stringify(snapshot), /secret-token/);
});
test('同一个仓库的多个来源只查询一次 revision，文档变化按 hash 判断', async () => {
  const sources = { topicId: 'example', sources: [manifest.sources[0], { ...manifest.sources[0], id: 'docs', watchUrl: 'https://raw.githubusercontent.com/example/repo/main/docs.md', reviewedSha256: '0'.repeat(64) }] };
  let calls = 0;
  const snapshot = await observeSources(sources, {}, async url => {
    if (url.includes('api.github.com')) { calls++; return new Response(JSON.stringify({ sha })); }
    return new Response('new document');
  });
  assert.equal(calls, 1);
  assert.equal(snapshot.observations.repo.pendingReview, false);
  assert.equal(snapshot.observations.docs.pendingReview, true);
});

test('使用同一个已观察 revision 获取 GitHub 文档，并核对内容 hash', async () => {
  const { createHash } = await import('node:crypto');
  const text = 'reviewed documentation';
  const source = { ...manifest.sources[0], id: 'doc', watchUrl: 'https://raw.githubusercontent.com/example/repo/main/docs.md', watchPath: 'docs.md', reviewedSha256: createHash('sha256').update(text).digest('hex') };
  let fileUrl;
  const snapshot = await observeSources({ ...manifest, sources: [source] }, {}, async url => {
    if (url.includes('/commits/')) return new Response(JSON.stringify({ sha }));
    fileUrl = url;
    return new Response(JSON.stringify({ encoding: 'base64', content: Buffer.from(text).toString('base64') }));
  });
  assert.ok(fileUrl.endsWith(`contents/docs.md?ref=${sha}`));
  assert.equal(snapshot.observations.doc.pendingReview, false);
});
