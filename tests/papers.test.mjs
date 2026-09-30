import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, cp, symlink, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { parsePaperContent, validatePaperLinks } from '../scripts/papers-content.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const run = promisify(execFile);
const paper = { id: 'paper-test', title: '示例', summary: '摘要', updated: '2026-09-30', tags: ['测试'], draft: false,
  paper_title: 'Original Title', authors: ['Author'], year: 2023, direction: 'llm', paper_url: 'https://arxiv.org/abs/2305.13245', evidence: '资料整理', note_ids: [] };
const report = { id: 'daily-test', title: '日报', summary: '摘要', updated: '2026-09-30', tags: [], draft: false,
  type: 'daily', date: '2026-09-30', directions: ['llm'], paper_ids: ['paper-test'] };
const source = (meta, body = '## 核心问题\n\n正文') => `---\n${JSON.stringify(meta)}\n---\n${body}`;
const parse = (meta, type, body) => parsePaperContent(source(meta, body), 'test.md', type);

test('论文保留原文与方向元数据，正文安全渲染', () => {
  const item = parse(paper, 'papers', '## 方法\n\n<script>alert(1)</script>\n\n[x](javascript:alert(1))');
  assert.equal(item.paperTitle, 'Original Title'); assert.equal(item.direction, 'llm');
  assert.equal(item.evidence, '资料整理'); assert.equal(item.toc.length, 1);
  assert.doesNotMatch(item.html, /<script>|href="javascript:/);
});
test('拒绝错误的原文地址、方向、日期、证据与版本', () => {
  for (const patch of [{ paper_url: 'javascript:alert(1)' }, { paper_url: 'https://user:password@example.com' },
    { direction: 'systems' }, { authors: [] }, { year: '2023' }, { updated: '2026-02-30' }, { evidence: '已读懂' }, { template_version: 2 }]) {
    assert.throws(() => parse({ ...paper, ...patch }, 'papers'));
  }
  for (const patch of [{ type: 'weekly' }, { date: '2026-02-30' }, { directions: [] }, { paper_ids: 'paper-test' }]) {
    assert.throws(() => parse({ ...report, ...patch }, 'reports'));
  }
  assert.deepEqual(parse({ ...report, paper_ids: [] }, 'reports').paperIds, []);
});
test('三个收录模板都是可校验但不发布的草稿', async () => {
  for (const [name, collection] of [['paper.md', 'papers'], ['daily-report.md', 'reports'], ['survey-report.md', 'reports']]) {
    assert.equal(parsePaperContent(await readFile(join(root, 'templates', name), 'utf8'), name, collection).draft, true);
  }
});
test('报告必须引用已发布论文，跨集合重复 id 和正文坏链接被拒绝', () => {
  const p = parse(paper, 'papers'); const r = parse(report, 'reports');
  validatePaperLinks([], [p], [r]);
  assert.throws(() => validatePaperLinks([], [{ ...p, draft: true }], [r]), /未发布或不存在/);
  assert.throws(() => validatePaperLinks([{ id: p.id }], [p], []), /重复/);
  assert.throws(() => validatePaperLinks([], [parse(paper, 'papers', '[错链](#report=missing)')], []), /未发布或不存在/);
  assert.throws(() => validatePaperLinks([], [{ ...p, noteIds: ['missing'] }], []), /未发布或不存在/);
  assert.throws(() => validatePaperLinks([], [p, { ...p, id: 'another-paper', paperUrl: p.paperUrl + 'v3' }], []), /同一论文/);
  assert.throws(() => validatePaperLinks([], [p], [r, { ...r, id: 'another-daily' }]), /当天 Daily/);
});
test('新建命令按方向生成草稿，重复 Daily 继续更新且不覆盖正文', async t => {
  const fixture = await mkdtemp(join(tmpdir(), 'laoliu-paper-new-'));
  t.after(() => rm(fixture, { recursive: true, force: true }));
  await cp(join(root, 'scripts'), join(fixture, 'scripts'), { recursive: true });
  await cp(join(root, 'templates'), join(fixture, 'templates'), { recursive: true });
  await symlink(join(root, 'node_modules'), join(fixture, 'node_modules'), 'dir');
  const create = (...args) => run(process.execPath, ['scripts/new-paper.mjs', ...args], { cwd: fixture });
  await create('paper', 'paper-example', 'infra');
  const file = join(fixture, 'content/papers/infra/paper-example.md');
  const before = await readFile(file, 'utf8');
  assert.equal(parsePaperContent(before, file, 'papers').direction, 'infra');
  await assert.rejects(create('paper', 'paper-example', 'infra'));
  assert.equal(await readFile(file, 'utf8'), before);
  await create('daily', '2026-09-30');
  const dailyFile = join(fixture, 'content/reports/daily/daily-2026-09-30.md');
  await writeFile(dailyFile, '本人已填写的阅读更新');
  const repeated = await create('daily', '2026-09-30');
  assert.match(repeated.stdout, /当天报告已存在/);
  assert.equal(await readFile(dailyFile, 'utf8'), '本人已填写的阅读更新');
  await assert.rejects(create('daily', '2026-02-30'));
  await create('survey', 'survey-example', 'diffusion');
  const survey = parsePaperContent(await readFile(join(fixture, 'content/reports/surveys/survey-example.md'), 'utf8'), 'survey', 'reports');
  assert.deepEqual(survey.directions, ['diffusion']); assert.equal(survey.draft, true);
});
