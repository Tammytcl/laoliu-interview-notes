import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseQuestion } from '../scripts/content.mjs';
import { validateInterviewCollection, generatedInterviewBlocks, checkInterviewCollection, collectionPath } from '../scripts/interview-collection.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const config = JSON.parse(await readFile(join(root, 'site.config.json'), 'utf8'));
const manifest = JSON.parse(await readFile(join(root, collectionPath), 'utf8'));
const questions = [];
async function walk(dir) {
  for (const file of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, file.name);
    if (file.isDirectory()) await walk(path);
    else if (file.name.endsWith('.md')) questions.push(parseQuestion(await readFile(path, 'utf8'), relative(root, path), config.categories));
  }
}
await walk(join(root, 'content/questions'));

test('首批20个主问题可独立阅读，目录与来源集中维护且没有生成内容漂移', async () => {
  assert.ok(manifest.questions.length >= 20);
  assert.deepEqual(manifest.questions.filter(q => q.number <= 20).map(q => q.number).sort((a, b) => a - b), Array.from({ length: 20 }, (_, i) => i + 1));
  const result = await checkInterviewCollection(root, questions);
  assert.equal(result.questions.length, manifest.questions.length);
  const companies = new Set(manifest.sources.map(s => s.company).filter(Boolean));
  for (const name of ['字节', '阿里', '腾讯', '美团', '快手', '百度']) assert.ok(companies.has(name));
  const blocks = generatedInterviewBlocks(manifest, questions);
  for (const item of manifest.questions) {
    const q = questions.find(q => q.id === item.id);
    assert.ok(q.body.length > 900, item.id);
    assert.ok(q.toc.some(section => section.title === '题源与延伸'));
    assert.doesNotMatch(q.html, /katex-error/);
    assert.ok(blocks.get(manifest.topicId).includes(`#q=${item.id}`));
  }
});

test('重复概念或编号、失效题源、缺失理论链接不能静默发布', () => {
  for (const mutate of [
    m => { m.questions[1].concept = m.questions[0].concept; },
    m => { m.questions[1].number = m.questions[0].number; },
    m => { m.questions[0].sourceIds = ['missing-source']; },
    m => { m.questions[0].theoryIds = ['missing-theory']; },
  ]) {
    const edited = structuredClone(manifest); mutate(edited);
    assert.throws(() => validateInterviewCollection(edited, questions));
  }
});

test('无公司面经依据的项目自拟题不会生成公司认证标签', () => {
  const blocks = generatedInterviewBlocks(manifest, questions);
  const project = manifest.questions.find(q => q.id === 'pt-escd-events');
  assert.equal(project.origin, '项目自拟');
  assert.match(blocks.get(project.id), /题目归类：项目自拟/);
  assert.doesNotMatch(blocks.get(project.id), /公司认证|字节真题|腾讯真题/);
  assert.match(blocks.get(project.id), /ESCD 公开实现与评测说明/);
});

test('回答标题变化会在目录中体现，而不是另外维护第二份标题', () => {
  const edited = structuredClone(questions);
  const first = edited.find(q => q.id === manifest.questions[0].id);
  first.title = 'Q01 · 新的选型表述';
  assert.match(generatedInterviewBlocks(manifest, edited).get(manifest.topicId), /新的选型表述/);
});
