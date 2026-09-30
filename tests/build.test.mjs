import test from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';

const run = promisify(execFile);
const root = fileURLToPath(new URL('../', import.meta.url));

test('内容更新使用新资源地址，同一内容重复构建保持地址稳定', async t => {
  const fixture = await mkdtemp(join(tmpdir(), 'laoliu-build-'));
  t.after(() => rm(fixture, { recursive: true, force: true }));
  await cp(join(root, 'scripts'), join(fixture, 'scripts'), { recursive: true });
  await cp(join(root, 'web'), join(fixture, 'web'), { recursive: true });
  await symlink(join(root, 'node_modules'), join(fixture, 'node_modules'), 'dir');
  await mkdir(join(fixture, 'content/questions'), { recursive: true });
  await writeFile(join(fixture, 'site.config.json'), JSON.stringify({
    title: '测试站点', description: '缓存更新测试',
    categories: [{ id: 'llm', name: 'LLM' }]
  }));
  const question = join(fixture, 'content/questions/example.md');
  const header = '---\nid: example\ntitle: "测试笔记"\ncategory: llm\ndifficulty: 基础\nupdated: 2026-09-30\nsummary: "测试摘要"\ntags: []\ndraft: false\n---\n';
  await writeFile(question, header + '第一版内容');
  async function build() {
    await run(process.execPath, ['scripts/build.mjs'], { cwd: fixture });
    const html = await readFile(join(fixture, 'dist/index.html'), 'utf8');
    const appFile = html.match(/src="\.\/(app-[a-f0-9]+\.js)"/)[1];
    const styleFile = html.match(/href="\.\/(style-[a-f0-9]+\.css)"/)[1];
    const app = await readFile(join(fixture, 'dist', appFile), 'utf8');
    const dataFile = app.match(/fetch\('\.\/(data-[a-f0-9]+\.json)'\)/)[1];
    const data = await readFile(join(fixture, 'dist', dataFile), 'utf8');
    assert.equal(data, await readFile(join(fixture, 'dist/data.json'), 'utf8'));
    assert.doesNotMatch(app, /__DATA_FILE__/);
    return { appFile, styleFile, dataFile, body: JSON.parse(data).questions[0].body };
  }
  const first = await build();
  assert.deepEqual(await build(), first);
  await writeFile(question, header + '第二版内容');
  const next = await build();
  assert.notEqual(next.dataFile, first.dataFile);
  assert.notEqual(next.appFile, first.appFile);
  assert.equal(next.styleFile, first.styleFile);
  assert.equal(next.body, '第二版内容');
});
