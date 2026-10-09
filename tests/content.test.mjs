import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { parseQuestion } from '../scripts/content.mjs';
const categories = [{ id: 'llm', name: '大模型基础' }];
const header = `---\nid: test-question\ntitle: "测试问题"\ncategory: llm\ndifficulty: 基础\nupdated: 2026-09-30\nsummary: "摘要"\ntags: [测试, 测试]\ndraft: false\n---\n`;
test('解析元数据、正文、目录并去重标签', () => {
  const q = parseQuestion(header + '## 回答\n\n内容\n\n## 回答\n\n后续', 'test.md', categories);
  assert.equal(q.id, 'test-question'); assert.deepEqual(q.tags, ['测试']);
  assert.equal(q.toc.length, 2); assert.equal(q.toc[1].id, 'section-2');
  assert.match(q.html, /id="section-1"/); assert.equal(q.draft, false);
});
test('CRLF、代码、表格均可解析', () => {
  const q = parseQuestion((header + '```python\nprint(1)\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |').replaceAll('\n', '\r\n'), 'test.md', categories);
  assert.match(q.html, /language-python/); assert.match(q.html, /<table>/);
});
test('不执行原始 HTML 或 javascript 链接', () => {
  const q = parseQuestion(header + '<script>alert(1)</script>\n\n[x](javascript:alert(1))', 'test.md', categories);
  assert.doesNotMatch(q.html, /<script>/); assert.doesNotMatch(q.html, /href="javascript:/);
});
test('代码块转义源码和标题，保留缩进、空行与语言标记', () => {
  const source = '    x = "<script>alert(1)</script>"\n\n    # $not_math$\n';
  const q = parseQuestion(header + '```python title="<img src=x>"\n' + source + '```', 'test.md', categories);
  assert.match(q.html, /class="code-block"/);
  assert.match(q.html, /class="language-python" data-trailing-newline="true"/);
  assert.match(q.html, /data-line="2"><\/span>/);
  assert.match(q.html, /    x = &quot;&lt;script&gt;/);
  assert.match(q.html, /&lt;img src=x&gt;/);
  assert.doesNotMatch(q.html, /<script>|<img |class="katex/);
  const untrusted = parseQuestion(header + '```python"onclick="alert(1)\nprint(1)\n```', 'test.md', categories);
  assert.match(untrusted.html, /language-text/);
  assert.doesNotMatch(untrusted.html, /onclick=/);
});
test('拒绝缺失字段、未知分类、无效日期和难度', () => {
  for (const source of [header.replace('category: llm', 'category: unknown'), header.replace('difficulty: 基础', 'difficulty: 难'), header.replace('2026-09-30', '2026-02-30'), header.replace('id: test-question', 'id: ../test'), header.replace('draft: false', 'draft: nope')]) {
    assert.throws(() => parseQuestion(source + '正文', 'test.md', categories));
  }
  assert.throws(() => parseQuestion('无元数据', 'test.md', categories));
});
test('模板是合法但不发布的草稿', async () => {
  const template = await readFile(new URL('../templates/question.md', import.meta.url), 'utf8');
  assert.equal(parseQuestion(template, 'template.md', categories).draft, true);
});
test('示例内容分类合法且 id 唯一', async () => {
  const config = JSON.parse(await readFile(new URL('../site.config.json', import.meta.url), 'utf8'));
  const root = new URL('../content/questions/', import.meta.url); const ids = new Set();
  async function visit(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) await visit(new URL(`${entry.name}/`, dir));
      else if (entry.isFile() && entry.name.endsWith('.md')) {
        const q = parseQuestion(await readFile(new URL(entry.name, dir), 'utf8'), entry.name, config.categories);
        assert.ok(!ids.has(q.id)); ids.add(q.id);
      }
    }
  }
  await visit(root);
});
