import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
test('外部清单逐编号合并且不转载原题，私人工作树按文件hash区分基线', async () => {
  const root = new URL('../', import.meta.url);
  const mapping = JSON.parse(await readFile(new URL('assets/learning/post-training-checklist-map/mapping.json', root), 'utf8'));
  assert.equal(mapping.actualItems, 222);
  assert.deepEqual(mapping.items.map(x => x.sourceNumber), Array.from({length:222}, (_, i) => i + 1));
  for (const item of mapping.items) {
    assert.match(item.target, /^#(?:q|paper)=[a-z0-9-]+$/);
    assert.ok(item.studyPrompt.length > 0 && item.studyPrompt.length < 70);
    assert.equal(item.sourceQuestion, undefined);
  }
  const evidence = JSON.parse(await readFile(new URL('assets/learning/agentic-turn-framework/code-evidence.json', root), 'utf8'));
  assert.equal(evidence.workingTreeDirty, true);
  assert.equal(evidence.files.length, 42);
  for (const file of evidence.files) {
    assert.match(file.sha256, /^[a-f0-9]{64}$/);
    assert.ok(file.lines > 0 && !file.path.startsWith('/') && !file.path.split('/').includes('..'));
  }
});
