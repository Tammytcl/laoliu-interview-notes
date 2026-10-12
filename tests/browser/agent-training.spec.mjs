import { test, expect } from '@playwright/test';

test('RL先修到Agentic和OPD项目问答的阅读链可用', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/#q=rl-actor-critic');
  for (const id of ['rl-returns-gae', 'rl-ppo-training']) {
    await page.locator('.prose a[href="#q=' + id + '"]').first().click();
    await expect(page.locator('.reader-header h1')).toBeVisible();
    await expect(page.locator('.prose')).toContainText('Q12');
  }
  for (const id of ['agentic-rl-training', 'agentic-opd-training', 'agentic-opd-pipeline', 'opd-interview-foundations', 'opd-interview-cross-tokenizer', 'opd-project-defense']) {
    await page.goto('/#q=' + id);
    await expect(page.locator('.prose')).toContainText('Q12');
    await expect(page.locator('.prose .katex-error')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), id).toBe(true);
  }
  await expect(page.locator('.prose')).toContainText('30秒');
  await expect(page.locator('.prose')).toContainText('真实runmanifest');
  const script = await page.request.get('/assets/learning/agent-training-lab.py');
  expect(script.ok()).toBe(true);
  expect(await script.text()).toContain('torch');
  const recorded = await page.request.get('/assets/learning/agent-training-lab-results.json');
  expect(recorded.ok()).toBe(true);
  expect(await recorded.json()).toHaveProperty('validation_scope');
});

test('新多教师审计原图和当前ESCD源码入口区分版本', async ({ page }) => {
  await page.goto('/#paper=paper-rethink-mopd');
  await expect(page.locator('#paper-body')).toContainText('14.8');
  await expect(page.locator('#paper-body')).toContainText('reporting benchmarks');
  await expect(page.locator('#paper-body img')).toHaveCount(7);
  await expect(page.locator('#paper-body .katex-error')).toHaveCount(0);
  await page.goto('/#paper=paper-opd-2609-34738');
  await expect(page.locator('#paper-body')).toContainText('completion_token_ids');
  await expect(page.locator('#paper-body')).toContainText('当前release');
});
