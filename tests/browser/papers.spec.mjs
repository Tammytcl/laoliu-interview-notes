import { test, expect } from '@playwright/test';

test('课堂代码块在手机上保留源码、空行、复制与独立滚动', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/#paper=paper-dpo');
  await expect(page.locator('#paper-space h1')).toContainText('Direct Preference Optimization');
  const block = page.locator('.code-block').filter({ has: page.locator('.code-title', { hasText: 'DPO 作者实现 · 取回答logprob' }) });
  await expect(block).toHaveCount(1);
  await expect(block.locator('.code-language')).toHaveText('Python');
  await expect(block.locator('.code-line')).toHaveCount(13);
  expect(await block.locator('.code-line').nth(3).textContent()).toBe('');
  expect(await block.locator('pre').evaluate(el => el.scrollWidth > el.clientWidth)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  const data = await (await page.request.get('/data.json')).json();
  const body = data.papers.find(p => p.id === 'paper-dpo').body;
  const source = body.match(/```python title="DPO 作者实现 · 取回答logprob"\n([\s\S]*?)\n```/)[1] + '\n';
  await block.locator('[data-copy-code]').click();
  await expect(page.locator('#toast')).toHaveText('代码已复制');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(source);
  await page.evaluate(() => Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async () => { throw new Error('unavailable'); } } }));
  await block.locator('[data-copy-code]').click();
  await expect(page.locator('#toast')).toContainText('复制未完成');
  expect(await page.evaluate(() => window.getSelection().toString().length)).toBeGreaterThan(0);
  for (const title of ['问题背景', '前置知识', '已有工作与本文位置']) {
    await expect(page.locator('.publication-body h3', { hasText: title })).toHaveCount(1);
  }
});

test('论文库、两行任务筛选、Daily 与专题报告相互链接', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/'); await page.locator('[data-paper-nav]').click();
  const data = await (await page.request.get('/data.json')).json();
  await expect(page.locator('.paper-card')).toHaveCount(data.papers.length);
  expect(data.papers.map(p => p.published)).toEqual([...data.papers.map(p => p.published)].sort().reverse());
  await expect(page.locator('.paper-citations')).toHaveCount(data.papers.length);
  await expect(page.locator('.paper-quality, [data-paper-quality]')).toHaveCount(0);
  await expect(page.locator('#paper-space')).not.toContainText('DDPM 级精读');
  await expect(page.locator('.paper-method-preview img')).toHaveCount(data.papers.filter(p => p.methodFigure).length);
  for (const preview of await page.locator('.paper-method-preview img').all()) {
    await preview.scrollIntoViewIfNeeded();
    await expect.poll(() => preview.evaluate(img => img.complete && img.naturalWidth > 0)).toBe(true);
  }
  const gqa = page.locator('.paper-card').filter({ has: page.locator('h3 a[href="#paper=paper-gqa"]') });
  await expect(gqa.locator('.paper-category')).toHaveCount(1);
  await expect(gqa.locator('.paper-task')).toHaveText('文本生成');
  await expect(gqa).toHaveAttribute('data-category', 'language');
  await expect(page.locator('[data-paper-nav]')).toHaveAttribute('aria-current', 'page');
  await page.locator('[data-paper-area="vision"]').click();
  await expect(page.locator('[data-paper-task="image-generation"]')).toBeVisible();
  await expect(page.locator('[data-paper-task="agents"]')).toHaveCount(0);
  await page.locator('[data-paper-task="image-generation"]').click();
  await expect(page.locator('.paper-card')).toHaveCount(1);
  await page.reload();
  await expect(page.locator('[data-paper-task="image-generation"]')).toHaveAttribute('aria-pressed', 'true');
  await page.locator('[data-paper-area="language"]').click();
  await expect(page.locator('.paper-card')).toHaveCount(data.papers.filter(p => p.areas.includes('language')).length);
  await expect(page.locator('[data-paper-task=""]')).toHaveAttribute('aria-pressed', 'true');
  await page.locator('#paper-search').fill('no-match-paper-123');
  await expect(page.locator('[data-paper-reset]')).toBeVisible();
  await page.locator('[data-paper-reset]').click();
  await expect(page.locator('.paper-card')).toHaveCount(data.papers.length);
  await page.getByRole('link', { name: 'Daily 报告', exact: true }).click();
  await expect(page.locator('.paper-card')).toHaveCount(data.reports.filter(r => r.type === 'daily').length);
  await page.locator('.paper-card h3 a[href="#report=daily-2026-09-30"]').click();
  await expect(page.locator('.paper-related a[href^="#paper="]')).toHaveCount(data.reports.find(r => r.id === 'daily-2026-09-30').paperIds.length);
  await page.locator('.paper-related a[href="#paper=paper-gqa"]').click();
  await expect(page.locator('#paper-space h1')).toContainText('GQA');
  await expect(page.locator('.paper-related a[href="#report=survey-inference-memory"]')).toBeVisible();
  await page.locator('.paper-related a[href="#report=survey-inference-memory"]').click();
  await page.reload(); await expect(page.locator('#paper-space h1')).toContainText('专题调研');
  await page.locator('.back-link').click();
  await expect(page.locator('.paper-tabs [aria-current="page"]')).toHaveText('专题调研');
  expect(errors).toEqual([]);
});

test('正式报告显示英文元数据、五模块目录、数学公式与可放大原图', async ({ page }) => {
  await page.goto('/#paper=paper-gqa');
  await expect(page.locator('#paper-space h1')).toHaveText('GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints');
  await expect(page.locator('.publication-authors')).toContainText('Joshua Ainslie');
  await expect(page.locator('.publication-affiliations')).toContainText('University of Southern California');
  await expect(page.locator('.publication-actions a', { hasText: 'arXiv' })).toHaveAttribute('href', 'https://arxiv.org/abs/2305.13245v3');
  await expect(page.locator('.publication-actions a', { hasText: 'GitHub' })).toHaveAttribute('href', 'https://github.com/google/flaxformer');
  await expect(page.locator('.reader-aside textarea')).toHaveCount(0);
  await expect(page.locator('.publication-toc button')).toHaveCount(5);
  await expect(page.locator('.publication-body .katex').first()).toBeVisible();
  await expect(page.locator('.paper-figure-button')).toHaveCount(7);
  await page.locator('.paper-figure-button').first().click();
  await expect(page.locator('#paper-figure-dialog')).toBeVisible();
  await expect(page.locator('#figure-caption')).toContainText('Figure 2');
  await expect.poll(() => page.locator('#figure-full').evaluate(img => img.complete && img.naturalWidth > 0)).toBe(true);
  await page.locator('[data-close-figure]').click();
  await expect(page.locator('#paper-figure-dialog')).not.toBeVisible();
  await page.reload();
  await expect(page.locator('[data-paper-star], [data-paper-stage], #paper-stage, #paper-starred, [data-reading-export], [data-reading-import]')).toHaveCount(0);
  await page.goto('/#view=papers');
  const data = await (await page.request.get('/data.json')).json();
  await expect(page.locator('.paper-card')).toHaveCount(data.papers.length);
  await expect(page.locator('[data-paper-star], #paper-stage, #paper-starred, .paper-backup')).toHaveCount(0);
});

test('OPD 专题精读展示方法图、实验结果和五模块目录', async ({ page }) => {
  const data = await (await page.request.get('/data.json')).json();
  expect(data.papers.filter(p => p.id.startsWith('paper-opd-')).every(p => p.toc.filter(item => item.level === 2).length === 5)).toBe(true);
  await page.goto('/#paper=paper-opd-2609-30837');
  await expect(page.locator('#paper-space h1')).toHaveText('MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation');
  await expect(page.locator('.publication-toc button')).toHaveCount(5);
  await expect(page.locator('.publication-body')).toContainText('ExpertAlign');
  await expect(page.locator('.publication-body')).toContainText('38.58');
  await expect(page.locator('.publication-header .paper-quality')).toHaveCount(0);
  await expect(page.locator('.paper-quality-notice')).toHaveCount(0);
  await expect(page.locator('.paper-figure-button')).toHaveCount(5);
  await page.locator('.paper-figure-button').first().click();
  await expect(page.locator('#figure-caption')).toContainText('Figure 1');
  await expect.poll(() => page.locator('#figure-full').evaluate(img => img.complete && img.naturalWidth > 0)).toBe(true);
});

test('补齐的 GKD、MiniLLM 与 SimCT 报告加载原文算法、表格和结果图', async ({ page }) => {
  for (const id of ['paper-opd-2306-13649', 'paper-opd-2306-08543', 'paper-opd-2605-07711']) {
    await page.goto(`/#paper=${id}`);
    await expect(page.locator('.publication-toc button')).toHaveCount(5);
    await expect(page.locator('.paper-figure-button')).toHaveCount(5);
    for (const figure of await page.locator('.publication-body img').all()) {
      await figure.scrollIntoViewIfNeeded();
      await expect.poll(() => figure.evaluate(img => img.complete && img.naturalWidth > 0)).toBe(true);
    }
  }
  await page.goto('/#view=papers');
  for (const id of ['paper-opd-2306-13649', 'paper-opd-2306-08543', 'paper-opd-2605-07711']) {
    const card = page.locator('.paper-card').filter({ has: page.locator(`h3 a[href="#paper=${id}"]`) });
    await expect(card.locator('.paper-method-preview img')).toHaveCount(1);
  }
});

test('工作台缓存按钮确认后请求清理，普通预览没有写接口', async ({ page }) => {
  const request = await page.request.get('/api/paper-cache'); expect(request.status()).toBe(404);
  let cleaned = false;
  await page.route('**/api/paper-cache', route => route.fulfill({ json: { files: cleaned ? 0 : 3, bytes: 4096, token: 'session-token' } }));
  await page.route('**/api/paper-cache/clean', async route => {
    expect(route.request().method()).toBe('POST'); expect(route.request().headers()['x-paper-cache-token']).toBe('session-token');
    cleaned = true; await route.fulfill({ json: { deletedFiles: 3, freedBytes: 4096, preserved: ['content/papers', 'assets/papers'] } });
  });
  await page.goto('/#view=papers&tab=guide');
  await expect(page.locator('#paper-cache-status')).toContainText('3 个缓存文件');
  page.once('dialog', dialog => dialog.dismiss()); await page.locator('[data-cache-clean]').click(); expect(cleaned).toBe(false);
  page.once('dialog', dialog => dialog.accept()); await page.locator('[data-cache-clean]').click();
  await expect(page.locator('#paper-cache-status')).toContainText('0 个缓存文件'); await expect(page.locator('[data-cache-clean]')).toBeDisabled();
});

test('三份模板可下载，手机与仓库子路径的论文路由可直接刷新', async ({ page }) => {
  await page.goto('/#view=papers&tab=guide');
  await expect(page.locator('.paper-template')).toHaveCount(3);
  for (const file of ['paper.md', 'daily-report.md', 'survey-report.md']) {
    const response = await page.request.get(`/templates/${file}`);
    expect(response.ok()).toBe(true); expect(await response.text()).toContain('draft: true');
  }
  await page.setViewportSize({ width: 390, height: 844 });
  for (const route of ['view=papers', 'view=papers&tab=daily', 'view=papers&tab=survey', 'view=papers&tab=guide', 'paper=paper-gqa', 'report=survey-inference-memory']) {
    await page.goto(`http://127.0.0.1:4176/interview-notes/#${route}`);
    await page.reload(); await expect(page.locator('#paper-space h1')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await page.screenshot({ path: 'test-results/paper-mobile.png', fullPage: true });
});

test('DSec系统精读保留完整作者、配置账本、原图表与源码边界', async ({ page }) => {
  const data = await (await page.request.get('/data.json')).json();
  const paper = data.papers.find(p => p.id === 'paper-dsec');
  expect(paper.authors).toHaveLength(131);
  expect(paper.affiliations).toEqual(['DeepSeek-AI', 'Tsinghua University']);
  expect(paper.paperUrl).toBe('https://arxiv.org/abs/2609.22978v1');
  await page.goto('/#paper=paper-dsec');
  await expect(page.locator('#paper-body')).toContainText('AMD EPYC9655');
  await expect(page.locator('#paper-body')).toContainText('时间积分');
  await expect(page.locator('#paper-body')).toContainText('未找到公开的整个DSec');
  await expect(page.locator('.publication-toc button')).toHaveCount(5);
  await expect(page.locator('#paper-body img')).toHaveCount(10);
  await expect(page.locator('#paper-body .katex-error')).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.reload();
  await expect(page.locator('#paper-body')).toContainText('Figure 13');
});
