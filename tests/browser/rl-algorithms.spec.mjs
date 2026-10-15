import { test, expect } from '@playwright/test';

test('RL总览串起基础、GRPO和论文，统一练习提供可核梯度', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/#q=rl-policy-optimization-map');
  await expect(page.locator('.reader-header h1')).toContainText('RL算法总览');
  await expect(page.locator('.prose')).toContainText('Q24');
  for (const id of ['rl-actor-critic', 'rl-returns-gae', 'rl-ppo-training', 'grpo-rlvr']) {
    await page.goto('/#q=rl-policy-optimization-map');
    await page.locator(`.prose a[href="#q=${id}"]`).first().click();
    await expect(page.locator('.prose')).toContainText('Q12');
    await expect(page.locator('.prose .katex-error')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), id).toBe(true);
  }
  await page.locator('.prose a[href="#report=survey-policy-optimization"]').first().click();
  await expect(page.locator('.reader-header h1')).toContainText('七篇策略优化论文');
  await expect(page.locator('#paper-body')).toContainText('真实源码节选');
  await page.locator('#paper-body a[href="#paper=paper-grpo"]').first().click();
  await expect(page.locator('#paper-body')).toContainText('在统一训练主线中的位置');
  await page.locator('#paper-body a[href="#q=grpo-rlvr"]').first().click();
  await expect(page.locator('.reader-header h1')).toContainText('GRPO从组奖励');
  const result = await (await page.request.get('/assets/learning/rl-algorithms-unified-lab-results.json')).json();
  expect(result.two_step_ppo.fixed_targets_unchanged).toBe(true);
  expect(result.two_step_ppo.updates[0].mean_ratio_before_step).toBe(1);
  expect(result.two_step_ppo.updates[1].max_ratio_deviation_before_step).toBeGreaterThan(0);
  expect(result.two_step_ppo.final_exact_expected_reward).toBeGreaterThan(.25);
  expect(result.checks.gspo_equal_value_and_gradient).toBe(true);
  const script = await page.request.get('/assets/learning/rl-algorithms-unified-lab.py');
  expect(script.ok()).toBe(true);
  expect(await script.text()).toContain('def two_step_ppo():');
});
