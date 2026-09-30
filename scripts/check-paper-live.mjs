import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const base = new URL(process.argv[2] || 'http://127.0.0.1:4273/');
if (!['http:', 'https:'].includes(base.protocol)) throw new Error('Expected HTTP(S) site');
const browser = await chromium.launch({headless:true});
const failures = [], results = [];
const version = Date.now().toString();
const artifacts = 'test-results/paper-live';
await mkdir(artifacts,{recursive:true});
try {
  const page = await browser.newPage();
  page.on('pageerror',e=>failures.push(e.message));
  const address = (hash) => {const url=new URL(base);url.searchParams.set('check',version);url.hash=hash;return url.href;};
  await page.goto(address('view=papers'),{waitUntil:'domcontentloaded'});
  await page.locator('.paper-card').first().waitFor();
  // Follow the deployed content fingerprint rather than a potentially cached data.json alias.
  const appUrl = await page.locator('script[type="module"]').getAttribute('src');
  const script = await (await page.request.get(new URL(appUrl,base).href)).text();
  const dataPath = script.match(/fetch\('\.\/(data-[a-f0-9]+\.json)'\)/)?.[1];
  assert.ok(dataPath,'Fingerprint for deployed content missing');
  const data = await (await page.request.get(new URL(dataPath,base).href)).json();
  assert.equal(await page.locator('.paper-card').count(),data.papers.length);
  assert.equal(await page.locator('[data-paper-area]').count(),3);
  assert.deepEqual(data.papers.map(p=>p.published),[...data.papers.map(p=>p.published)].sort().reverse());
  for (const viewport of [{width:1440,height:1000},{width:390,height:844}]) {
    await page.setViewportSize(viewport);
    await page.goto(address('view=papers'));
    await page.locator('.paper-card').first().waitFor();
    await page.locator('.paper-method-preview img').evaluateAll(images=>images.forEach(i=>i.loading='eager'));
    await page.waitForFunction(()=>[...document.querySelectorAll('.paper-method-preview img')].every(i=>i.complete&&i.naturalWidth>0),{},{timeout:60000});
    await page.screenshot({path:`${artifacts}/library-${viewport.width}.png`,fullPage:true});
    for (const paper of data.papers) {
      await page.goto(address(`paper=${paper.id}`));
      await page.locator('#paper-body').waitFor();
      await page.locator('#paper-body img').evaluateAll(images=>images.forEach(i=>i.loading='eager'));
      await page.waitForFunction(()=>[...document.querySelectorAll('#paper-body img')].every(i=>i.complete&&i.naturalWidth>0),{},{timeout:60000});
      const images = await page.locator('#paper-body img').evaluateAll(images=>images.map(i=>{
        const canvas=document.createElement('canvas');canvas.width=120;canvas.height=80;
        const context=canvas.getContext('2d');context.drawImage(i,0,0,120,80);
        const rgba=context.getImageData(0,0,120,80).data;let ink=0;
        for(let j=0;j<rgba.length;j+=4)if(Math.min(rgba[j],rgba[j+1],rgba[j+2])<220)ink++;
        const rect=i.getBoundingClientRect();
        return {src:i.getAttribute('src'),width:i.naturalWidth,height:i.naturalHeight,ink,ratioError:Math.abs(rect.width/rect.height-i.naturalWidth/i.naturalHeight)};
      }));
      const deep = paper.depthStandard === 'ddpm' || paper.templateVersion >= 5;
      if (deep) assert.ok(images.length >= 3,`${paper.id}: DDPM-depth report lacks evidence figures`);
      for (const image of images) {
        if (deep) assert.ok(image.src.endsWith('-pdf.png'),`${paper.id}: DDPM-depth report uses an unverified capture`);
        assert.ok(image.ink>35,`${paper.id}: image appears blank`);
        assert.ok(image.ratioError<0.02,`${paper.id}: distorted aspect ratio`);
      }
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`${paper.id}: horizontal overflow`);
      assert.equal(await page.locator('.publication-toc button').count(),5);
      if (images.length) {
        await page.locator('[data-figure]').first().click();
        await page.keyboard.press('Escape');
        assert.equal(await page.locator('#paper-figure-dialog').evaluate(d=>d.open),false);
        assert.equal(await page.locator('#paper-body').count(),1,'Escape navigated away from report');
      }
      await page.screenshot({path:`${artifacts}/${paper.id}-${viewport.width}.png`,fullPage:true});
      results.push({paperId:paper.id,viewport:viewport.width,images:images.length});
    }
  }
  assert.deepEqual(failures,[]);
  await writeFile(`${artifacts}/result.json`,JSON.stringify({site:base.href,checkedAt:new Date().toISOString(),results,errors:failures},null,2));
  console.log(JSON.stringify({papers:data.papers.length,viewports:2,figureChecks:results.reduce((n,r)=>n+r.images,0),errors:failures}));
} finally {await browser.close();}
