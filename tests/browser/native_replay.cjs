// Run against an exported replay directory served over HTTP; no simulation mocks.
// NODE_PATH may point to an existing Playwright installation.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const [base, output, mode = 'candidate'] = process.argv.slice(2);
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [], failedResponses = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('response', r => { if (r.status() >= 400) failedResponses.push([r.url(), r.status()]); });
  const catalog = await (await page.request.get(new URL('catalog.json', base).href)).json();
  assert(catalog.length >= 1);
  const cases = [];
  const ready = async () => {
    await page.locator('#callDetails .call-name').waitFor();
    assert.equal(await page.locator('#error').isVisible(), false, await page.locator('#error').textContent());
  };
  const load = async id => { await page.goto(`${base}?case=${id}`, { waitUntil: 'networkidle' }); await ready(); };
  const time = async () => Number(await page.locator('#time').inputValue());
  for (const item of catalog) {
    const data = await (await page.request.get(new URL(item.file, base).href)).json();
    assert(data.calls.length > 0);
    await load(item.id);
    assert.equal(await page.locator('#caseSelect').inputValue(), item.id);
    // Fixed state captures compare layout, data rendering and responsive behavior.
    await page.locator('#timeInput').fill('0'); await page.locator('#timeInput').dispatchEvent('change');
    await page.mouse.move(0, 0);
    await page.screenshot({ path: path.join(output, `${item.id}-desktop.png`), fullPage: true });
    await page.setViewportSize({ width: 768, height: 1000 });
    await page.screenshot({ path: path.join(output, `${item.id}-mobile.png`), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('#reload').click(); await page.waitForTimeout(100); await ready();
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    await page.locator('#speed').selectOption('0.02');
    await page.locator('#timeInput').fill('0'); await page.locator('#timeInput').dispatchEvent('change');
    await page.locator('#play').click(); await page.waitForFunction(() => Number(document.querySelector('#time').value) > 0); await page.locator('#play').click();
    const paused = await time(); assert(paused > 0);
    await page.waitForTimeout(100); assert.equal(await time(), paused);
    await page.locator('#speed').selectOption('5');
    await page.locator('#timeInput').fill('1'); await page.locator('#timeInput').dispatchEvent('change');
    assert.equal(await time(), 1e6);
    await page.locator('#time').evaluate(el => { el.value = '0'; el.dispatchEvent(new Event('input', { bubbles: true })); });
    assert.equal(await time(), 0);
    const next = data.events.find(e => e.time > 1 && e.time <= Math.min(data.horizon, Math.max(1e6, data.end * 1.06)));
    await page.locator('#next').click(); if (next) assert.equal(await time(), next.time);
    await page.locator('#previous').click(); assert.equal(await time(), 0);
    const moment = page.locator('#moments [data-time]').first();
    if (await moment.count()) { const target = Number(await moment.getAttribute('data-time')); await moment.click(); assert.equal(await time(), target); }
    const before = await page.locator('#windowInfo').textContent();
    await page.locator('#focusTime').click(); assert.notEqual(await page.locator('#windowInfo').textContent(), before);
    await page.locator('#allTime').click();
    const node = page.locator('#topology [data-device]').first();
    await node.click(); assert((await page.locator('#deviceDetail').textContent()).trim().length > 0);
    const link = page.locator('#topology [data-link]').first();
    if (await link.count()) { await link.focus(); await link.press('Enter'); assert((await page.locator('#deviceDetail').textContent()).trim().length > 0); }
    await page.locator('#deviceFilter').selectOption('');
    const cellOptions = await page.locator('#cellFilter option').evaluateAll(xs => xs.map(x => x.value));
    if (cellOptions.length > 1) { await page.locator('#cellFilter').selectOption(cellOptions[1]); await page.locator('#cellFilter').selectOption(''); }
    const kindOptions = await page.locator('#kindFilter option').evaluateAll(xs => xs.map(x => x.value));
    if (kindOptions.length > 1) { await page.locator('#kindFilter').selectOption(kindOptions[1]); await page.locator('#kindFilter').selectOption(''); }
    // Exercise the canvas itself: hover/click a recorded long compute bar, then wheel zoom.
    const compute = data.calls.map((c, i) => ({ c, i })).filter(({c}) => !c.comm)
      .sort((a,b) => (b.c.end-b.c.start)-(a.c.end-a.c.start))[0];
    await page.locator('#cellFilter').selectOption(compute.c.cell);
    await page.locator('#deviceFilter').selectOption(compute.c.device);
    await page.locator('#kindFilter').selectOption('compute');
    await page.locator('#search').fill(compute.c.id);
    await page.locator('#callSelect').selectOption(String(compute.i));
    const canvas = page.locator('#timeline');
    await canvas.scrollIntoViewIfNeeded();
    const box = await canvas.boundingBox();
    const span = Math.max(compute.c.end-compute.c.start, 50000);
    const limit = Math.min(data.horizon, Math.max(1e6, data.end*1.06));
    const left = Math.max(0, compute.c.start-span*.15), right = Math.min(limit, compute.c.end+span*.15);
    const x = box.x+124+((compute.c.start+compute.c.end)/2-left)/(right-left)*(box.width-139);
    const y = box.y+47;
    await page.mouse.move(x,y);
    assert((await page.locator('#tooltip').textContent()).includes(compute.c.id));
    assert(await page.locator('#tooltip').isVisible());
    await page.mouse.click(x,y);
    assert.equal(await page.locator('#callSelect').inputValue(), String(compute.i));
    const windowBeforeWheel = await page.locator('#windowInfo').textContent();
    await page.mouse.wheel(0,-120);
    await page.waitForTimeout(150);
    assert.notEqual(await page.locator('#windowInfo').textContent(), windowBeforeWheel);
    await page.locator('#cellFilter').selectOption('');
    await page.locator('#deviceFilter').selectOption('');
    await page.locator('#kindFilter').selectOption('');
    const ci = data.calls.findIndex(c => c.deps.length > 0);
    assert(ci >= 0);
    await page.locator('#search').fill(data.calls[ci].id);
    await page.locator('#callSelect').selectOption(String(ci));
    assert((await page.locator('#callDetails .call-name').textContent()).includes(data.calls[ci].id));
    const dependency = page.locator('#dependencies [data-call]').first();
    assert(await dependency.count()); await dependency.click();
    await page.locator('#search').fill('');
    const step = page.locator('#steps [data-step]').first();
    assert(await step.count()); await step.click();
    assert((await page.locator('#stepDetail').textContent()).trim().length > 0);
    const migration = page.locator('#migrations [data-flow]').first();
    if (await migration.count()) await migration.click();
    assert(await page.locator('#checks .check').count());
    await page.locator('details.source summary').click();
    assert((await page.locator('#sourceDetails').textContent()).trim().length > 0);
    await ready();
    cases.push({ id: item.id, calls: data.calls.length, migrationLinks: await migration.count(), checks: await page.locator('#checks .check').count() });
  }
  // Regression: retain a large call index, then switch to a shorter scene.
  let switching = 'single-case';
  if (catalog.length > 1) {
    const big = [...catalog].sort((a,b) => b.calls-a.calls)[0];
    const small = [...catalog].sort((a,b) => a.calls-b.calls)[0];
    await load(big.id);
    const data = await (await page.request.get(new URL(big.file, base).href)).json();
    const last = data.calls.length - 1;
    await page.locator('#search').fill(data.calls[last].id);
    await page.locator('#callSelect').selectOption(String(last));
    await page.locator('#caseSelect').selectOption(small.id);
    await page.waitForTimeout(300);
    const visible = await page.locator('#error').isVisible();
    if (mode === 'baseline') { assert(visible); switching = 'known baseline index error reproduced'; }
    else { assert.equal(visible, false, await page.locator('#error').textContent()); await ready(); switching = 'passed'; }
  }
  assert.deepEqual(errors, []); assert.deepEqual(failedResponses, []);
  const result = { cases, switching, errors, failedResponses };
  fs.writeFileSync(path.join(output, 'browser-result.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
