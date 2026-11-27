// Verify the local simulator entry and both workspaces against the running backend.
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const { chromium } = require('playwright');

async function checkSuPanel(page, lang, cards) {
  const panel = page.locator('#form-network-0');
  const help = panel.locator('.topology-description');
  await help.filter({hasText:lang === 'en' ? 'Each SU supports up to' : '每个 SU 最多'}).waitFor();
  assert.equal(await panel.locator('select, button').count(), 0);
  const bandwidth = page.locator('#field-hardware-intra_satellite_bandwidth_gbps');
  if (cards === 1) {
    assert.ok(await bandwidth.isDisabled());
    assert.match(await help.innerText(), /one SU domain|只有一个 SU 域/);
  } else if (cards === 16) {
    assert.ok(await bandwidth.isEnabled());
    assert.match(await help.innerText(), /one uplink to the same satellite egress|一条上联接入同一个卫星出口/);
  }
}

async function checkSuEditor(editor) {
  const chapter = name => editor.locator(`#chapter-rail [data-stage-target=${name}]`);
  await chapter('formation').click();
  const layouts = editor.locator('select[id$="-layout"] option');
  assert.deepEqual((await layouts.allTextContents()).sort(), ['环形','链式','全连接'].sort());
  await editor.locator('#field-hardware-default_hardware_id').selectOption('nvidia_h200');
  const cards = editor.locator('#field-hardware-accelerators_per_satellite');
  await cards.fill('16');
  await cards.press('Tab');
  await chapter('network').click();
  await checkSuPanel(editor, 'zh-CN', 16);
  const group = editor.locator('#field-hardware-su_group_sizes');
  const bandwidth = editor.getByLabel('SU 间上联带宽', {exact:true});
  const latency = editor.getByLabel('SU 间单跳时延', {exact:true});
  assert.equal(await group.getAttribute('placeholder'), '8 + 8');
  assert.match(await editor.locator('.topology-description').innerText(), /SU 0（8 卡）、SU 1（8 卡）。域内使用 NVLink，单向 3600 Gbps/);
  assert.equal(await bandwidth.inputValue(), '800');
  assert.equal(await latency.inputValue(), '500');
  for (const invalid of ['9 + 7', '8 + 7']) {
    await group.fill(invalid);
    assert.equal(await group.getAttribute('aria-invalid'), 'true');
  }
  await group.fill('4 + 4 + 8');
  await group.press('Tab');
  assert.match(await editor.locator('.topology-description').innerText(), /SU 2（8 卡）/);
  await bandwidth.fill('640');
  await latency.fill('321');
}

async function checkSavedInterconnects(page, defaults) {
  await page.goto('/cell/');
  const start = page.locator('[data-action=start-config]:enabled');
  await start.waitFor();
  const panel = page.locator('#form-network-0');
  for (const [type, label] of [['switch','共享交换互联'], ['fully_connected','全连接'], ['ring','环形']]) {
    const draft = structuredClone(defaults);
    draft.hardware.accelerators_per_satellite = 16;
    draft.topology.intra_satellite.type = type;
    await page.evaluate(value => localStorage.setItem('/cell/astermind-config', JSON.stringify(value)), draft);
    await page.reload();
    await start.click();
    await page.locator('#chapter-rail [data-stage-target=network]').click();
    await panel.getByText(label, {exact:true}).waitFor();
    assert.equal(await panel.locator('select').count(), 0);
    const current = () => page.locator('#advanced-config').inputValue().then(JSON.parse);
    assert.equal((await current()).topology.intra_satellite.type, type);
    await panel.getByRole('button', {name:'改用 SU 分域互联'}).click();
    await checkSuPanel(page, 'zh-CN', 16);
    const converted = await current();
    assert.equal(converted.topology.intra_satellite.type, 'profile_su');
    assert.equal(converted.hardware.accelerators_per_satellite, 16);
    assert.equal(converted.hardware.intra_satellite_bandwidth_gbps, draft.hardware.intra_satellite_bandwidth_gbps);
    assert.equal(converted.hardware.intra_satellite_latency_ns, draft.hardware.intra_satellite_latency_ns);
  }
  await page.reload();
  await start.click();
  await page.locator('#chapter-rail [data-stage-target=network]').click();
  await checkSuPanel(page, 'zh-CN', 16);
}

async function main() {
  const baseURL = new URL(process.argv[2] || 'http://127.0.0.1:8766').origin;
  const browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
    args: ['--enable-unsafe-swiftshader', '--no-proxy-server'],
  });
  try {
    const context = await browser.newContext({ baseURL, reducedMotion: 'reduce' });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', err => errors.push(err.message));
    const gitRoot = path.resolve(__dirname, '../..');
    const health = await (await context.request.get('/health')).json();
    assert.equal(health.commit, execFileSync('git', ['rev-parse', 'HEAD'], { cwd: gitRoot, encoding: 'utf8' }).trim());
    const caps = await (await context.request.get('/cell/api/capabilities')).json();
    assert.equal(Object.hasOwn(caps.default_configuration.runtime,'timeout_seconds'),false);
    const scopes = ['single_su', 'intra_cross_su', 'inter_satellite'];
    const config = structuredClone(caps.default_configuration);
    config.hardware.accelerators_per_satellite = 4;
    config.hardware.su_group_sizes = [2, 2];
    config.collective_policies = scopes.map(scope => ({
      scope, operation: 'all_reduce', algorithm: caps.collective_algorithms[scope].all_reduce.choices[0],
    }));
    const response = await context.request.post('/cell/api/config/validate', { data: config });
    assert.ok(response.ok(), await response.text());
    const { effective_configuration: effective } = await response.json();
    for (const sat of config.hardware.satellite_ids) {
      const hierarchy = effective.physical.intra_hierarchies[sat];
      assert.deepEqual(hierarchy.su_groups.map(group => group.length), [2, 2]);
      assert.equal(hierarchy.su_switches.length, 2);
      for (const source of hierarchy.su_switches) {
        const uplinks = effective.physical.links.filter(link => link.source === source);
        assert.deepEqual(uplinks.map(link => [link.target, link.bandwidth_gbps, link.latency_ns]),
          [[sat + '.exit', config.hardware.intra_satellite_bandwidth_gbps, config.hardware.intra_satellite_latency_ns]]);
      }
    }
    for (const { scope, operation, algorithm } of config.collective_policies) {
      assert.equal(effective.communication.cell_policies.cell.algorithms[scope][operation], algorithm);
    }

    const language = lang => page.waitForFunction(expected => document.documentElement.lang === expected, lang);
    const back = async destination => {
      await page.locator('.studio-logo').click();
      await page.waitForURL(baseURL + destination);
    };
    await page.goto('/');
    await page.waitForURL(baseURL + '/platform');
    assert.equal(await page.locator('a.portal-card').count(), 2);
    await page.goto('/en/');
    await language('en');
    await page.locator('.portal-card[href="/cell/"]').click();
    await language('en');
    await page.locator('[data-action=start-config]').click();
    await page.locator('#chapter-rail [data-stage-target=workload]').click();
    const modelIDs = await page.locator('#field-model option').evaluateAll(rows => rows.map(row => row.value));
    assert.ok(modelIDs.length >= 15);
    assert.ok(modelIDs.includes('kimi-k3') && modelIDs.includes('glm-5.3'));
    assert.ok(modelIDs.every(id => Object.hasOwn(caps.model_details, id)));
    await page.locator('#chapter-rail [data-stage-target=network]').click();
    await checkSuPanel(page, 'en', caps.default_configuration.hardware.accelerators_per_satellite);
    assert.ok(await page.locator('[data-field-path]').evaluateAll(fields =>
      fields.some(field => field.dataset.fieldPath === '["hardware","su_group_sizes"]')));
    await page.locator('.language-switch [lang="zh-CN"]').click();
    await language('zh-CN');
    await checkSuEditor(page);
    await page.locator('.language-switch [lang=en]').click();
    assert.match(await page.locator('.topology-description').innerText(), /SU 2 \(8 GPUs\)/);
    await page.locator('.language-switch [lang="zh-CN"]').click();
    await page.locator('.studio-logo').click();
    await page.waitForSelector('body[data-stage=home]');
    await back('/platform');
    await page.locator('.portal-card[href="/workspace"]').click();
    await page.locator('main a[href="/configuration"]').click();
    await page.locator('main a[href="/physical"]').first().click();
    await page.locator('#jumpCells').click();
    const formation = page.locator('#cells [data-formation-name]').first();
    await formation.locator('.formation-cell-row button').first().click();
    const dialog = page.locator('dialog.cell-workspace-dialog');
    await dialog.locator('footer button.primary:enabled').waitFor();
    assert.match(await dialog.locator('footer').innerText(), /应用到此 Cell/);
    assert.match(await dialog.locator('footer').innerText(), /应用到全部/);
    const editor = page.frameLocator('dialog.cell-workspace-dialog iframe');
    await checkSuEditor(editor);
    await dialog.locator('footer button.primary').click();
    await dialog.waitFor({ state: 'detached' });
    assert.match(await formation.locator('.formation-cell-row').first().innerText(), /单独配置/);
    await formation.locator('.formation-cell-row button').first().click();
    await editor.locator('#chapter-rail [data-stage-target=network]').click();
    assert.equal(await editor.locator('#field-hardware-su_group_sizes').inputValue(), '4 + 4 + 8');
    assert.equal(await editor.getByLabel('SU 间上联带宽', {exact:true}).inputValue(), '640');
    assert.equal(await editor.getByLabel('SU 间单跳时延', {exact:true}).inputValue(), '321');
    await dialog.locator('footer button').first().click();
    await dialog.waitFor({ state: 'detached' });
    const rows = await formation.locator('.formation-cell-row').allInnerTexts();
    assert.ok(rows.length > 0 && rows.every(row => row.includes('星座模板')));
    await back('/workspace');
    await back('/platform');
    assert.equal(await page.locator('.studio-logo').getAttribute('href'), 'https://guanyu-tech.github.io/AsterMind/');
    await language('zh-CN');
    await checkSavedInterconnects(page, caps.default_configuration);
    assert.deepEqual(errors, []);
    console.log('PASS', health.commit.slice(0, 8), Object.keys(caps.model_details).length, 'catalog models,',
      modelIDs.length, 'Detail Mode choices, three communication scopes, constellation editor, language and navigation');
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
