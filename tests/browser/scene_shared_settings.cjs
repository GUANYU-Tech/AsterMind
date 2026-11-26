const assert = require('node:assert/strict');
const {chromium} = require('playwright');

async function verifyPdEndpoints(ui, page, embedded) {
  const stage = name => ui.locator('#chapter-rail [data-stage-target='+name+']').click();
  const configuration = async () => JSON.parse(await ui.locator('#advanced-config').inputValue());
  await stage('formation');
  await ui.locator('#field-hardware-default_hardware_id').selectOption('nvidia_h200');
  const cards = ui.locator('#field-hardware-accelerators_per_satellite');
  await cards.fill((await configuration()).hardware.satellite_count === 1 ? '16' : '8');
  await cards.press('Tab');
  if (!embedded) {
    await stage('network');
    await ui.getByRole('button',{name:'默认模式',exact:true}).click();
  }
  await stage('deployment');
  const route = ui.locator('details.editor-section:has(> summary:text-is("跨 pool 通信"))');
  assert.ok(await route.isHidden());
  await ui.locator('#field-deployment-mode').selectOption('pd_disagg');
  const source = ui.getByLabel('源 pool（Prefill）',{exact:true});
  const target = ui.getByLabel('目标 pool（Decode）',{exact:true});
  assert.ok(await source.isVisible());
  assert.ok(await target.isVisible());
  assert.equal(await route.locator('select,input').count(),2);
  assert.equal(await source.inputValue(),'prefill');
  assert.equal(await target.inputValue(),'decode');
  await ui.locator('#match-parallelism').click();
  const {placements} = (await configuration()).deployment;
  assert.equal(placements.length,16);
  await target.selectOption('prefill');
  assert.equal(await source.inputValue(),'decode');
  const endpoint = embedded ? '/cell/api/scene/validate' : '/cell/api/config/validate';
  const pending = page.waitForResponse(r=>r.url().endsWith(endpoint),{timeout:150000});
  await stage('runtime');
  const response = await pending;
  assert.ok(response.ok(),await response.text());
  const data = await response.json();
  const cell = embedded ? data.cell : data.effective_configuration.cells[0];
  assert.equal(cell.deployment,'pd_disagg');
  for (const [pool,role] of [['decode','prefill'],['prefill','decode']]) {
    const expected = placements.filter(row=>row.worker_pool===pool)
      .map(row=>row.satellite_id+(embedded?'/':'.')+'gpu'+row.local_index).sort();
    assert.equal(expected.length,8);
    assert.deepEqual(cell.placements.filter(row=>row.pool===role).map(row=>row.device).sort(),expected);
  }
  await stage('deployment');
  await source.selectOption('prefill');
  assert.equal(await target.inputValue(),'decode');
  await ui.locator('#field-deployment-mode').selectOption('colocated');
  assert.ok(await route.isHidden());
}

async function main() {
  const baseURL = process.argv[2] || 'http://127.0.0.1:8766';
  const browser = await chromium.launch({channel:process.env.PLAYWRIGHT_CHANNEL, args:['--no-proxy-server','--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({baseURL, reducedMotion:'reduce'});
    await page.goto('/physical#cells');
    const commTab = page.getByRole('tab',{name:'通信星座',exact:true});
    await commTab.click();
    const time = page.locator('#time input');
    assert.equal(await time.count(),3);
    assert.ok(await page.locator('#time').evaluate(el=>
      el.getBoundingClientRect().bottom<=document.querySelector('[role=tablist]').getBoundingClientRect().top));
    const values = ['2026-07-21T01:00:00Z','75','5'];
    for (const [i,value] of values.entries()) await time.nth(i).fill(value);
    for (const tab of await page.getByRole('tab').all()) {
      await tab.click();
      for (const [i,value] of values.entries()) {
        assert.ok(await time.nth(i).isVisible());
        assert.equal(await time.nth(i).inputValue(),value);
      }
    }
    await page.locator('.language-switch [lang=en]').click();
    await page.getByRole('tab',{name:'Communications Constellation',exact:true}).waitFor();
    await page.getByText(/Edit shell-specific budgets under Communications Constellation/).waitFor();
    await page.locator('.language-switch [lang="zh-CN"]').click();
    await page.locator('[data-tab=compute]').click();
    const sceneMode = page.locator('#computeMode');
    const rows = page.locator('.formation-cell-row button');
    const dialog = page.locator('dialog.cell-workspace-dialog');
    const editor = page.frameLocator('dialog.cell-workspace-dialog iframe');
    const apply = dialog.locator('footer button.primary');
    async function openCell(index, name) {
      await rows.nth(index).click();
      await dialog.locator('footer button.primary:enabled').waitFor();
      await editor.locator('#chapter-rail [data-stage-target=workload]').click();
      assert.equal(await editor.locator('#field-compute_mode').count(),0);
      assert.ok((await editor.locator('#supported-models').innerText()).includes(name));
    }
    assert.equal(await page.locator('#epStrategy').count(),0);
    await page.locator('.formation-cell-row input').first().fill('边缘计算一组');
    await page.locator('.formation-cell-row input').first().press('Tab');
    const advanced = page.locator('#cells [data-formation-name] > details').first();
    await advanced.locator(':scope > summary').click();
    const limits=advanced.getByLabel(/接入成员对外 ISL 终端数/);
    assert.equal(await advanced.getByLabel('每成员 ISL 终端数',{exact:true}).count(),0);
    const memberCount=(await advanced.getByLabel(/对外接入成员编号/).inputValue()).split(',').length;
    await limits.fill(Array(memberCount).fill('2').join(','));
    await openCell(0,'Detail Mode');
    await verifyPdEndpoints(editor,page,true);
    await dialog.locator('header button').click();
    await dialog.waitFor({state:'detached'});
    await openCell(0,'Detail Mode');
    assert.equal(await editor.locator('.roofline-settings,#automatic-deployment,#field-ep_strategy').count(),0);
    assert.equal(await editor.locator('#form-runtime-0 [id$="-max_model_len"],#field-memory_policy_file,#field-detail_prediction_method').count(),0);
    assert.deepEqual(await editor.locator('#config-profile .hardware-specs dt').allTextContents(),['模型','计算模式']);
    await editor.locator('#chapter-rail [data-stage-target=runtime]').click();
    const json = editor.locator('#advanced-config');
    for (const details of ['details:has(> #config-profile)','details:has(> #advanced-config)']) {
      await editor.locator(details+' > summary').click();
    }
    const draft = JSON.parse(await json.inputValue());
    assert.equal(Object.hasOwn(draft,'compute_mode'),false);
    await json.fill(JSON.stringify({...draft,compute_mode:'fast_mode'}));
    await editor.locator('#apply-json').click();
    await dialog.locator('[role=status]').filter({hasText:/统一设置计算模式|compute mode for all Cells/i}).waitFor();
    await editor.locator('#discard-json').click();
    const checked = page.waitForResponse(r=>r.url().endsWith('/cell/api/scene/validate') && r.request().postDataJSON().compute_mode==='detail_mode');
    await apply.click();
    const response = await checked;
    assert.ok(response.ok(), await response.text());
    await dialog.waitFor({state:'detached'});
    const data = response.request().postDataJSON();
    assert.equal(data.layout.configuration.compute_mode,'detail_mode');
    const mismatch = await page.request.post('/cell/api/scene/validate',{data:{...data,compute_mode:'fast_mode'}});
    assert.equal(mismatch.status(),400);
    assert.match((await mismatch.json()).detail,/编队与全场景的计算模式必须一致/);
    assert.equal(await sceneMode.inputValue(),'detail_mode');
    await sceneMode.selectOption('fast_mode');
    await page.locator('#linkContention').check();
    for (const index of [0,1]) {
      await openCell(index,'Fast Mode');
      if(index===0){
        await editor.locator('#chapter-rail [data-stage-target=formation]').click();
        await editor.locator('#field-hardware-default_hardware_id').selectOption('nvidia_h200');
        assert.equal(await editor.locator('#hardware-profile .roofline-settings').count(),0);
        await editor.locator('#chapter-rail [data-stage-target=workload]').click();
        const box=editor.locator('#form-workload-0 .roofline-settings');
        assert.ok(await box.isVisible());
        assert.equal(await box.locator('#field-fast_roofline-tensor_percent').inputValue(),'70');
        assert.match(await box.innerText(),/692\.58 TFLOP\/s/);
        assert.match(await box.innerText(),/4\.608 TB\/s/);
        await box.locator('#field-fast_roofline-tensor_percent').fill('50');
        assert.match(await box.innerText(),/494\.7 TFLOP\/s/);
        const pending=page.waitForResponse(r=>r.url().endsWith('/cell/api/scene/validate'));
        await apply.click();
        const validated=await pending;
        assert.ok(validated.ok(),await validated.text());
        const cell=(await validated.json()).cell;
        assert.deepEqual(cell.fast_roofline,{tensor_percent:50,hbm_percent:96});
        assert.equal(cell.fast_link_contention,true);
      }else await dialog.locator('header button').click();
      await dialog.waitFor({state:'detached'});
    }
    assert.equal(await sceneMode.inputValue(),'fast_mode');
    // Inspect the save request without adding a test draft to the user's catalog.
    await page.route('**/save-physical',route=>route.fulfill({json:{
      ...route.request().postDataJSON(),id:'test-shared-settings',saved_at:'2026-07-21T01:00:00Z',
    }}));
    const saveRequest = page.waitForRequest('**/save-physical');
    await page.locator('#save').click();
    const saved = (await saveRequest).postDataJSON();
    assert.deepEqual(saved.physical_network.time,{epoch_utc:values[0],duration_s:75,step_s:5});
    assert.equal(saved.compute_configuration.mode,'fast_mode');
    const override = Object.values(saved.compute_configuration.formations).flatMap(f=>Object.values(f.cells))[0];
    assert.equal(override.configuration.compute_mode,'fast_mode');
    assert.equal(saved.compute_configuration.fast_link_contention,true);
    assert.equal(override.configuration.fast_link_contention,true);
    assert.equal(override.configuration.fast_roofline.tensor_percent,50);
    const settings=Object.values(saved.compute_configuration.formations)[0];
    assert.deepEqual(Object.values(settings.cell_names),['边缘计算一组']);
    assert.equal(saved.physical_network.constellations.compute_formations[0].ingress_isl_terminals[0],2);
    await page.goto('/cell/');
    await page.locator('[data-action=start-config]:enabled').click();
    await verifyPdEndpoints(page,page,false);
    await page.locator('#chapter-rail [data-stage-target=formation]').click();
    await page.locator('#restore-defaults').click();
    await page.locator('#field-hardware-default_hardware_id').selectOption('nvidia_h200');
    await page.locator('#chapter-rail [data-stage-target=network]').click();
    await page.getByRole('button',{name:'默认模式',exact:true}).click();
    await page.locator('#chapter-rail [data-stage-target=workload]').click();
    const mode = page.locator('#field-compute_mode');
    const panel = page.locator('#form-workload-0 .roofline-settings');
    assert.equal(await panel.count(),0);
    await mode.selectOption('fast_mode');
    await page.locator('#field-model').selectOption('qwen3-4b');
    assert.ok(await panel.isVisible());
    const tensor = panel.getByLabel('算力效率',{exact:true});
    const hbm = panel.getByLabel('HBM 带宽效率',{exact:true});
    const rates = panel.locator('output');
    assert.equal(await tensor.inputValue(),'70');
    assert.equal(await hbm.inputValue(),'96');
    assert.deepEqual(await rates.allTextContents(),['692.58 TFLOP/s','4.608 TB/s']);
    await tensor.fill('0');
    assert.equal(await rates.first().textContent(),'—');
    await tensor.fill('50');
    await page.locator('#chapter-rail [data-stage-target=formation]').click();
    assert.ok(await page.locator('#hardware-profile').isVisible());
    assert.equal(await page.locator('#hardware-profile .roofline-settings').count(),0);
    await page.locator('#chapter-rail [data-stage-target=workload]').click();
    assert.equal(await tensor.inputValue(),'50');
    await hbm.fill('80');
    assert.deepEqual(await rates.allTextContents(),['494.7 TFLOP/s','3.84 TB/s']);
    await mode.selectOption('detail_mode');
    assert.equal(await panel.count(),0);
    await mode.selectOption('fast_mode');
    await page.reload();
    await page.locator('[data-action=start-config]:enabled').click();
    await page.locator('#chapter-rail [data-stage-target=workload]').click();
    assert.equal(await mode.inputValue(),'fast_mode');
    assert.deepEqual(await rates.allTextContents(),['494.7 TFLOP/s','3.84 TB/s']);
    const preflight = page.waitForResponse(r=>r.url().endsWith('/cell/api/config/validate'),{timeout:150000});
    await page.locator('#chapter-rail [data-stage-target=runtime]').click();
    const checkedConfig = await preflight;
    assert.ok(checkedConfig.ok(),await checkedConfig.text());
    assert.deepEqual((await checkedConfig.json()).effective_configuration.cells[0].fast_roofline,
      {tensor_percent:50,hbm_percent:80});
    console.log('PASS shared time across four tabs, translated labels, save payload, two Cells, backend preflight and standalone mode');
  } finally {
    await browser.close();
  }
}

main().catch(error=>{console.error(error);process.exitCode=1});
