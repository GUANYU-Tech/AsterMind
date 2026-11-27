// Explicit end-to-end run. Creates a small physical model and launches one full-scene job.
const assert = require('node:assert/strict');
const {chromium} = require('playwright');

async function main() {
  const baseURL = process.argv[2] || 'http://127.0.0.1:8766';
  const browser = await chromium.launch({channel:process.env.PLAYWRIGHT_CHANNEL, args:['--no-proxy-server','--enable-unsafe-swiftshader']});
  try {
    const context = await browser.newContext({baseURL, reducedMotion:'reduce'});
    const page = await context.newPage();
    const errors=[];
    page.on('pageerror', e=>errors.push(e.message));
    page.on('response',async r=>{if(r.status()>=400)console.error(r.status(),r.url(),(await r.text()).slice(0,600))});
    const physical=await (await context.request.get('/templates/sharing_demo')).json();
    physical.constellations.communication_shells=[];
    const formation=physical.constellations.compute_formations[0];
    formation.source.clusters_per_plane=1;
    Object.assign(formation.cluster,{satellites_per_cluster:2,motion_model:'static_formation'});
    delete formation.cluster.grid_shape;
    formation.sgl_terminals=1;
    physical.ground_ingresses=[{...physical.ground_ingresses[0],lon_deg:62,ground_kind:'user',terminal_count:1}];
    physical.network.ground_roles=['compute'];physical.route_probes=[];
    physical.time.duration_s=12;
    const response=await context.request.post('/save-physical',{data:{name:'编队配置联调验证',physical_network:physical}});
    assert.ok(response.ok(),await response.text());
    await page.goto('/physical?id='+(await response.json()).id+'#cells');
    await page.getByRole('tab',{name:'连接与链路预算'}).click();
    await page.getByLabel('星间链路计算模型',{exact:true}).selectOption('fixed');
    const budget=page.locator('[data-link-profile="cell_isl"]');
    await budget.getByLabel('固定单向带宽（Gbps）',{exact:true}).fill('123');
    await page.getByRole('tab',{name:'计算星座与编队'}).click();
    await page.locator('.formation-cell-row button').first().click();
    const dialog=page.locator('dialog.cell-workspace-dialog');
    const editor=page.frameLocator('dialog.cell-workspace-dialog iframe');
    const stage=name=>editor.locator('#chapter-rail [data-stage-target='+name+']').click();
    await dialog.locator('footer button.primary:enabled').waitFor();
    const pd=process.argv.includes('--pd');
    if(pd){
      await editor.locator('#field-hardware-default_hardware_id').selectOption('nvidia_h200');
      await editor.locator('#field-hardware-accelerators_per_satellite').fill('8');
      await editor.locator('#field-hardware-accelerators_per_satellite').press('Tab');
    }
    await stage('network');
    await editor.getByText('编队内星间链路使用「连接与链路预算」中的设置。',{exact:true}).waitFor();
    await stage('workload');
    assert.equal(await editor.locator('#field-model').isVisible(),true);
    assert.equal(await editor.locator('[data-group="workload-1"],#request-upload').count(),0);
    assert.equal(Object.hasOwn(JSON.parse(await editor.locator('#advanced-config').inputValue()),'requests'),false);
    const outputTokens=pd?2:1;
    await stage('deployment');
    assert.equal(await editor.locator('#field-deployment-mode option[value=pd_disagg]').count(),1);
    if(pd){
      await editor.locator('#field-deployment-mode').selectOption('pd_disagg');
      await editor.locator('#automatic-deployment').click();
    }
    await stage('runtime');
    assert.equal(await editor.locator('[id*=timeout_seconds],#run-simulation').count(),0);
    const validation=page.waitForResponse(r=>r.url().endsWith('/cell/api/scene/validate'));
    await dialog.locator('footer button.primary').click();
    const check=await validation;assert.ok(check.ok(),await check.text());
    if(pd){
      const {cell}=await check.json();
      assert.equal(cell.deployment,'pd_disagg');assert.equal(cell.placements.length,16);
      assert.deepEqual([...new Set(cell.placements.map(row=>row.pool))],['prefill','decode']);
    }
    await dialog.waitFor({state:'detached'});
    const next=page.waitForResponse(r=>r.url().endsWith('/save-physical')&&r.request().method()==='POST');
    await page.locator('#next').click();
    const revision=await (await next).json();
    const savedLayout=Object.values(Object.values(revision.compute_configuration.formations)[0].cells)[0];
    assert.equal(Object.hasOwn(savedLayout,'workload_ground'),false);
    assert.equal(Object.hasOwn(savedLayout.configuration,'requests'),false);
    const profile=revision.physical_network.network.link_profiles.cell_isl;
    assert.equal(profile.bandwidth_bps,123e9);assert.equal(profile.setup_time_s,2);
    await page.waitForURL('**/scene?**',{timeout:180000});
    await page.locator('#run:enabled').waitFor({timeout:120000});
    const formationTable=page.locator('#cellDeploymentList');
    assert.equal(await formationTable.locator('tbody tr').count(),1);
    assert.equal(await formationTable.locator('thead th').count(),3);
    assert.equal(await formationTable.locator('button,a,input,select').count(),0);
    const topology=page.locator('.topology-panel');
    await topology.locator('.topology-edge').first().focus();
    await topology.locator('.topology-edge').first().press('Enter');
    assert.equal(await topology.locator('input,button').count(),0);
    assert.equal(await page.locator('#requests tr').count(),0);
    await page.locator('#generateRequests').click();
    await page.locator('#requestBatchCount').fill('1');
    await page.locator('details:has(> #requestBatchDistribution) > summary').click();
    await page.locator('#requestBatchDistribution').selectOption('simultaneous');
    await page.locator('#requestBatchInput').fill('4');
    await page.locator('#requestBatchOutput').fill(String(outputTokens));
    await page.locator('#previewRequestBatch').click();
    await page.locator('#applyRequestBatch:enabled').click();
    assert.equal(await page.locator('#requests tr').count(),1);
    const sceneResponse=page.waitForResponse(r=>r.url().endsWith('/save-scene')&&r.request().method()==='POST');
    await page.locator('#run').click();
    const scene=await (await sceneResponse).json();
    assert.equal(scene.experiment.requests.length,1);
    assert.ok(scene.experiment.physical_dataset);
    assert.equal(scene.experiment.requests[0].user_device,physical.ground_ingresses[0].ingress_id+'/terminal');
    assert.equal(scene.experiment.requests[0].input_bytes,16);
    assert.ok(scene.experiment.physical.links.some(l=>l.bandwidth_gbps===123));
    if(pd)assert.equal(scene.experiment.cells[0].deployment,'pd_disagg');
    await page.waitForURL('**/runs/*/progress**',{timeout:60000});
    const id=new URL(page.url()).pathname.split('/')[2];
    console.log('Full scene run',id,'scene',scene.id,'physical model',revision.id);
    await expectCompletedRun(context.request,id);
    await page.waitForURL('**/results?**',{timeout:300000});
    const summary=await (await context.request.get('/runs/'+id+'/native-summary')).json();
    assert.deepEqual(Object.values(summary.requests).map(r=>[r.status,r.received_tokens]),[['completed',outputTokens]]);
    assert.deepEqual(errors,[]);
    console.log('PASS',page.url());
  } finally {await browser.close()}
}
async function expectCompletedRun(request,id) {
  const deadline=Date.now()+300000;
  while(Date.now()<deadline) {
    const response=await request.get('/runs/'+id+'/status');
    assert.ok(response.ok(),await response.text());
    const status=await response.json();
    if(status.state==='completed') return;
    assert.ok(['queued','running','preparing'].includes(status.state),status.error||status.log_tail||status.state);
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  throw Error('Browser test timed out waiting for run '+id);
}
main().catch(e=>{console.error(e);process.exitCode=1});
