const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

function catalog() {
  const lines = fs.readFileSync(path.join(__dirname, '../../viz/assets/single-shell.tle'), 'utf8').split(/\r?\n/);
  const first = lines.find(line => line.startsWith('1 '));
  const second = lines.find(line => line.startsWith('2 '));
  const checksum = body => body + [...body].reduce((sum,c) => sum + (c==='-' ? 1 : Number(c)||0),0) % 10;
  const angle = degrees => degrees.toFixed(4).padStart(8);
  const rows = [];
  for (let i = 0; i <= 10368; i++) {
    const id = String(10000+i);
    // A 96 x 108 regular shell plus a single object at another inclination.
    const inc = i===10368 ? 70 : 53;
    const raan = (Math.floor(i/108)%96)*360/96;
    const phase = (i%108)*360/108;
    rows.push('DEMO-SATELLITE-'+id,checksum(first.slice(0,2)+id+first.slice(7,68)),
      checksum(second.slice(0,2)+id+' '+angle(inc)+' '+angle(raan)+second.slice(25,43)+angle(phase)+second.slice(51,68)));
  }
  return Buffer.from(rows.join('\n')+'\n');
}

(async()=>{
  const base = process.argv[2] || 'http://127.0.0.1:8781';
  const browser = await chromium.launch({channel:process.env.PLAYWRIGHT_CHANNEL, args:['--no-proxy-server']});
  try {
    const page=await browser.newPage({baseURL:base});
    page.setDefaultTimeout(90000);
    await page.goto('/physical');
    const ids=page.locator('#shells input[aria-label="壳层 ID"]');
    await ids.first().waitFor();
    const before=await ids.count();
    const dialog=page.locator('#addCommunication');
    const choose=async method=>{
      await page.locator('#addShell').click();
      assert.equal(await dialog.locator('#confirmCommunication').isDisabled(),true);
      await dialog.locator('input[value="'+method+'"]').check();
    };
    assert.equal(await page.locator('#tleFile').isVisible(),false);
    await choose('parameters');
    await dialog.getByLabel('轨道模型').selectOption('dawn_dusk');
    assert.equal(await dialog.getByLabel('中心轨道升交点地方时').isVisible(),true);
    await dialog.locator('#cancelCommunication').click();
    assert.equal(await ids.count(),before);
    const time={epoch_utc:'2026-11-21T02:03:04.000000Z',duration_s:420,step_s:2};
    const start=page.getByLabel('实验开始时间（UTC）',{exact:true});
    const duration=page.getByLabel('实验时长 s',{exact:true});
    const step=page.getByLabel('几何时间片间隔 s',{exact:true});
    await start.fill(time.epoch_utc);
    await duration.fill(String(time.duration_s));
    await step.fill(String(time.step_s));
    const checkTime=async()=>{
      assert.deepEqual([await start.inputValue(),await duration.inputValue(),await step.inputValue()],
        [time.epoch_utc,String(time.duration_s),String(time.step_s)]);
    };
    const model=(await (await page.request.get('/templates/starlink_550')).json());
    const file={name:'synthetic-10369.tle',mimeType:'text/plain',buffer:catalog()};
    assert.ok(file.buffer.length>1500000);
    const importFile=async()=>{
      await page.locator('#tleFile').setInputFiles(file);
      const response=page.waitForResponse(r=>r.url().endsWith('/inspect-tle'));
      await dialog.locator('#confirmCommunication').click();
      const res=await response;
      assert.equal(res.status(),200,(await res.text()).slice(0,240));
    };
    await choose('tle');
    assert.equal(await dialog.locator('#useTleEpoch').count(),0);
    await importFile();
    await page.locator('#shells [role=status]').filter({hasText:'10369'}).waitFor();
    assert.equal(await ids.count(),before+2);
    await checkTime();
    await choose('tle');
    await importFile();
    await page.waitForFunction(()=>document.querySelector('#tleStatus').textContent.includes('目录号已存在'));
    assert.equal(await ids.count(),before+2);
    await page.keyboard.press('Escape');
    const regular=page.locator('#shells').getByLabel('标称轨道面数（含缺失面）',{exact:true}).first();
    assert.equal(await regular.inputValue(),'96');
    await page.getByLabel('每面标称槽位数（含空位）',{exact:true}).first().fill('110');
    await choose('parameters');
    const planes=dialog.getByLabel('轨道面数',{exact:true});
    await planes.fill('0');
    await dialog.locator('#confirmCommunication').click();
    assert.equal(await planes.evaluate(input=>input.validity.valid),false);
    assert.equal(await ids.count(),before+2);
    await planes.fill('3');
    await dialog.getByLabel('每面卫星数',{exact:true}).fill('5');
    await dialog.locator('#confirmCommunication').click();
    assert.equal(await page.locator('#shells select[aria-label="轨道来源"]').last().inputValue(),'walker_delta');
    await page.locator('#name').fill('TLE mixed-shell regression (10,369 synthetic observations)');
    const saved=page.waitForResponse(r=>r.url().endsWith('/save-physical'));
    await page.locator('#save').click();
    const res=await saved;assert.equal(res.status(),200,(await res.text()).slice(0,300));
    const doc=await res.json();const cfg=doc.physical_network;
    assert.deepEqual(cfg.time,time);
    assert.equal(cfg.constellations.communication_shells.at(-1).source.planes,3);
    assert.equal(cfg.constellations.communication_shells.at(-1).source.satellites_per_plane,5);
    assert.deepEqual(cfg.ground_ingresses.map(g=>g.ingress_id),model.ground_ingresses.map(g=>g.ingress_id));
    assert.deepEqual(cfg.constellations.compute_formations.map(g=>g.name),model.constellations.compute_formations.map(g=>g.name));
    assert.deepEqual(cfg.route_probes,model.route_probes);
    await page.waitForURL(url=>url.pathname==='/physical' && url.searchParams.get('id')===doc.id);
    await page.reload();
    await regular.waitFor();
    await checkTime();
    assert.equal(await page.getByLabel('每面标称槽位数（含空位）',{exact:true}).first().inputValue(),'110');
    assert.equal(await ids.count(),before+3);
    console.log('PASS',doc.id,file.buffer.length,'bytes, 10,369 satellites, preserved model, editable saved shells');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
