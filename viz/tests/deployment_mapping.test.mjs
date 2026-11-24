import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {mappingDraft, matchParallelismToResources, deploymentCapacity} from '../cell/deployment-mapping.mjs';

const defaults = JSON.parse(readFileSync(new URL('../cell/presentation.json', import.meta.url))).default_configuration;
function fixture(satellites = 2, cards = 1, tp = 1, pp = 2) {
  const config = structuredClone(defaults);
  Object.assign(config.hardware, {satellite_count:satellites, satellite_ids:Array.from({length:satellites}, (_, i) => `sat-${i}`), accelerators_per_satellite:cards});
  config.deployment.worker_pools = [config.deployment.worker_pools[0]];
  Object.assign(config.deployment.worker_pools[0].parallelism, {tensor_parallel_size:tp, pipeline_parallel_size:pp});
  config.deployment.placements = [];
  return config;
}
const locations = rows => rows.map(row => `${row.satellite_id}/${row.local_index}`);
function assertCoverage(config) {
  const rows = config.deployment.placements;
  assert.equal(rows.length, deploymentCapacity(config).physical);
  assert.equal(new Set(locations(rows)).size, rows.length);
  assert.ok(rows.every(row => config.hardware.satellite_ids.includes(row.satellite_id) && Number.isInteger(row.local_index) && row.local_index >= 0 && row.local_index < config.hardware.accelerators_per_satellite));
}

test('default mapping covers each card and rank once without mutating the config', () => {
  const config = fixture(2, 2, 2, 2), before = structuredClone(config);
  const rows = mappingDraft(config);
  assert.deepEqual(locations(rows), ['sat-0/0','sat-0/1','sat-1/0','sat-1/1']);
  assert.deepEqual(rows.map(row => [row.parallel_coordinate.pp, row.parallel_coordinate.tp]), [[0,0],[0,1],[1,0],[1,1]]);
  assert.deepEqual(config, before);
});

test('reserve later manual assignments before filling new ranks; explicit reset returns to inventory order', () => {
  const config = fixture();
  const name = config.deployment.worker_pools[0].name;
  config.deployment.placements = [{worker_pool:name, parallel_coordinate:{dp:0,cp:0,pp:1,tp:0}, satellite_id:'sat-0',local_index:0}];
  assert.deepEqual(locations(mappingDraft(config)), ['sat-1/0','sat-0/0']);
  assert.deepEqual(locations(mappingDraft(config,{reset:true})), ['sat-0/0','sat-1/0']);
});

test('resource edits keep valid manual cards and replace removed cards; no implicit TP/PP edit', () => {
  const config = fixture(2,2,2,2);
  config.deployment.placements = mappingDraft(config);
  const parallel = structuredClone(config.deployment.worker_pools[0].parallelism);
  config.hardware.satellite_ids = ['sat-0','renamed'];
  config.deployment.placements = mappingDraft(config);
  assertCoverage(config);
  assert.deepEqual(locations(config.deployment.placements), ['sat-0/0','sat-0/1','renamed/0','renamed/1']);
  assert.deepEqual(config.deployment.worker_pools[0].parallelism,parallel);
});

test('insufficient physical cards remain unassigned instead of reusing a card', () => {
  const config = fixture(2,1,1,4), rows = mappingDraft(config);
  assert.deepEqual(locations(rows.slice(0,2)), ['sat-0/0','sat-1/0']);
  assert.ok(rows.slice(2).every(row => row.satellite_id === '' && row.local_index === null));
});

test('explicit resource matching preserves divisible TP and generates complete mappings', () => {
  const config = fixture(3,4,2,1);
  assert.deepEqual(matchParallelismToResources(config), {tp:2,pp:6});
  assertCoverage(config);
});

test('nondivisible TP falls back to a per-satellite divisor', () => {
  const config = fixture(3,2,4,1);
  assert.deepEqual(matchParallelismToResources(config), {tp:2,pp:3});
  assertCoverage(config);
});

test('PD pools receive the same parallel shape and disjoint physical cards', () => {
  const config = fixture(2,2,1,1);
  const second = structuredClone(config.deployment.worker_pools[0]); second.name = 'decode';
  config.deployment.worker_pools.push(second);
  assert.deepEqual(matchParallelismToResources(config), {tp:1,pp:2});
  assertCoverage(config);
});

test('Fast Mode maps every replica and retains manual assignments by DP coordinate', () => {
  const config = fixture(2, 4, 2, 2);
  config.compute_mode = 'fast_mode';
  const pool = config.deployment.worker_pools[0];
  Object.assign(pool.parallelism, {data_parallel_size:2, enable_expert_parallel:true});
  config.deployment.placements = [{
    worker_pool:pool.name, parallel_coordinate:{dp:1, pp:0, cp:0, tp:0},
    satellite_id:'sat-0', local_index:0,
  }];
  config.deployment.placements = mappingDraft(config);
  assertCoverage(config);
  assert.equal(deploymentCapacity(config).required,8);
  assert.deepEqual(config.deployment.placements.map(({parallel_coordinate:p})=>[p.dp,p.pp,p.tp]), [
    [0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],
  ]);
  assert.equal(locations(config.deployment.placements)[4],'sat-0/0');
});

test('odd inventory leaves spare cards idle; unsupported parallelism still fails explicitly', () => {
  const config = fixture(3,1,1,1);
  const second = structuredClone(config.deployment.worker_pools[0]); second.name = 'decode';
  config.deployment.worker_pools.push(second);
  assert.deepEqual(matchParallelismToResources(config),{tp:1,pp:1});
  assert.equal(config.deployment.placements.length,2);
  assert.equal(new Set(locations(config.deployment.placements)).size,2);
  config.deployment.worker_pools.pop();
  config.deployment.worker_pools[0].parallelism.data_parallel_size = 2;
  assert.throws(() => matchParallelismToResources(config), /DP/);
  assert.equal(mappingDraft(config),config.deployment.placements);
});
