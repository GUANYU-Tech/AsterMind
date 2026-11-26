import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {synchronizeResources,setAutomaticDeployment,clearDependentDrafts} from '../cell/resource-sync.mjs';
import {editDependencies} from '../cell/config-edit.mjs';
import {configurationErrors} from '../cell/config-validation.mjs';
const capabilities=JSON.parse(readFileSync(new URL('../cell/presentation.json',import.meta.url)));
const fixture=()=>structuredClone(capabilities.default_configuration);
function validMapping(config) {
  assert.ok(config.deployment.placements.every(row=>config.hardware.satellite_ids.includes(row.satellite_id) && row.local_index>=0 && row.local_index<config.hardware.accelerators_per_satellite));
  assert.equal(new Set(config.deployment.placements.map(row=>`${row.satellite_id}/${row.local_index}`)).size,config.deployment.placements.length);
  assert.deepEqual(configurationErrors(config,capabilities.schema).filter(issue=>issue.loc[0]==='deployment'),[]);
}
test('default resources are runnable without manual ID or parallel edits',()=>{
  const config=fixture(); assert.equal(synchronizeResources(config,capabilities).error,''); validMapping(config);
});
test('shrinking and expanding resources automatically regenerate every dependent card reference',()=>{
  const config=fixture();
  for(const [satellites,cards] of [[3,1],[2,2],[4,2],[2,1]]) {
    config.hardware.satellite_count=satellites;
    editDependencies(config,['hardware','satellite_count'],satellites,capabilities);
    config.hardware.accelerators_per_satellite=cards;
    editDependencies(config,['hardware','accelerators_per_satellite'],cards,capabilities);
    validMapping(config);
    assert.ok(config.topology.inter_satellite.links.every(link=>config.hardware.satellite_ids.includes(link.source)&&config.hardware.satellite_ids.includes(link.target)));
  }
});
test('resource renaming repairs stale placement IDs and gateway card references',()=>{
  const config=fixture(); config.topology.intra_satellite.gateway_local_index=5;
  config.hardware.satellite_ids=['new-a','new-b'];
  editDependencies(config,['hardware','satellite_ids'],config.hardware.satellite_ids,capabilities);
  assert.equal(config.topology.intra_satellite.gateway_local_index,0); validMapping(config);
});
test('extra cards may stay idle when using every card violates model geometry',()=>{
  const config=fixture(); config.hardware.accelerators_per_satellite=30;
  const result=synchronizeResources(config,capabilities);
  assert.equal(result.error,''); assert.ok(result.idle>0); validMapping(config);
  const parallel=config.deployment.worker_pools[0].parallelism, shape=capabilities.model_shapes[config.model];
  assert.equal(shape.num_attention_heads%parallel.tensor_parallel_size,0);
  assert.equal(shape.num_hidden_layers%parallel.pipeline_parallel_size,0);
});
test('manual mode retains TP/PP while still repairing invalid card mappings',()=>{
  const config=fixture(); setAutomaticDeployment(config,false); config.hardware.accelerators_per_satellite=2;
  synchronizeResources(config,capabilities);
  assert.equal(config.deployment.worker_pools[0].parallelism.tensor_parallel_size,2);
  assert.equal(config.deployment.worker_pools[0].parallelism.pipeline_parallel_size,1); validMapping(config);
});

test('tied Qwen weights avoid every cross-pipeline plan, leaving surplus cards idle',()=>{
  for(const [satellites,cards] of [[3,1],[2,2]]) {
    const config=fixture();
    config.hardware.satellite_count=satellites;
    editDependencies(config,['hardware','satellite_count'],satellites,capabilities);
    config.hardware.accelerators_per_satellite=cards;
    const result=synchronizeResources(config,capabilities);
    assert.equal(result.error,''); assert.equal(result.required,2);
    assert.equal(result.idle,satellites*cards-2);
    assert.equal(config.deployment.worker_pools[0].parallelism.pipeline_parallel_size,1);
    validMapping(config);
  }
});

test('manual tied-weight PP incompatibility is reported before publishing',()=>{
  const config=fixture(); setAutomaticDeployment(config,false);
  config.hardware.accelerators_per_satellite=2;
  config.deployment.worker_pools[0].parallelism.pipeline_parallel_size=2;
  assert.match(synchronizeResources(config,capabilities).error,/PP/);
  assert.ok(configurationErrors(config,capabilities.schema).some(issue=>/PP/.test(issue.msg)));
});
test('resource changes discard obsolete derived drafts and retain unrelated manual input errors',()=>{
  const placement=JSON.stringify(['deployment','placements',1,'satellite_id']);
  const topology=JSON.stringify(['topology','inter_satellite','links',1,'target']);
  const request=JSON.stringify(['requests','synthetic','num_requests']);
  const drafts=new Map([[placement,{raw:'removed'}],[topology,{raw:'removed'}],[request,{raw:'invalid'}]]);
  clearDependentDrafts(drafts); assert.deepEqual([...drafts.keys()],[request]);
});
