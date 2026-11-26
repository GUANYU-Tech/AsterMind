import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {sceneCellDraft, sceneCellLayout, preserveOtherCellTopologies} from '../assets/scene-cell-parameters.js';
import {completedDestination} from '../assets/experiment_flow.js';

const presentation=JSON.parse(readFileSync(new URL('../cell/presentation.json',import.meta.url)));
const defaults=structuredClone(presentation.default_configuration);
defaults.scheduler.main.vllm={};
const capabilities={
  default_configuration:defaults,
  memory_policies:[],
  su_profiles:{nvidia_a800:{id:'nvidia_a800',fast_gpu_limit_per_satellite:8}},
  model_details:{'qwen3-4b':{id:'configs/model/qwen3-4b.json',experts:0}},
};

test('scene Cell drafts omit standalone request settings, including those in older layouts',()=>{
  const members=['a-1','a-2'];
  const draft=sceneCellDraft(capabilities,members,'detail_mode');
  assert.ok(!Object.hasOwn(draft,'requests'));
  const legacy={configuration:{...draft,requests:{source:'canonical',file:'archived.csv'}},workload_ground:'ground-00'};
  const restored=sceneCellDraft(capabilities,members,'detail_mode',legacy);
  assert.ok(!Object.hasOwn(restored,'requests'));
  assert.equal(legacy.configuration.requests.file,'archived.csv');
  assert.ok(capabilities.default_configuration.requests);
});

test('per-Cell form keeps explicit budget and remaps ranks when applied to another Cell',()=>{
  const first=['a-1','a-2'], second=['b-1','b-2'];
  const draft=sceneCellDraft(capabilities,first,'detail_mode',{
    deployment:{model:'configs/model/qwen3-4b.json',hardware_id:'nvidia_a800',gpus_per_satellite:2,tp:2,pp:1,
      max_model_len:400,max_batched_tokens:1024,num_kv_blocks:64},
    satellite_gpus:{'a-1':2,'a-2':2},rank_devices:['a-2/gpu1','a-1/gpu0'],
    scale_up_groups:[1,1],scale_out:{bandwidth_gbps:40,latency_ns:10},
  });
  const left=sceneCellLayout(draft,capabilities,first);
  const right=sceneCellLayout(draft,capabilities,second,{remap:true});
  assert.deepEqual(left.rank_devices,['a-2/gpu1','a-1/gpu0']);
  assert.deepEqual(right.rank_devices,['b-1/gpu0','b-1/gpu1']);
  assert.deepEqual(right.satellite_gpus,{'b-1':2,'b-2':2});
  assert.equal(right.deployment.max_batched_tokens,1024);
  assert.equal(right.deployment.expert_routing_seed,null);
  assert.deepEqual(right.scale_out,{bandwidth_gbps:40,latency_ns:10});
});

test('physical rebuild returns to the matching scene draft',()=>{
  assert.equal(completedDestination({physical_model_id:'new-topology'},'?next=scene&scene_draft=draft-1'),
    '/scene?model=new-topology&scene_draft=draft-1');
});

test('topology overrides stay tied to their original dataset',()=>{
  const original={physical_dataset:'old'};
  const rebuilt={physical_dataset:'new'};
  assert.equal(preserveOtherCellTopologies(original,rebuilt,new Set()),rebuilt);
});
