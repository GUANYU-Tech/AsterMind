import {normalizeComputeSettings} from '../cell/compute-config.mjs';
import {mappingDraft} from '../cell/deployment-mapping.mjs';
import {editDependencies} from '../cell/config-edit.mjs';

const gpuAddress = (satellite, slot) => `${satellite}/gpu${slot}`;
const modelId = file => file.split('/').at(-1).replace(/\.json$/, '');

export function sceneCellDraft(capabilities, members, mode, layout) {
  const draft = normalizeComputeSettings(structuredClone(layout?.configuration ?? capabilities.default_configuration), capabilities);
  delete draft.requests;
  if (layout?.configuration) {
    if (draft.compute_mode !== mode) {
      draft.compute_mode = mode;
      editDependencies(draft,['compute_mode'],mode,capabilities);
    }
    return draft;
  }
  const options = layout?.deployment || {};
  const base = draft.deployment.worker_pools[0];
  const scheduler = draft.scheduler[base.name];
  const fast = mode === 'fast_mode';
  Object.assign(draft, {
    ...(options.fast_roofline?{fast_roofline:structuredClone(options.fast_roofline)}:{}),
    compute_mode: mode, model: options.model ? modelId(options.model) : draft.model,
    detail_prediction_method: fast ? null : options.detail_prediction_method || 'auto',
    memory_policy_file: fast ? options.memory_policy_file || '' : '',
    fast_glm_dsa_score: fast ? options.fast_glm_dsa_score || '' : '',
    expert_routing_seed: options.expert_routing_seed ?? 0,
    collective_policies: structuredClone(layout?.collective_policies || []),
  });
  const registered = capabilities.memory_policies.find(row=>row.file===draft.memory_policy_file);
  if (registered) draft.memory_policy_file=registered.id;
  Object.assign(draft.hardware, {
    satellite_ids:[...members],satellite_count:members.length,
    default_hardware_id:options.hardware_id || draft.hardware.default_hardware_id,
    accelerators_per_satellite:options.gpus_per_satellite ?? 1,
    su_group_sizes:[...(layout?.scale_up_groups || [])],
    intra_satellite_bandwidth_gbps:layout?.scale_out?.bandwidth_gbps ?? draft.hardware.intra_satellite_bandwidth_gbps,
    intra_satellite_latency_ns:layout?.scale_out?.latency_ns ?? draft.hardware.intra_satellite_latency_ns,
  });
  Object.assign(base.parallelism, {
    tensor_parallel_size:options.tp ?? 1,pipeline_parallel_size:options.pp ?? 1,
    data_parallel_size:options.dp ?? 1,enable_expert_parallel:options.expert_parallel ?? false,
  });
  scheduler.max_requests=options.max_requests ?? scheduler.max_requests;
  scheduler.max_batched_tokens=options.max_batched_tokens ?? null;
  scheduler.num_kv_blocks=options.num_kv_blocks ?? null;
  scheduler.kv_block_size_tokens=options.kv_block_size_tokens ?? null;
  Object.assign(scheduler.vllm, {
    model:draft.model,max_model_len:options.max_model_len ?? null,
    enable_chunked_prefill:options.enable_chunked_prefill ?? (layout?mode!=='fast_mode':true),
    long_prefill_token_threshold:options.long_prefill_token_threshold ?? 0,
    enable_prefix_caching:options.prefix_caching ?? false,
  });
  if (options.deployment==='pd_disagg') {
    draft.deployment.mode='pd_disagg';
    draft.deployment.worker_pools=['prefill','decode'].map(role=>({...structuredClone(base),name:role,role}));
    draft.scheduler=Object.fromEntries(['prefill','decode'].map(role=>[role,structuredClone(scheduler)]));
  }
  draft.deployment.placements=mappingDraft(draft,{reset:true});
  if (layout?.rank_devices?.length===draft.deployment.placements.length &&
      members.length===Object.keys(layout.satellite_gpus || {}).length &&
      members.every(sat=>Object.hasOwn(layout.satellite_gpus,sat))) {
    for (const [index, place] of draft.deployment.placements.entries()) {
      const match=/(.*)\/gpu(\d+)$/.exec(layout.rank_devices[index]);
      if (match && members.includes(match[1]) && Number(match[2])<draft.hardware.accelerators_per_satellite) {
        place.satellite_id=match[1];place.local_index=Number(match[2]);
      }
    }
  }
  return draft;
}

export function sceneCellLayout(draft, capabilities, members, {remap=false}={}) {
  const hw=draft.hardware;
  const pool=draft.deployment.worker_pools[0];
  const settings=draft.scheduler[pool.name];
  const v=settings.vllm;
  const profile=capabilities.su_profiles[hw.default_hardware_id];
  const model=capabilities.model_details[draft.model];
  const {tensor_parallel_size:tp,pipeline_parallel_size:pp,data_parallel_size:dp,enable_expert_parallel:ep}=pool.parallelism;
  const cards=hw.accelerators_per_satellite;
  const groups=hw.su_group_sizes;
  const domains=groups.length || Math.ceil(cards/profile.fast_gpu_limit_per_satellite);
  const required = draft.deployment.worker_pools.reduce((total,p) => total +
    p.parallelism.tensor_parallel_size*p.parallelism.pipeline_parallel_size*(p.parallelism.data_parallel_size ?? 1),0);
  if (required>cards*members.length) throw Error(`部署需要 ${required} 张 GPU，本 Cell 有 ${cards*members.length} 张`);
  const mapped=remap?mappingDraft({...draft,hardware:{...hw,satellite_ids:members}},{reset:true}):draft.deployment.placements;
  const ranks=mapped.map(row=>gpuAddress(row.satellite_id,row.local_index));
  if (new Set(ranks).size!==ranks.length) throw Error('Rank 需要映射到不同的 GPU');
  return {
    deployment:{
      ...(draft.compute_mode==='fast_mode' && draft.fast_roofline?{fast_roofline:structuredClone(draft.fast_roofline)}:{}),
      model:model.id,hardware_id:profile.id,gpus_per_satellite:cards,tp,pp,dp,expert_parallel:ep,
      expert_routing_seed:model.experts?draft.expert_routing_seed:null,
      detail_prediction_method:draft.detail_prediction_method,memory_policy_file:draft.memory_policy_file||null,
      fast_glm_dsa_score:draft.fast_glm_dsa_score||null,max_requests:settings.max_requests,
      max_model_len:v.max_model_len,max_batched_tokens:settings.max_batched_tokens,
      kv_block_size_tokens:settings.kv_block_size_tokens,num_kv_blocks:settings.num_kv_blocks,
      prefix_caching:v.enable_prefix_caching,enable_chunked_prefill:v.enable_chunked_prefill,
      long_prefill_token_threshold:v.long_prefill_token_threshold,
    },
    satellite_gpus:Object.fromEntries(members.map(sat=>[sat,cards])),
    rank_devices:ranks,scale_up_groups:[...groups],
    scale_out:domains>1?{bandwidth_gbps:hw.intra_satellite_bandwidth_gbps,latency_ns:hw.intra_satellite_latency_ns}:null,
    collective_policies:structuredClone(draft.collective_policies),
    ...(draft.topology.source === 'file' || draft.deployment.mode==='pd_disagg'
      ? {configuration:structuredClone(draft)} : {}),
  };
}

export function layoutFromExperiment(experiment, cell, members) {
  const devices=experiment.physical.devices;
  const counts=Object.fromEntries(members.map(sat=>[sat,devices.filter(d=>d.satellite===sat&&d.kind==='gpu').length]));
  const owner=members[0];
  const hierarchy=experiment.physical.intra_hierarchies?.[owner];
  const scale=experiment.physical.links.find(link=>link.id.startsWith(owner+'/scale-out-'));
  const algorithms=experiment.communication?.cell_policies?.[cell.id]?.algorithms || {};
  return {
    deployment:{...cell,tp:cell.parallel.tp,pp:cell.parallel.pp,dp:cell.parallel.dp,
      expert_parallel:cell.native_expert_parallel||cell.parallel.ep>1,
      gpus_per_satellite:Math.max(...Object.values(counts)),
      detail_prediction_method:cell.detail_prediction_method??experiment.detail_prediction_method??(experiment.compute_mode==='roofline'?'roofline':undefined),
    },
    satellite_gpus:counts,
    rank_devices:[...cell.placements].sort((a,b)=>a.rank-b.rank).map(row=>row.device),
    scale_up_groups:hierarchy?.su_groups.length>1?hierarchy.su_groups.map(group=>group.length):[],
    scale_out:scale?{bandwidth_gbps:scale.bandwidth_gbps,latency_ns:scale.latency_ns}:null,
    collective_policies:Object.entries(algorithms).flatMap(([scope,operations])=>
      Object.entries(operations).map(([operation,algorithm])=>({scope,operation,algorithm}))),
  };
}

export function preserveOtherCellTopologies(previous, rebuilt, changedCells) {
 if(previous.physical_dataset!==rebuilt.physical_dataset)return rebuilt;
 const retained = new Set(previous.cells.filter(cell=>!changedCells.has(cell.id)).map(cell=>cell.id));
 if(!retained.size) return rebuilt;
 const owners = new Map(previous.physical.cells.filter(cell=>retained.has(cell.id))
  .flatMap(cell=>cell.satellites.map(satellite=>[satellite,cell.id])));
 const oldDevices = new Map(previous.physical.devices.map(device=>[device.id,device]));
 const newDevices = new Map(rebuilt.physical.devices.map(device=>[device.id,device]));
 const sameCell = (source,target)=>owners.has(source)&&owners.get(source)===owners.get(target);
 const localLink = (link,devices)=>sameCell(devices.get(link.source)?.satellite,devices.get(link.target)?.satellite);
 const restoredLinks = previous.physical.links.filter(link=>localLink(link,oldDevices));
 const replacedLinks = rebuilt.physical.links.filter(link=>localLink(link,newDevices));
 const replacedIds = new Set(replacedLinks.map(link=>link.id));
 rebuilt.physical.links = rebuilt.physical.links.filter(link=>!replacedIds.has(link.id)).concat(restoredLinks);
 rebuilt.physical.devices = rebuilt.physical.devices.filter(device=>!owners.has(device.satellite))
  .concat(previous.physical.devices.filter(device=>owners.has(device.satellite)));
 const oldHierarchies=previous.physical.intra_hierarchies??{};
 const newHierarchies=rebuilt.physical.intra_hierarchies??={};
 for(const satellite of owners.keys()) {
  if(oldHierarchies[satellite]) newHierarchies[satellite]=oldHierarchies[satellite];
  else delete newHierarchies[satellite];
 }
 const bindings = rebuilt.physical_link_bindings;
 for(const {id} of replacedLinks) delete bindings[id];
 const restoredPhysicalIds = new Set();
 for(const {id} of restoredLinks) {
  const physical = previous.physical_link_bindings[id];
  if(physical) {bindings[id]=physical;restoredPhysicalIds.add(physical);}
 }
 rebuilt.additional_cell_links.push(...previous.additional_cell_links.filter(link=>sameCell(link.source,link.target)));
 rebuilt.disabled_physical_links.push(...previous.disabled_physical_links.filter(id=>restoredPhysicalIds.has(id)));
 return rebuilt;
}
