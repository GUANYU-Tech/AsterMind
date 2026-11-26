import {mappingDraft, deploymentCapacity} from './deployment-mapping.mjs';
import {fastMode} from './compute-config.mjs';
const states = new WeakMap();
function state(config) {
  if (!states.has(config)) states.set(config,{error:''});
  return states.get(config);
}
export const resourceSyncError = config => state(config).error;
export function synchronizeResources(config, capabilities) {
  const hardware = config.hardware, ids = hardware.satellite_ids;
  const cards = hardware.accelerators_per_satellite;
  const gateway = config.topology?.intra_satellite;
  if (gateway && Number.isSafeInteger(cards) && cards > 0 && gateway.gateway_local_index >= cards) gateway.gateway_local_index = 0;
  hardware.accelerator_overrides = (hardware.accelerator_overrides || []).filter(row => ids.includes(row.satellite_id) && row.local_index >= 0 && row.local_index < cards);
  state(config).error = '';
  try {
    config.deployment.placements = mappingDraft(config);
    if (!fastMode(config) && capabilities.model_shapes?.[config.model]?.tie_word_embeddings
      && config.deployment.worker_pools.some(pool=>pool.parallelism.pipeline_parallel_size>1)) {
      throw Error('当前模型共享输入与输出权重，执行器暂不支持 PP 大于 1。请将 PP 设为 1。');
    }
  } catch (error) { state(config).error = error.message; }
  const {physical,required} = deploymentCapacity(config);
  return {physical,required,idle:Math.max(0,physical-required),error:resourceSyncError(config)};
}

export function clearDependentDrafts(drafts) {
  for (const id of drafts.keys()) {
    const path = JSON.parse(id);
    if (path[0] === 'deployment' && path[1] === 'placements'
      || path[0] === 'topology' && ['inter_satellite','intra_satellite'].includes(path[1])) drafts.delete(id);
  }
}
