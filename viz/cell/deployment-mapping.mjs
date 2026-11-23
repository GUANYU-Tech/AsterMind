const positive = value => Number.isSafeInteger(value) && value > 0;
const supported = pool => ['data_parallel_size', 'context_parallel_size', 'sequence_parallel_size']
  .every(key => (pool.parallelism?.[key] ?? 1) === 1);

function executableShape(config, shape, tp, pp) {
  const pools=config.deployment.worker_pools;
  // The legacy residency adapter cannot share tied embeddings across PP ranks.
  if (shape.tie_word_embeddings && pp>1) return false;
  if(tp===1 && pp===1) return true;
  if(config.deployment.mode!=='colocated' || pools.length!==1) return false;
  if(shape.model_type==='qwen3' && !pools[0].parallelism.enable_expert_parallel) {
    if(tp===2 && pp===1) return true;
    if(config.hardware.default_hardware_id!=='nvidia_a800') return false;
    if(tp===2 && pp===2 && shape.torch_dtype==='bfloat16') return true;
    return tp===1 && !shape.tie_word_embeddings && shape.torch_dtype==='bfloat16';
  }
  return false;
}

export function deploymentCapacity(config) {
  const hardware = config.hardware, pools = config.deployment.worker_pools;
  const physical = hardware.satellite_ids.length * hardware.accelerators_per_satellite;
  const required = pools.reduce((sum, pool) => sum + pool.parallelism.tensor_parallel_size * pool.parallelism.pipeline_parallel_size, 0);
  return {physical, required, pools};
}

export function mappingDraft(config, {reset = false} = {}) {
  const {hardware, deployment} = config;
  const pools = deployment.worker_pools;
  if (!positive(hardware.accelerators_per_satellite) || pools.some(pool => !supported(pool)
    || !positive(pool.parallelism.tensor_parallel_size) || !positive(pool.parallelism.pipeline_parallel_size)
    || pool.parallelism.tensor_parallel_size > 1024 || pool.parallelism.pipeline_parallel_size > 1024
    || pool.parallelism.tensor_parallel_size * pool.parallelism.pipeline_parallel_size > 4096)) return deployment.placements;
  const ids = hardware.satellite_ids;
  if (new Set(ids).size !== ids.length || ids.some(id => typeof id !== 'string' || !id.trim())) return deployment.placements;
  const old = new Map();
  const rankKey = row => JSON.stringify([row.worker_pool, row.parallel_coordinate.pp, row.parallel_coordinate.tp]);
  for (const row of reset ? [] : deployment.placements) {
    if (row.parallel_coordinate && (row.parallel_coordinate.dp || 0) === 0 && (row.parallel_coordinate.cp || 0) === 0) old.set(rankKey(row), row);
  }
  const rows = [];
  const occupied = new Set();
  const location = row => JSON.stringify([row.satellite_id, row.local_index]);
  for (const pool of pools) {
    for (let pp = 0; pp < pool.parallelism.pipeline_parallel_size; pp++) for (let tp = 0; tp < pool.parallelism.tensor_parallel_size; tp++) {
      const row = {worker_pool:pool.name, parallel_coordinate:{dp:0, pp, cp:0, tp}, satellite_id:'', local_index:null};
      const existing = old.get(rankKey(row));
      if (existing && ids.includes(existing.satellite_id) && Number.isSafeInteger(existing.local_index)
        && existing.local_index >= 0 && existing.local_index < hardware.accelerators_per_satellite && !occupied.has(location(existing))) {
        Object.assign(row, structuredClone(existing)); occupied.add(location(row));
      }
      rows.push(row);
    }
  }
  // Reserve every valid manual assignment before filling gaps, so a new rank
  // cannot take a card already selected for a later rank.
  let cursor = 0;
  const cardCount = ids.length * hardware.accelerators_per_satellite;
  for (const row of rows) {
    if (row.local_index !== null) continue;
    while (cursor < cardCount) {
      const card = {satellite_id:ids[Math.floor(cursor / hardware.accelerators_per_satellite)], local_index:cursor % hardware.accelerators_per_satellite};
      cursor++;
      if (occupied.has(location(card))) continue;
      Object.assign(row, card); occupied.add(location(row)); break;
    }
  }
  return rows;
}

export function matchParallelismToResources(config, {modelShape, supportedChoices} = {}) {
  const {physical, pools} = deploymentCapacity(config);
  if (!pools.length || !positive(config.hardware.accelerators_per_satellite) || !positive(physical)) throw Error('请先配置有效的卫星和每星卡数');
  const ids = config.hardware.satellite_ids;
  if (ids.length !== config.hardware.satellite_count || new Set(ids).size !== ids.length || ids.some(id => typeof id !== 'string' || !id.trim())) throw Error('请先修正卫星 ID 清单，数量须与卫星数一致且不能重复');
  const names = pools.map(pool => pool.name);
  if (names.some(name => typeof name !== 'string' || !name.trim()) || new Set(names).size !== names.length) throw Error('请先修正重复或空的 Worker pool 名称');
  if (pools.some(pool => !supported(pool))) throw Error('当前自动方案支持 DP / CP / SP 为 1，请先调整对应并行参数');
  const perPool = Math.floor(physical / pools.length);
  if (!perPool) throw Error(`${physical} 张物理卡不足以为 ${pools.length} 个 pool 分配资源`);
  const currentTP = pools[0].parallelism.tensor_parallel_size;
  let tp = pools.every(pool => pool.parallelism.tensor_parallel_size === currentTP) && positive(currentTP) && perPool % currentTP === 0 ? currentTP : 0;
  if (!tp) {
    // Prefer a TP group within one satellite when the previous TP cannot fit.
    let a = perPool, b = config.hardware.accelerators_per_satellite;
    while (b) [a, b] = [b, a % b];
    tp = a;
  }
  const pp = perPool / tp;
  let selected = {tp, pp};
  if (supportedChoices) {
    const available = supportedChoices.filter(choice => choice.tp * choice.pp <= perPool);
    available.sort((a,b) => b.tp*b.pp - a.tp*a.pp || Number(b.tp === currentTP) - Number(a.tp === currentTP));
    if (!available.length) throw Error('当前模型与卡型的并行方案需要更多算力卡');
    selected = available[0];
    for (const pool of pools) pool.parallelism.enable_expert_parallel = selected.expert_parallel;
  } else if (modelShape?.num_hidden_layers && modelShape?.num_attention_heads) {
    const shape = modelShape;
    const gqa = ['qwen3','qwen3_moe','llama'].includes(shape.model_type);
    const maxPP = gqa ? shape.num_hidden_layers : 1;
    const candidates = [];
    for (let tensor = 1; tensor <= Math.min(perPool, shape.num_attention_heads, 1024); tensor++) {
      if ([shape.num_attention_heads,shape.hidden_size,Math.ceil(shape.vocab_size/64)*64].some(value => value && value % tensor)) continue;
      if (gqa && shape.num_key_value_heads && shape.num_key_value_heads % tensor && tensor % shape.num_key_value_heads) continue;
      if ([shape.intermediate_size,...(pools.some(pool => !pool.parallelism.enable_expert_parallel) ? [shape.moe_intermediate_size] : [])].some(value => value && value % tensor)) continue;
      for (let pipeline = 1; pipeline <= Math.min(maxPP,1024); pipeline++) {
        const used = tensor * pipeline;
        if (shape.num_hidden_layers % pipeline || used > perPool || used > 4096 || !executableShape(config,shape,tensor,pipeline)) continue;
        candidates.push({tp:tensor,pp:pipeline});
      }
    }
    const currentPP = pools[0].parallelism.pipeline_parallel_size;
    candidates.sort((a,b) => b.tp*b.pp-a.tp*a.pp
      || Number(b.tp===currentTP&&b.pp===currentPP)-Number(a.tp===currentTP&&a.pp===currentPP)
      || Number(b.tp===config.hardware.accelerators_per_satellite)-Number(a.tp===config.hardware.accelerators_per_satellite)
      || Number(b.tp===currentTP)-Number(a.tp===currentTP) || Math.abs(a.tp-currentTP)-Math.abs(b.tp-currentTP));
    if (!candidates.length) throw Error('当前资源不足以生成符合模型并行约束的方案，请调整资源或模型');
    selected = candidates[0];
  }
  if (selected.tp > 1024 || selected.pp > 1024 || selected.tp * selected.pp > 4096) throw Error('资源数量超出当前编辑器支持的自动并行范围，请手动配置');
  for (const pool of pools) Object.assign(pool.parallelism, {tensor_parallel_size:selected.tp, pipeline_parallel_size:selected.pp});
  config.deployment.placements = mappingDraft(config);
  return selected;
}
