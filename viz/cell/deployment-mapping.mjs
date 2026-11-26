const positive = value => Number.isSafeInteger(value) && value > 0;
import {fastMode} from './compute-config.mjs';

const replicas = pool => pool.parallelism.data_parallel_size ?? 1;
const requiredCards = pool => pool.parallelism.tensor_parallel_size * pool.parallelism.pipeline_parallel_size * replicas(pool);
const supported = (config, pool) => ['context_parallel_size', 'sequence_parallel_size']
  .every(key => (pool.parallelism?.[key] ?? 1) === 1)
  && (replicas(pool) === 1 || (fastMode(config) || config.compute_mode === 'detail_mode') && positive(replicas(pool)));

export function deploymentCapacity(config) {
  const hardware = config.hardware, pools = config.deployment.worker_pools;
  const physical = hardware.satellite_ids.length * hardware.accelerators_per_satellite;
  const required = pools.reduce((sum, pool) => sum + requiredCards(pool), 0);
  return {physical, required, pools};
}

export function mappingDraft(config, {reset = false} = {}) {
  const {hardware, deployment} = config;
  const pools = deployment.worker_pools;
  if (!positive(hardware.accelerators_per_satellite) || pools.some(pool => !supported(config, pool)
    || !positive(pool.parallelism.tensor_parallel_size) || !positive(pool.parallelism.pipeline_parallel_size)
    || pool.parallelism.tensor_parallel_size > 1024 || pool.parallelism.pipeline_parallel_size > 1024
    || requiredCards(pool) > 4096)) return deployment.placements;
  const ids = hardware.satellite_ids;
  if (new Set(ids).size !== ids.length || ids.some(id => typeof id !== 'string' || !id.trim())) return deployment.placements;
  const old = new Map();
  const rankKey = row => JSON.stringify([row.worker_pool, row.parallel_coordinate.dp ?? 0, row.parallel_coordinate.pp, row.parallel_coordinate.tp]);
  for (const row of reset ? [] : deployment.placements) {
    if (row.parallel_coordinate && (row.parallel_coordinate.cp || 0) === 0) old.set(rankKey(row), row);
  }
  const rows = [];
  const occupied = new Set();
  const location = row => JSON.stringify([row.satellite_id, row.local_index]);
  for (const pool of pools) {
    for (let dp = 0; dp < replicas(pool); dp++) {
      for (let pp = 0; pp < pool.parallelism.pipeline_parallel_size; pp++) for (let tp = 0; tp < pool.parallelism.tensor_parallel_size; tp++) {
        const row = {worker_pool:pool.name, parallel_coordinate:{dp, pp, cp:0, tp}, satellite_id:'', local_index:null};
        const existing = old.get(rankKey(row));
        if (existing && ids.includes(existing.satellite_id) && Number.isSafeInteger(existing.local_index)
          && existing.local_index >= 0 && existing.local_index < hardware.accelerators_per_satellite && !occupied.has(location(existing))) {
          Object.assign(row, structuredClone(existing)); occupied.add(location(row));
        }
        rows.push(row);
      }
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
