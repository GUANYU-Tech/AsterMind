// Navigation follows the experiment, not the Python package/schema hierarchy.
export const stages = ["formation", "network", "workload", "deployment", "runtime", "analysis"];
export const stageNames = ["星座资源", "通信网络", "模型负载", "并行部署", "调度运行", "仿真结果"];
// Presentation policy only: these values remain in imported/exported configs.
export const hiddenFields = ["hardware.accelerator_overrides", "constellation.earth_orientation_mode", "constellation.tle_file", "topology.file", "runtime_inputs.expert_routing"];
export function visibleField(path) {
  const name = path.join(".");
  return !hiddenFields.some(key => name === key || name.startsWith(key + "."));
}
export const groups = {
  formation: [
    ["物理资源", ["hardware.satellite_count", "hardware.accelerators_per_satellite", "hardware.satellite_ids", "hardware.default_hardware_id", "hardware.accelerator_overrides"]],
    ["星座几何", ["topology.source", "topology.file", "constellation"]],
  ],
  network: [
    ["星内网络", ["topology.intra_satellite", "hardware.intra_satellite_bandwidth_gbps", "hardware.intra_satellite_latency_ns"]],
    ["星间连接", ["topology.inter_satellite"]],
    ["光学发射端", ["fso.tx"]], ["光学接收端", ["fso.rx"]], ["光学信道", ["fso.channel"]],
    ["FlowSim", ["flowsim"]],
  ],
  workload: [
    ["模型", ["model"]], ["请求流", ["requests"]],
  ],
  deployment: [
    ["部署模式", ["deployment.mode"]], ["Worker pools", ["deployment.worker_pools"]],
    ["显式部署映射", ["deployment.placements"]], ["跨 pool 通信", ["deployment.routes"]],
  ],
  runtime: [
    ["调度器", ["scheduler"]],
    ["Attention 与 Graph", ["runtime_inputs.attention_kv_layout", "runtime_inputs.attention_kv_distribution", "runtime_inputs.graph_execution", "launch_policy"]],
    ["KV 分层", ["deployment.worker_pools.*.kv_cache", "kv_offload"]],
    ["主机与运行", ["deployment.router", "runtime"]],
  ],
};

export function fieldStage(path) {
  const normalized = path.map(k => typeof k === "number" ? "*" : k).join(".");
  if (normalized === "runtime_inputs.expert_routing" || normalized.startsWith("runtime_inputs.expert_routing.")) return "workload";
  if (/^deployment\.worker_pools\.[^.]+\.kv_cache(?:\.|$)/.test(normalized)) return "runtime";
  if (/^deployment\.worker_pools\.[^.]+\.hardware_id$/.test(normalized)) return "derived";
  let winner = null, length = -1;
  for (const [stage, sections] of Object.entries(groups)) for (const [, paths] of sections) for (const candidate of paths) {
    if ((normalized === candidate || normalized.startsWith(candidate + ".")) && candidate.length > length) {
      winner = stage; length = candidate.length;
    }
  }
  return winner;
}

export function validationTarget(detail) {
  const error = Array.isArray(detail) ? detail[0] : null;
  const path = (error?.loc || []).filter(k => !["body", "configuration"].includes(k));
  if (path.length && fieldStage(path)) return {stage:fieldStage(path), path};
  const message = String(error?.msg || detail || "");
  if (/scheduler|kv_|mtp|launch|timing_mode/i.test(message)) return {stage:"runtime", path:["scheduler"]};
  if (/placement|parallel|worker|pool|rank|inventory coverage|physical card|worker count/i.test(message)) return {stage:"deployment", path:["deployment", "placements"]};
  if (/satellite_count|satellite_ids|accelerators_per_satellite|hardware|constellation|tle|timestamp/i.test(message)) return {stage:"formation", path:["hardware"]};
  if (/scheduler|kv_|mtp|launch|timing_mode/i.test(message)) return {stage:"runtime", path:["scheduler"]};
  if (/model|request|expert/i.test(message)) return {stage:"workload", path:["model"]};
  if (/link|topology|network|fso|bandwidth/i.test(message)) return {stage:"network", path:["topology"]};
  return null;
}

export function configurationFields(schema) {
  const walk = (raw, path) => {
    const spec = raw?.$ref ? schema.$defs[raw.$ref.split("/").at(-1)] : raw;
    if (!spec) return [];
    if (spec.anyOf) return spec.anyOf.filter(s => s.type !== "null").flatMap(s => walk(s, path));
    if (spec.type === "object") return [
      ...Object.entries(spec.properties || {}).flatMap(([key, s]) => walk(s, [...path, key])),
      ...(typeof spec.additionalProperties === "object" ? walk(spec.additionalProperties, [...path, "*"]) : []),
    ];
    if (spec.type === "array") return spec.prefixItems ? spec.prefixItems.flatMap((s, i) => walk(s, [...path, i])) : walk(spec.items, [...path, 0]);
    return [{path, stage:fieldStage(path)}];
  };
  return walk(schema, []);
}

export function hardwareForPool(config, name) {
  const hardware = config.hardware;
  const overrides = new Map((hardware.accelerator_overrides || []).map(c => [`${c.satellite_id}\0${c.local_index}`, c.hardware_id]));
  const places = config.deployment.placements.filter(p => p.worker_pool === name);
  if (!places.length) return "待配置映射";
  if (places.some(p => !hardware.satellite_ids.includes(p.satellite_id) || p.local_index < 0 || p.local_index >= hardware.accelerators_per_satellite)) return "映射超出资源清单";
  const ids = new Set(places.map(p => overrides.get(`${p.satellite_id}\0${p.local_index}`) || hardware.default_hardware_id));
  return ids.size === 1 ? [...ids][0] : "型号混用 · 不支持";
}

export function importResourceHardware(config) {
  // Only migrate old saved browser documents. Never change placements or groups.
  const pools = config.deployment?.worker_pools || [];
  if (!config.hardware?.default_hardware_id) {
    if (!pools.length || pools.some(p => !p.hardware_id)) throw new Error("旧配置缺少硬件型号，请载入有效配置");
    config.hardware.default_hardware_id = pools[0].hardware_id;
    const ids = new Map(pools.map(p => [p.name, p.hardware_id]));
    config.hardware.accelerator_overrides = config.deployment.placements
      .filter(p => ids.has(p.worker_pool) && ids.get(p.worker_pool) !== config.hardware.default_hardware_id)
      .map(p => ({satellite_id:p.satellite_id, local_index:p.local_index, hardware_id:ids.get(p.worker_pool)}));
    pools.forEach(p => delete p.hardware_id);
  }
  return config;
}

export function mappingDraft(config) {
  const rows = [];
  for (const pool of config.deployment.worker_pools) {
    const parallel = pool.parallelism;
    for (let pp = 0; pp < parallel.pipeline_parallel_size; pp++) for (let tp = 0; tp < parallel.tensor_parallel_size; tp++) {
      const coordinate = {dp:0, pp, cp:0, tp};
      const old = config.deployment.placements.find(p => p.worker_pool === pool.name && p.parallel_coordinate.pp === pp && p.parallel_coordinate.tp === tp && (p.parallel_coordinate.dp || 0) === 0 && (p.parallel_coordinate.cp || 0) === 0);
      rows.push(old || {worker_pool:pool.name, parallel_coordinate:coordinate, satellite_id:"", local_index:null});
    }
  }
  return rows;
}
