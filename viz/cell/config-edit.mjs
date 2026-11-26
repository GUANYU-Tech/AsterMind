import { defaultValue, resolveSchema } from "./config-form.mjs?v=validation-1";
import { mappingDraft } from "./config-layout.mjs?v=compact-2";
import {newInterSatelliteLink} from './link-mode.mjs';
import {synchronizeManualResources as synchronizeResources} from './resource-sync.mjs';
import {fastMode, modelOptions, normalizeComputeSettings} from './compute-config.mjs';

export const networkInputs = config => Object.fromEntries(["hardware", "topology", "constellation", "fso"].map(key => [key, config[key]]));

export function formationPairs(ids, layout) {
  const pairs = layout === "fully_connected"
    ? ids.flatMap((source, i) => ids.slice(i + 1).map(target => [source, target]))
    : ids.slice(1).map((id,index) => [ids[index],id]);
  if (layout === "ring" && ids.length > 2) pairs.push([ids.at(-1),ids[0]]);
  return pairs;
}

export function syncFormationLinks(config) {
  if (config.topology?.source !== "formation" || !config.constellation?.formation) return;
  const ids = config.hardware.satellite_ids;
  const formation = config.constellation.formation;
  const links = config.topology.inter_satellite.links;
  const key = (a,b) => JSON.stringify([a,b].sort());
  const existing = new Map(links.map(link => [key(link.source,link.target),link]));
  const template = links[0] || newInterSatelliteLink(config, '', '');
  config.topology.inter_satellite.links = formationPairs(ids, formation.layout).map(([source,target]) => ({
    ...structuredClone(existing.get(key(source,target)) || template), source, target,
  }));
}

export function syncDeployment(config, capabilities, path, previous) {
  const pools = config.deployment.worker_pools;
  const names = pools.map(pool => pool.name);
  if (names.some(name => !name?.trim() || ["__proto__", "prototype", "constructor"].includes(name)) || new Set(names).size !== names.length) return;
  if (path?.length === 4 && path.at(-1) === "name" && previous && previous !== names[path[2]]) {
    const next = names[path[2]];
    for (const row of config.deployment.placements) if (row.worker_pool === previous) row.worker_pool = next;
    for (const route of config.deployment.routes) for (const key of ["source_pool", "destination_pool"]) if (route[key] === previous) route[key] = next;
    for (const entries of [config.scheduler, config.kv_offload?.replay_files]) if (entries && Object.hasOwn(entries, previous)) { entries[next] = entries[previous]; delete entries[previous]; }
  }
  // Unsupported parallelisms remain unchanged and are rejected by the backend.
  if (pools.every(pool => ["data_parallel_size", "context_parallel_size", "sequence_parallel_size"].every(k => pool.parallelism[k] === 1)
      && ["pipeline_parallel_size", "tensor_parallel_size"].every(k => Number.isInteger(pool.parallelism[k]) && pool.parallelism[k] > 0 && pool.parallelism[k] <= 1024)
      && pool.parallelism.pipeline_parallel_size * pool.parallelism.tensor_parallel_size <= 4096)) {
    config.deployment.placements = mappingDraft(config);
  }
  const template = Object.values(config.scheduler)[0] || Object.values(capabilities.default_configuration.scheduler)[0];
  config.scheduler = Object.fromEntries(pools.filter(p => !["ffn", "prefill_ffn", "decode_ffn"].includes(p.role)).map(pool => {
    const settings = config.scheduler[pool.name] || structuredClone(template);
    if (settings.vllm) settings.vllm.model = config.model;
    return [pool.name, settings];
  }));
  if (config.kv_offload?.replay_files) {
    for (const name of Object.keys(config.kv_offload.replay_files)) if (!Object.hasOwn(config.scheduler, name)) delete config.kv_offload.replay_files[name];
  }
}

export const moeModels = new Set(["glm-5.2", "glm-5.2-fp8", "kimi-k2.6", "qwen3-30b-a3b"]);
export function defaultModelRouting(model, denseRouting, details) {
  if (!(details ? details.experts : moeModels.has(model))) return structuredClone(denseRouting);
  return {source_kind:"explicit_assumption", source_id:"astermind-web-uniform-topk-seed-0", algorithm:"sha256_weighted_rejection_topk_without_replacement_model_step_v2", root_seed:0, distribution:"maximum_entropy_uniform_topk_v1", hot_expert_count:null, hot_expert_weight:null, replay:null, profile:null, request_stream_sha256:null, vllm_runtime_binding_sha256:null};
}
export function newWebConfiguration(capabilities) {
  const config = normalizeComputeSettings(structuredClone(capabilities.default_configuration), capabilities);
  if (config.constellation) config.constellation.earth_orientation_mode = "stk11_compat";
  config.runtime_inputs.expert_routing = defaultModelRouting(config.model, capabilities.default_configuration.runtime_inputs.expert_routing, capabilities.model_details?.[config.model]);
  syncFormationLinks(config);
  config.deployment.placements = mappingDraft(config);
  return config;
}

export function editorChoices(config, capabilities) {
  const defaults = capabilities.default_configuration;
  const definitions = capabilities.schema.$defs;
  return {
    model: capabilities.models, default_hardware_id: capabilities.hardware, hardware_id: capabilities.hardware,
    satellite_id: config.hardware.satellite_ids, worker_pool: config.deployment.worker_pools.map(p => p.name),
    source_pool: config.deployment.worker_pools.map(p => p.name), destination_pool: config.deployment.worker_pools.map(p => p.name),
    "deployment.mode": ["colocated", "pd_disagg", "af_disagg", "pd_af_disagg"],
    role: ["colocated", "prefill", "decode", "attention", "ffn", "prefill_attention", "prefill_ffn", "decode_attention", "decode_ffn"],
    operation: ["all_reduce", "all_gather", "all_to_all"],
    configured_mode: ["NONE", "FULL", "PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"],
    source_kind: ["not_applicable", "explicit_assumption", "profile_observation", "replay_observation"],
    modelContext: capabilities.model_details?.[config.model]?.max_context,
    visible(path) {
      return !['expert_routing_seed','enable_expert_parallel','data_parallel_size'].includes(path.at(-1))
        || !!capabilities.model_details?.[config.model]?.experts;
    },
    optionLabels: {
      ...Object.fromEntries((capabilities.compute_modes || []).map(mode => [mode.id, mode.name])),
      ...Object.fromEntries(Object.entries(capabilities.model_details || {}).map(([id, model]) => [id, model.name])),
      bf16_timing_proxy: "BF16 计时代理",
    },
    options(path) {
      if (capabilities.compatibility) {
        const full=path.join('.');
        if(full==='model') return modelOptions(config, capabilities);
        if(full==='memory_policy_file') return ['', ...capabilities.memory_policies.map(policy=>policy.id)];
        if(full==='detail_prediction_method') return ['auto', ...capabilities.detail_prediction_methods];
        if(path[0]==='collective_policies' && path.length===3) {
          const row=config.collective_policies[path[1]];
          const scopes=capabilities.collective_algorithms;
          if(path[2]==='scope') return Object.keys(scopes);
          if(path[2]==='operation') return Object.keys(scopes[row.scope] || {});
          if(path[2]==='algorithm') return scopes[row.scope]?.[row.operation]?.choices;
        }
        if(full==='requests.source') return ['synthetic','canonical'];
        if(full==='requests.synthetic.arrival_distribution') return ['poisson','constant'];
        if(full==='deployment.mode') return fastMode(config)?['colocated']:['colocated','pd_disagg'];
        if(path[0]==='scheduler' && path.at(-1)==='framework') return ['vllm'];
        if(full==='runtime_inputs.graph_execution.configured_mode') return ['NONE'];
        if(full==='runtime.timing_mode') return ['device_only'];
        if(full==='launch_policy') return ['analytical_launch'];
      }
      if (path.at(-1) !== "algorithm" || path[0] !== "deployment") return undefined;
      if (!path.includes("collectives")) return ["send_recv"];
      const operation = path.slice(0, -1).reduce((v, k) => v[k], config).operation;
      return operation === "all_to_all" ? ["pairwise"] : operation === "all_gather" ? ["ring", "mesh", "fc2"] : ["ring", "tree", "mesh", "fc2"];
    },
    onObservation(path, data) { bindObservation(config, path, data); },
    seed(path, raw) {
      const parent = path.at(-2);
      if (parent === 'collective_policies') {
        for (const [scope, operations] of Object.entries(capabilities.collective_algorithms)) {
          for (const [operation, rule] of Object.entries(operations)) {
            if (!config.collective_policies.some(row => row.scope===scope && row.operation===operation)) {
              return {scope,operation,algorithm:rule.default};
            }
          }
        }
        throw Error('全部通信范围和操作均已配置');
      }
      if (path.at(-1)==='max_model_len') return 128;
      if (path.at(-1)==='kv_block_size_tokens') return 16;
      if (path.at(-1)==='num_kv_blocks') return 64;
      if (path.at(-1)==='max_batched_tokens') return 256;
      if (parent === "worker_pools") {
        const pool = structuredClone(defaults.deployment.worker_pools[0]);
        let i = config.deployment.worker_pools.length + 1;
        while (config.deployment.worker_pools.some(p => p.name === `pool-${i}`)) i++;
        pool.name = `pool-${i}`; return pool;
      }
      if (path[0] === "scheduler" && path.length === 2) return structuredClone(Object.values(defaults.scheduler)[0]);
      if (parent === "accelerator_overrides") return {satellite_id:config.hardware.satellite_ids[0], local_index:0, hardware_id:config.hardware.default_hardware_id};
      if (parent === "collectives") return {operation:"all_reduce", algorithm:"ring", link:"rdma", topology:"ring", resource_id:"physical-network", backend:"runtime_comm"};
      if (parent === "pipeline_p2p") return {source_stage:0, destination_stage:1, algorithm:"send_recv", link:"infiniband", topology:"point_to_point", resource_id:"pipeline-network", backend:"runtime_comm"};
      if (parent === "routes") return {source_pool:"", destination_pool:"", payload:"activation", algorithm:"send_recv", link:"rdma", topology:"point_to_point", resource_id:"route-network", backend:"runtime_comm"};
      if (parent === "links") return newInterSatelliteLink(config, config.hardware.satellite_ids[0] || '', config.hardware.satellite_ids[1] || '');
      return defaultValue(resolveSchema(raw, definitions), definitions);
    },
  };
}

export function editDependencies(config, path, value, capabilities, previous) {
  const key = path.at(-1), full = path.join(".");
  function ensureSupportedModel() {
    const supported = modelOptions(config, capabilities);
    if (!supported.length || supported.includes(config.model)) return;
    const preferred = capabilities.default_configuration.model;
    config.model = supported.includes(preferred) ? preferred : supported[0];
    editDependencies(config, ['model'], config.model, capabilities);
  }
  if (full === 'compute_mode') {
    const fast = fastMode(config);
    config.detail_prediction_method = fast ? null : 'auto';
    if (fast && config.deployment.mode !== 'colocated') {
      config.deployment.mode = 'colocated';
      editDependencies(config, ['deployment','mode'], 'colocated', capabilities);
    }
    if (!fast) {
      config.memory_policy_file = '';
      config.fast_glm_dsa_score = '';
    }
    ensureSupportedModel();
    synchronizeResources(config, capabilities);
    return true;
  }
  if (full === 'deployment.mode' && ['colocated','pd_disagg'].includes(value)) {
    const roles=value==='colocated'?['colocated']:['prefill','decode'];
    const existing=config.deployment.worker_pools;
    config.deployment.worker_pools=roles.map(role=>{
      const matched=existing.find(pool=>pool.role===role);
      if(matched) return matched;
      const pool=structuredClone(existing[0] || capabilities.default_configuration.deployment.worker_pools[0]);
      pool.name=role==='colocated'?'main':role; pool.role=role; return pool;
    });
    config.deployment.routes=[];
    syncDeployment(config,capabilities,path,previous);
    ensureSupportedModel();
    synchronizeResources(config,capabilities); return true;
  }
  if (full === 'topology.inter_satellite.links') return true;
  if (full === 'topology.intra_satellite.type') {
    if (value !== 'profile_su') config.hardware.su_group_sizes = [];
    return true;
  }
  if (full === 'hardware.su_group_sizes') return true;
  if (path[0] === 'collective_policies' && ['scope','operation'].includes(key)) {
    const row=config.collective_policies[path[1]];
    const options=capabilities.collective_algorithms[row.scope];
    if (!options[row.operation]) row.operation=Object.keys(options)[0];
    const rule=options[row.operation];
    if (!rule.choices.includes(row.algorithm)) row.algorithm=rule.default;
    return true;
  }
  if (full === 'hardware.accelerators_per_satellite' && config.hardware.su_group_sizes?.length &&
      config.hardware.su_group_sizes.reduce((sum,n) => sum+n,0) !== value) config.hardware.su_group_sizes=[];
  if (path[0] === "deployment" && path[1] === "worker_pools" && (path.length === 2 || (path.length === 4 && ["name", "role"].includes(key)) || path[3] === "parallelism")) {
    syncDeployment(config, capabilities, path, previous);
    synchronizeResources(config, capabilities);
    return true;
  }
  if (path.includes("links") && key === "mode") {
    const link = config.topology.inter_satellite.links[path[3]];
    if (value === "fixed" && link.bandwidth_gbps == null) link.bandwidth_gbps = 400;
    return true;
  }
  if (full === "hardware.satellite_count") {
    const ids = config.hardware.satellite_ids.slice(0, value);
    for (let i = 0; ids.length < value; i++) if (!ids.includes(`sat-${i}`)) ids.push(`sat-${i}`);
    config.hardware.satellite_ids = ids;
    syncFormationLinks(config);
  }
  if (full === "hardware.satellite_ids") config.hardware.satellite_count = config.hardware.satellite_ids.length;
  if (["hardware.satellite_count", "hardware.satellite_ids", "hardware.accelerators_per_satellite", "hardware.default_hardware_id"].includes(full)) synchronizeResources(config, capabilities);
  if (["constellation.formation.layout", "hardware.satellite_ids"].includes(full)) syncFormationLinks(config);
  if (full === "hardware.default_hardware_id") {
    const limit = capabilities.su_profiles?.[value]?.fast_gpu_limit_per_satellite;
    if (config.hardware.su_group_sizes?.some(size => size > limit)) config.hardware.su_group_sizes = [];
    config.memory_policy_file = '';
    config.fast_glm_dsa_score = '';
    ensureSupportedModel();
  }
  if (full === "model") {
    const model = capabilities.model_details?.[value];
    for (const settings of Object.values(config.scheduler)) {
      if (!settings.vllm) continue;
      settings.vllm.model = value;
      settings.vllm.dtype = model?.dtype || 'bfloat16';
      settings.vllm.max_model_len = null;
      settings.num_kv_blocks = null;
      settings.kv_block_size_tokens = null;
    }
    if (model && !model.experts) for (const pool of config.deployment.worker_pools) {
      pool.parallelism.enable_expert_parallel = false;
      pool.parallelism.data_parallel_size = 1;
    }
    config.memory_policy_file = '';
    config.fast_glm_dsa_score = '';
    config.runtime_inputs.expert_routing = defaultModelRouting(value, capabilities.default_configuration.runtime_inputs.expert_routing, model);
    synchronizeResources(config, capabilities);
  }
  if (full === "topology.source") {
    config.topology.file = null;
    if (value === "file") config.constellation = null;
    else {
      config.constellation ||= {timestamp:new Date().toISOString(), earth_orientation_mode:"stk11_compat"};
      config.constellation.formation = value === "formation" ? structuredClone(capabilities.default_configuration.constellation.formation) : null;
      config.constellation.tle_file = null;
    }
    syncFormationLinks(config);
  }
  if (full === "requests.source") {
    config.requests.synthetic = value === "synthetic" ? structuredClone(capabilities.default_configuration.requests.synthetic) : null;
    if (value === "synthetic") config.requests.file = null;
    else config.requests.file ||= "";
    config.requests.production = value === "production" ? {
      source_format:"csv", arrival_time_field:"arrival_time_s", input_tokens_field:"input_tokens", output_tokens_field:"output_tokens",
      prefix_hit_tokens_field:"prefix_hit_tokens", request_id_field:"request_id", input_time_unit:"s",
      successful_request_filter:{mode:"upstream_attested", field_name:null, accepted_value:null, provenance:"请填写数据来源及成功请求筛选依据"},
    } : null;
  }
  if (key === "framework") {
    const settings = config.scheduler[path[1]];
    settings.vllm = value === "vllm" ? {
      model:config.model, dtype:"bfloat16", max_model_len:null, gpu_memory_utilization:.9, trust_remote_code:false,
      enable_chunked_prefill:true, enable_prefix_caching:false, async_scheduling:false, dummy_token_id:42,
      mtp_num_speculative_tokens:0, mtp_acceptance_policy:"disabled", mtp_fixed_accepted_tokens:0, mtp_acceptance_source_id:"disabled",
      mtp_acceptance_replay:null, mtp_acceptance_assumption:null, request_stream_sha256:null,
      decode_admission_contract_sha256:null, closed_loop_admission_contract_sha256:null,
    } : null;
    settings.sglang = value === "sglang" ? structuredClone(Object.values(capabilities.default_configuration.scheduler)[0].sglang) : null;
  }
  if (path[0] === 'scheduler' && key === 'enable_chunked_prefill') {
    if (!value) config.scheduler[path[1]].vllm.long_prefill_token_threshold = 0;
    return true;
  }
  if (path.includes("collectives") && ["algorithm", "operation"].includes(key)) {
    const collective = path.slice(0, -1).reduce((v, k) => v[k], config);
    if (key === "operation") collective.algorithm = value === "all_to_all" ? "pairwise" : "ring";
    collective.topology = {ring:"ring", tree:"tree", mesh:"fully_connected", fc2:"fully_connected", pairwise:"fully_connected"}[collective.algorithm] || collective.topology;
  }
  if (full === "runtime_inputs.expert_routing.source_kind") {
    const routing = config.runtime_inputs.expert_routing;
    if (value === "not_applicable") Object.assign(routing, structuredClone(capabilities.default_configuration.runtime_inputs.expert_routing));
    if (value === "explicit_assumption") Object.assign(routing, {
      source_id:"explicit-user-assumption", algorithm:"sha256_weighted_rejection_topk_without_replacement_model_step_v2",
      root_seed:0, distribution:"maximum_entropy_uniform_topk_v1", hot_expert_count:null, hot_expert_weight:null,
      replay:null, profile:null, request_stream_sha256:null, vllm_runtime_binding_sha256:null,
    });
  }
  return ["topology.source", "requests.source", "constellation.formation.layout", "runtime_inputs.expert_routing.source_kind", "hardware.default_hardware_id", "hardware.satellite_count", "hardware.satellite_ids", "hardware.accelerators_per_satellite", "model"].includes(full)
    || (path[0] === "scheduler" && key === "framework") || (path.includes("collectives") && ["operation", "algorithm"].includes(key));
}

export function bindObservation(config, path, data) {
  if (path[0] === "runtime_inputs" && path[1] === "expert_routing") {
    const replay = path.at(-1) === "replay";
    if (replay && (!data.artifact_id || !data.context?.request_stream_sha256)) throw new Error("专家回放缺少 artifact_id 或请求流哈希");
    if (!replay && !data.canonical_model_id) throw new Error("专家观测缺少 canonical_model_id");
    Object.assign(config.runtime_inputs.expert_routing, {
      source_kind:replay ? "replay_observation" : "profile_observation",
      source_id:replay ? data.artifact_id : data.canonical_model_id,
      algorithm:replay ? "expert_histogram_replay_v1" : "sha256_profile_weighted_topk_without_replacement_model_step_v1",
      distribution:replay ? "captured_expert_histogram_v1" : "model_profile_layer_mean_v1",
      root_seed:null, hot_expert_count:null, hot_expert_weight:null,
      request_stream_sha256:replay ? data.context.request_stream_sha256 : null,
      vllm_runtime_binding_sha256:null, profile:null, replay:null,
    });
  }
  if (path.at(-1) === "mtp_acceptance_replay") {
    const context = data.scheduler_context;
    if (!data.artifact_id || !context?.request_stream_sha256) throw new Error("MTP 回放缺少 artifact_id 或 scheduler_context");
    Object.assign(config.scheduler[path[1]].vllm, {
      mtp_acceptance_policy:"replay", mtp_fixed_accepted_tokens:0, mtp_acceptance_source_id:data.artifact_id,
      mtp_acceptance_assumption:null, mtp_num_speculative_tokens:context.num_speculative_tokens,
      request_stream_sha256:context.ingress_request_stream_sha256 || context.request_stream_sha256,
      decode_admission_contract_sha256:context.decode_admission_contract_sha256 || null,
      closed_loop_admission_contract_sha256:null,
    });
  }
}

export function bindRequestIdentity(config, identity) {
  let count = 0;
  for (const [name, settings] of Object.entries(config.scheduler)) {
    const options = settings[settings.framework], assumption = options?.mtp_acceptance_assumption;
    if (!assumption && !(settings.framework === "vllm" && options?.mtp_acceptance_policy === "fixed_count")) continue;
    const pool = config.deployment.worker_pools.find(p => p.name === name);
    const decode = ["decode", "decode_attention"].includes(pool?.role);
    const closed = !!config.requests.closed_loop_client_count && !decode;
    if (assumption && ((decode && !identity.decode_admission_contract_sha256) || (closed && !identity.closed_loop_admission_contract_sha256))) throw new Error("MTP admission 绑定需要至少一个输出超过 1 Token 的请求");
    if (assumption) {
      assumption.request_stream_sha256 = identity.request_stream_sha256;
      options.mtp_num_speculative_tokens = assumption.proposal_depth;
      options.mtp_acceptance_source_id = assumption.source_id;
    }
    if (settings.framework === "vllm") {
      Object.assign(options, {
        request_stream_sha256:identity.request_stream_sha256,
        decode_admission_contract_sha256:assumption && decode ? identity.decode_admission_contract_sha256 : null,
        closed_loop_admission_contract_sha256:assumption && closed ? identity.closed_loop_admission_contract_sha256 : null,
      });
      if (assumption) Object.assign(options, {mtp_acceptance_policy:"maximum_entropy", mtp_fixed_accepted_tokens:0, mtp_acceptance_replay:null});
    }
    count++;
  }
  return count;
}
