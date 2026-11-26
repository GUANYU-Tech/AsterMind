export const fastMode = config => config?.compute_mode === 'fast_mode';

export function modelOptions(config, capabilities) {
  const hardware = config.hardware.default_hardware_id;
  if (fastMode(config)) {
    return Object.keys(capabilities.model_details).filter(
      id => capabilities.model_details[id].fast_hardware.includes(hardware)
    );
  }
  if (config.deployment.mode === 'pd_disagg') {
    return hardware === 'nvidia_h200' ? ['kimi-k2.6'] : [];
  }
  return capabilities.model_details ? Object.keys(capabilities.model_details).filter(id => !id.endsWith('-dspark') && id !== 'glm-5.2-fp8')
    : Object.keys(capabilities.models_by_hardware?.[hardware] || {});
}

export function detailChoices(config, capabilities) {
  if (config.deployment.mode === 'pd_disagg') {
    return config.hardware.default_hardware_id === 'nvidia_h200' && config.model === 'kimi-k2.6'
      ? [{tp:8, pp:1, expert_parallel:true}] : [];
  }
  return capabilities.models_by_hardware?.[config.hardware.default_hardware_id]?.[config.model];
}

export function normalizeComputeSettings(config, capabilities) {
  if (config.runtime) delete config.runtime.timeout_seconds;
  if (['midlayer', 'roofline'].includes(config.compute_mode)) {
    config.detail_prediction_method ??= config.compute_mode;
    config.compute_mode = 'detail_mode';
  }
  if (config.compute_mode === 'batch_roofline') config.compute_mode = 'fast_mode';
  config.compute_mode ??= 'detail_mode';
  if (capabilities.fast_roofline_defaults) config.fast_roofline ??= structuredClone(capabilities.fast_roofline_defaults);
  if (config.compute_mode === 'detail_mode') config.detail_prediction_method ??= 'auto';
  const policy = capabilities.memory_policies?.find(row => row.file === config.memory_policy_file);
  config.memory_policy_file = policy?.id ?? config.memory_policy_file ?? '';
  config.fast_glm_dsa_score ??= '';
  config.expert_routing_seed ??= config.runtime_inputs?.expert_routing?.root_seed ?? 0;
  config.hardware.su_group_sizes ??= [];
  config.collective_policies ??= [];
  config.ep_strategy ??= 'auto';
  for (const settings of Object.values(config.scheduler || {})) {
    settings.max_batched_tokens ??= null;
    if (settings.vllm) settings.vllm.long_prefill_token_threshold ??= 0;
  }
  return config;
}
