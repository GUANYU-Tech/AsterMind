const preferences = new WeakMap();
const valid = mode => mode === 'fixed' || mode === 'islsim';
export function linkMode(config) {
  const modes = new Set(config.topology.inter_satellite.links.map(link => link.mode || 'islsim'));
  if (!modes.size) return preferences.get(config) || 'islsim';
  const [mode] = modes;
  if (modes.size === 1 && valid(mode)) { preferences.set(config, mode); return mode; }
  return null;
}
export function rememberEmptyLinkMode(config, mode) {
  if (valid(mode)) preferences.set(config, mode);
}
export function setLinkMode(config, mode) {
  if (!valid(mode)) throw Error('请选择有效的星间链路模式');
  preferences.set(config, mode);
  for (const link of config.topology.inter_satellite.links) {
    link.mode = mode;
    if (mode === 'fixed' && link.bandwidth_gbps == null) link.bandwidth_gbps = 400;
  }
}
export function newInterSatelliteLink(config, source, target) {
  const mode = linkMode(config);
  if (!mode) throw Error('请先选择统一的星间链路模式');
  return {source, target, mode, bandwidth_gbps:mode === 'fixed' ? 400 : null};
}

export function applyOpticalDefaults(config, defaults) {
  if (!config.fso || typeof config.fso !== 'object' || Array.isArray(config.fso)) return;
  for (const [section, values] of Object.entries(defaults)) {
    const overrides=config.fso[section];
    if (overrides === undefined || (overrides && typeof overrides === 'object' && !Array.isArray(overrides))) {
      config.fso[section]={...structuredClone(values), ...overrides};
    }
  }
}
