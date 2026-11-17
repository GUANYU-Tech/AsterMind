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

const migrationNotes = new WeakMap();
export function migrateOpticalConfig(config, defaults) {
  const previous = config.fso;
  const legacy = previous && ('beam_waist_m' in (previous.tx || {}) || 'baud_rate_options_ghz' in (previous.tx || {})
    || 'wdm_loss_db' in (previous.channel || {}) || 'pointing_error_rad' in (previous.channel || {}));
  if (!legacy) return '';
  const next = structuredClone(previous);
  next.tx ||= {}; next.rx ||= {}; next.channel ||= {};
  const options = next.tx?.baud_rate_options_ghz;
  const convertedRate = !Object.hasOwn(next.tx, 'baud_rate_ghz');
  if (convertedRate) {
    const rates = Array.isArray(options) ? options.filter(value => typeof value === 'number' && Number.isFinite(value) && value > 0) : [];
    next.tx.baud_rate_ghz = rates.length ? Math.max(...rates) : defaults.tx.baud_rate_ghz;
  }
  next.rx.preamp_gain_db ??= defaults.rx.preamp_gain_db;
  delete next.tx.beam_waist_m; delete next.tx.baud_rate_options_ghz;
  delete next.channel.wdm_loss_db; delete next.channel.pointing_error_rad;
  config.fso = next;
  const note = `旧光学参数已转换：波特率固定为 ${next.tx.baud_rate_ghz} GBaud（${convertedRate ? '取原候选列表最大值，可调整' : '保留原固定波特率'}）。当前 ISLSim 不接收束腰、附加 WDM 损耗和指向误差参数，这些旧字段已移除；请核对光学参数。`;
  migrationNotes.set(config, note);
  return note;
}
export const opticalMigrationNote = config => migrationNotes.get(config) || '';
