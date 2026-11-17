// Inter-cell links use an independent budget. Start from the configured compute
// terminal rather than inventing a new range, rate or optical hardware model.
export function ensureInterCellProfile(network) {
  if (!network.inter_cell_links || network.link_profiles?.inter_cell) return false;
  const source = network.link_profiles?.cell_isl;
  if (!source || typeof source !== 'object' || Array.isArray(source)) return false;
  network.link_profiles.inter_cell = structuredClone(source);
  return true;
}

export const interCellProfileNote = 'cell 间直连使用独立链路参数。首次启用且缺少参数时，从 cell 内部链路复制初值（包括最大距离）；请按实际跨 cell 距离与终端能力调整。关闭直连会保留已编辑参数。';
export const missingInterCellProfile = '缺少 cell 间直连参数，且没有可用于初始化的 cell 内部链路参数；请在完整物理模型 JSON 中补充 network.link_profiles.inter_cell，或关闭 cell 间直接连接。';
