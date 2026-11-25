import { element } from './results.mjs?v=compact-2';
import {renderConfigForm} from './config-form.mjs';
import { optionLabels } from './config-labels.mjs?v=compact-2';

const number = value => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('zh-CN', {maximumFractionDigits: 3}) : '未提供';
const unit = (value, suffix, divisor = 1) => typeof value === 'number' && Number.isFinite(value)
  ? `${number(value / divisor)} ${suffix}` : '未提供';
function bytes(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '未提供';
  const scale = value >= 2 ** 30 ? [2 ** 30, 'GiB'] : value >= 2 ** 20 ? [2 ** 20, 'MiB'] : [2 ** 10, 'KiB'];
  return unit(value, scale[1], scale[0]);
}
const componentNames = {
  sm:'计算阵列', aiv_core:'向量核心', subcore:'子核心', tensor_core:'张量核心',
  fma_pipe:'乘加单元', sfu_pipe:'特殊函数单元', alu_pipe:'整数运算单元',
  shared_mem:'共享内存', l1_cache:'L1 缓存', l2:'L2 缓存', dram:'显存', xbar:'片上互联',
  l0a_buffer:'L0A 缓冲', l0b_buffer:'L0B 缓冲', l0c_buffer:'L0C 缓冲', bt_buffer:'BT 缓冲',
  unified_buffer:'统一缓冲', vector_pipe:'向量运算单元', reduction_service:'归约单元',
  vscalar_pipe:'标量运算单元', mte2_pipe:'数据搬入单元', mte3_pipe:'数据搬出单元',
  warp_dispatch:'线程束分发', warp_scheduler:'线程束调度', integer_dependency:'整数依赖单元',
  serial_dependency:'串行依赖单元', block_reduction_dependency:'块归约依赖单元',
  silu_scalar_dependency:'SiLU 标量依赖单元', staged_ldgsts_l2_read:'分阶段 L2 读取',
  staged_ldgsts_k32_depth6_l2_read:'K32 深度 6 L2 读取',
};

function definitionList(rows, className) {
  const list = element('dl', undefined, className);
  for (const [label, value] of rows) {
    const row = element('div'); row.append(element('dt', label), element('dd', value)); list.append(row);
  }
  return list;
}

function componentTree(components) {
  const list = element('ul', undefined, 'hardware-components');
  for (const component of components) {
    const row = element('li');
    const heading = element('div', undefined, 'hardware-component-name');
    heading.append(element('span', componentNames[component.name] || component.name), element('b', `× ${number(component.count)}`));
    row.append(heading);
    const metrics = [];
    if (component.size_bytes != null) metrics.push(bytes(component.size_bytes));
    if (component.throughput_per_clk != null) metrics.push(`吞吐 ${number(component.throughput_per_clk)} / 周期`);
    for (const variant of component.variants || []) metrics.push(`${variant.name.toUpperCase()} ${number(variant.throughput_per_clk)} / 周期`);
    if (metrics.length) row.append(element('small', metrics.join(' ')));
    if (component.children?.length) row.append(componentTree(component.children));
    list.append(row);
  }
  return list;
}

export function rooflinePanel(config, capabilities, changed, choices) {
  const peaks = capabilities.su_profiles[config.hardware.default_hardware_id].roofline_peaks;
  const box=element('fieldset',undefined,'basic-fields roofline-settings');
  box.append(element('legend','Roofline 参数配置'));
  for(const [key,caption,peak,unit] of [
    ['tensor_percent','有效算力（BF16）',peaks.bf16_tflops,'TFLOP/s'],
    ['hbm_percent','有效 HBM 带宽',peaks.hbm_tb_s,'TB/s'],
  ]){
    const row=element('div',undefined,'roofline-row'), form=element('div');
    const label=element('span',caption), output=element('output');
    renderConfigForm(form,config,['fast_roofline.'+key],capabilities.schema,changed,choices);
    const input = form.querySelector('input');
    output.htmlFor.add(input.id);
    const update = () => {
      output.textContent = input.validity.valid ? number(peak * input.valueAsNumber / 100) + ' ' + unit : '—';
    };
    input.addEventListener('input', update);
    update();
    const result=element('div',undefined,'roofline-effective');result.append(label,output);
    row.append(form,result);box.append(row);
  }
  return box;
}

export function renderHardwareProfiles(container, ids, details = {}) {
  const opened = new Set([...container.querySelectorAll('details[open]')].map(node => node.dataset.hardwareId));
  container.replaceChildren();
  for (const id of ids) {
    const data = details[id], card = element('article', undefined, 'hardware-card');
    const header = element('header'); header.append(element('h2', optionLabels[id] || id), element('span', '单卡参数'));
    card.append(header);
    if (!data) { card.append(element('p', '暂无硬件档案', 'muted')); container.append(card); continue; }
    const components = data.components || [], clocks = data.clock_domains || {}, links = data.interconnect || {};
    const compute = components.find(c => c.name === 'sm');
    const memory = components.find(c => c.role === 'dram');
    const cache = components.find(c => c.role === 'l2_cache');
    card.append(definitionList([
      ['显存容量', bytes(memory?.size_bytes)],
      ['计算单元', compute ? `${number(compute.count)} ${compute.clock_domain === 'aic' ? 'AI Core' : 'SM'}` : '未提供'],
      ['核心频率', unit(clocks[compute?.clock_domain || 'sm'], 'GHz', 1e9)],
      ['L2 缓存', bytes(cache?.size_bytes)],
    ], 'hardware-metrics'));
    card.append(definitionList([
      ['显存频率', unit(clocks[memory?.clock_domain || 'dram'], 'GHz', 1e9)],
      ['卡间单向互联带宽', unit(links.nvlink_unidir_bw_GB_s, 'GB/s')],
      ['PCIe 带宽', unit(links.pcie_bw_GB_s, 'GB/s')],
      ['InfiniBand 单向带宽', unit(links.infiniband_unidir_bw_GB_s, 'GB/s')],
    ], 'hardware-specs'));
    const architecture = element('details', undefined, 'hardware-architecture');
    architecture.dataset.hardwareId = id; architecture.open = opened.has(id);
    architecture.append(element('summary', '架构参数'), componentTree(components));
    card.append(architecture); container.append(card);
  }
}
