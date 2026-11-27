import {field} from '/assets/studio_common.js';

const radioNames={
  starlink_v2_mini_2024:'Starlink V2 Mini（2024 年）',
  starlink_v3_design_160_20261021:'Starlink V3 设计规格（上行 160 Gbps）',
  starlink_v3_progress_2025:'Starlink V3 报告下限（2025 年）',
};
const radioName=(id,profile)=>radioNames[id] || profile.generation;

export function applyGeneration(shell, id, profiles) {
  if (!id) {
    for(const key of ['radio_profile','optical_terminal_bandwidth_bps','terminal_assumptions',
      'radio_capacity_bps','capacity_assumptions'])delete shell[key];
    return;
  }
  const profile=profiles[id];
  shell.radio_profile=id;
  shell.isl_terminals=profile.terminals.optical_count;
  shell.sgl_terminals=profile.terminals.rf_feeder_antennas;
  for(const key of ['optical_terminal_bandwidth_bps','terminal_assumptions','radio_capacity_bps','capacity_assumptions'])shell[key]=structuredClone(profile.simulation_defaults[key]);
}

export function fillMissingGeneration(shell, profiles){
 const profile=profiles[shell.radio_profile];if(!profile)return;
 const defaults=profile.simulation_defaults;
 for(const key of ['optical_terminal_bandwidth_bps','terminal_assumptions'])shell[key]??=structuredClone(defaults[key]);
 shell.radio_capacity_bps={...defaults.radio_capacity_bps,...shell.radio_capacity_bps};
 shell.capacity_assumptions=[...new Set([...defaults.capacity_assumptions,...shell.capacity_assumptions||[]])];
}

export function renderRadio(fields, card, shell, profiles, changed) {
  fillMissingGeneration(shell,profiles);
  const panel=document.createElement('section');panel.className='radio-panel';
  card.append(panel);card=panel;fields=document.createElement('div');fields.className='fields radio-profile-fields';panel.append(fields);
  const select=field(fields,shell,'radio_profile','通信卫星硬件预设','text',[
    ['', '自定义硬件'], ...Object.entries(profiles).map(([id,p])=>[id,radioName(id,p)])]);
  select.addEventListener('change',()=>{applyGeneration(shell,select.value,profiles);changed()});
  const profile=profiles[shell.radio_profile];
  const advanced=document.createElement('details'), title=document.createElement('summary');
  title.textContent=profile?'高级硬件参数（预设只读）':'高级硬件参数（手动设置）';
  advanced.append(title);panel.append(advanced);
  const settings=document.createElement('fieldset');
  settings.disabled=Boolean(profile);advanced.append(settings);card=settings;
  fields=document.createElement('div');fields.className='fields';card.append(fields);
  for(const [key,caption] of [['isl_terminals','ISL 终端数'],['sgl_terminals','每星地面关联总上限']])field(fields,shell,key,caption);
}

export function renderRadioCapacity(card,shell,profiles) {
  const limits={...profiles[shell.radio_profile]?.capacities_bps,...shell.radio_capacity_bps};
  const rows=[
    ['user_uplink_bps','用户上行'],['user_downlink_bps','用户下行'],
    ['feeder_uplink_bps','网关上行'],['feeder_downlink_bps','网关下行'],
    ['feeder_combined_bps','网关双向共享'],
  ].filter(([key])=>Number.isFinite(limits[key]));
  if (!rows.length) return;
  const details=document.createElement('details'),summary=document.createElement('summary');
  const caption=document.createElement('span');caption.textContent='整星共享 RF 容量';
  summary.append(shell.shell_id+' · ',caption);details.append(summary);
  const values=document.createElement('dl');
  for (const [key,label] of rows) {
    const term=document.createElement('dt'),rate=document.createElement('dd');
    term.textContent=label;rate.textContent=limits[key]/1e9+' Gbps';values.append(term,rate);
  }
  details.append(values);card.append(details);
}
