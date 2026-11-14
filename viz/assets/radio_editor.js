import {field} from '/assets/studio_common.js';
import {nativeParameters,renderISLSim} from '/assets/islsim_editor.js';

export function applyGeneration(shell, id, profiles) {
  if (!id) {delete shell.radio_profile; return}
  const profile=profiles[id];
  shell.radio_profile=id;
  shell.isl_terminals=profile.terminals.optical_count;
  shell.sgl_terminals=profile.terminals.rf_feeder_antennas;
  for(const key of ['optical_terminal_bandwidth_bps','terminal_assumptions','radio_capacity_bps','capacity_assumptions','hardware_link_profiles'])shell[key]=structuredClone(profile.simulation_defaults[key]);
}

export function fillMissingGeneration(shell, profiles){
 const profile=profiles[shell.radio_profile];if(!profile)return;
 const defaults=profile.simulation_defaults;
 for(const key of ['optical_terminal_bandwidth_bps','terminal_assumptions','hardware_link_profiles'])shell[key]??=structuredClone(defaults[key]);
 shell.radio_capacity_bps={...defaults.radio_capacity_bps,...shell.radio_capacity_bps};
 shell.capacity_assumptions=[...new Set([...defaults.capacity_assumptions,...shell.capacity_assumptions||[]])];
}

function scaledField(parent,obj,key,label,scale=1){const wrap=document.createElement('label');wrap.textContent=label;const input=document.createElement('input');input.type='number';input.step='any';input.setAttribute('aria-label',label);input.value=obj[key]/scale;input.oninput=()=>{obj[key]=Number(input.value)*scale;document.dispatchEvent(new Event('modelchange'))};wrap.append(input);parent.append(wrap)}

function renderLinkDefaults(card,shell,changed){
 const descriptions={user_uplink:'用户→卫星（上传）',user_downlink:'卫星→用户（下载）',feeder_uplink:'网关→卫星（回传上行）',feeder_downlink:'卫星→网关（回传下行）',communication_isl:'通信卫星↔通信卫星（ISL）'};
 const note=document.createElement('p');note.textContent='四个星地方向分别使用独立预算。网关上行不等于用户上传。整星总容量由多条链路共享，不能作为每条 GSL 的速率。';card.append(note);
 const overview=document.createElement('table');overview.style.cssText='width:100%;text-align:left;font-size:13px;border-spacing:8px';
 const head=document.createElement('tr');for(const label of ['已预填链路','中心频率 / 波长','载波宽度','发射功率']){const th=document.createElement('th');th.textContent=label;head.append(th)}overview.append(head);
 for(const [kind,p]of Object.entries(shell.hardware_link_profiles)){const row=document.createElement('tr'),native=p.capacity_model==='islsim'?nativeParameters(p):null;for(const text of [descriptions[kind],kind==='communication_isl'?(native?.tx.wavelength_m??p.wavelength_m)*1e9+' nm':p.frequency_hz/1e9+' GHz',kind==='communication_isl'?(p.capacity_model==='islsim'?'ISLSim':p.capacity_model):p.channel_bandwidth_hz/1e6+' MHz',(native?.tx.tx_power_w??p.transmit_power_w)+' W']){const td=document.createElement('td');td.textContent=text;row.append(td)}overview.append(row)}card.append(overview);
 const caveat=document.createElement('p');caveat.textContent='上表为可运行的单载波工程配置，实际带宽还随距离计算；载波宽度、功率和孔径等未公开项不标为厂商参数。';card.append(caveat);
 const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='查看 / 修改已预填的四方向 RF 与星间光链路参数';details.append(summary);card.append(details);
 for(const [kind,p]of Object.entries(shell.hardware_link_profiles)){
  const section=document.createElement('section'),h=document.createElement('h4');h.textContent=descriptions[kind];section.append(h);details.append(section);
  const desc=document.createElement('p');desc.textContent=kind==='communication_isl'?'光学预算 · 每方向速率同时受模型、链路与两端光口上限约束（单双向工程假设）':`${p.frequency_hz/1e9} GHz · ${p.channel_bandwidth_hz/1e6} MHz · ${p.transmit_power_w} W · 以地面为参照标注方向`;section.append(desc);
  const fields=document.createElement('div');fields.className='fields';section.append(fields);
  if(kind==='communication_isl'){
   if(renderISLSim(fields,p,changed)){
    scaledField(fields,p,'bandwidth_bps',descriptions[kind]+' · 每方向上限 Gbps',1e9);
    continue;
   }
  }
  for(const [key,label,scale]of [['bandwidth_bps','单链路每方向预算上限 Gbps',1e9],...(kind==='communication_isl'?[['wavelength_m','波长 nm',1e-9],['required_photons_per_bit','接收光子 / bit',1]]:[['frequency_hz','中心频率 GHz',1e9],['channel_bandwidth_hz','单载波带宽 MHz',1e6],['noise_temperature_k','噪声温度 K',1]]),['transmit_power_w','发射功率 W',1],['transmit_aperture_m','发射有效孔径 m',1],['receive_aperture_m','接收有效孔径 m',1],['system_loss_db','系统损耗 dB',1],['implementation_efficiency','实现效率',1]])scaledField(fields,p,key,descriptions[kind]+' · '+label,scale);
  const assumptions=document.createElement('p');assumptions.textContent=p.assumptions.join(' ');section.append(assumptions);
 }
 const geometry=document.createElement('p');geometry.textContent='星地建链时间、限距、太阳避让仍由“连接与链路预算”的通用星地几何规则控制；这里四个方向配置射频速率预算。';details.append(geometry);
}

export function renderRadio(fields, card, shell, profiles, changed) {
  fillMissingGeneration(shell,profiles);
  const panel=document.createElement('section');panel.className='card';panel.style.gridColumn='1 / -1';
  fields.append(panel);card=panel;fields=document.createElement('div');fields.className='fields';panel.append(fields);
  const select=field(fields,shell,'radio_profile','通信卫星硬件代际','text',[
    ['', '自定义工程模型'], ...Object.entries(profiles).map(([id,p])=>[id,p.generation+(id==='starlink_v3_design_160_20261021'?' · 规格页 160 Gbps UL':'')])]);
  select.addEventListener('change',()=>{applyGeneration(shell,select.value,profiles);changed()});
  const profile=profiles[shell.radio_profile];
  if (!profile) return;
  const reset=document.createElement('button');reset.textContent='重填本代际的全部通信参数';reset.onclick=()=>{applyGeneration(shell,shell.radio_profile,profiles);changed()};card.append(reset);
  const info=document.createElement('p'), t=profile.terminals;
  info.textContent=`已公布设备：${t.optical_count} 个光口，标称 ${t.optical_advertised_bps/1e9} Gbps/口（单双向未明确）；${t.rf_feeder_antennas} 副 RF 网关天线；用户波束下行 ${t.user_downlink_beams} / 上行 ${t.user_uplink_beams}。星座构型保持独立。`;
  card.append(info);
  const topology=document.createElement('p'); topology.className='danger';
  topology.textContent=`${shell.topology==='grid'?'Grid 有 4 个候选通信邻居':'Honeycomb 有 3 个候选通信邻居'}，计算 cell 接入还要争用同一光口池。实际建链总数受 ${shell.isl_terminals} 个光口限制，不能假定所有候选边同时连通。`;
  card.append(topology);
  const capLabel=document.createElement('label');capLabel.textContent='每光口每方向工程上限 Gbps';
  const cap=document.createElement('input');cap.type='number';cap.step='any';cap.min='0';cap.setAttribute('aria-label',capLabel.textContent);
  cap.value=shell.optical_terminal_bandwidth_bps==null?'':shell.optical_terminal_bandwidth_bps/1e9;
  cap.oninput=()=>{
    const previous=shell.optical_terminal_bandwidth_bps,optical=shell.hardware_link_profiles?.communication_isl;
    if(cap.value==='')delete shell.optical_terminal_bandwidth_bps;
    else shell.optical_terminal_bandwidth_bps=Number(cap.value)*1e9;
    if(optical?.bandwidth_bps===previous&&cap.value!==''){
      optical.bandwidth_bps=shell.optical_terminal_bandwidth_bps;
      const input=card.querySelector('[aria-label="通信卫星↔通信卫星（ISL） · 每方向上限 Gbps"]');if(input)input.value=cap.value;
    }
    if(optical?.islsim_config?.rx.terminal_cap_gbps===previous/1e9&&cap.value!==''){
      optical.islsim_config.rx.terminal_cap_gbps=shell.optical_terminal_bandwidth_bps/1e9;
      const input=card.querySelector('[aria-label="ISLSim · 终端总速率上限 Gbps（留空为不限）"]');if(input)input.value=cap.value;
    }
    document.dispatchEvent(new Event('modelchange'));
  };
  cap.onchange=cap.oninput;
  capLabel.append(cap);fields.append(capLabel);
  const published=document.createElement('p');
  published.textContent='公开的整星容量：'+Object.entries(profile.capacities_bps).map(([k,v])=>`${k} = ${v/1e9} Gbps`).join('；')+'。其余已预填推算 / 场景假设，展开下方可查看来源口径。';
  card.append(published);
  const details=document.createElement('details'), summary=document.createElement('summary');
  summary.textContent='已预填的整星 RF 容量与估算说明';details.append(summary);card.append(details);
  details.open=!('feeder_combined_bps' in profile.capacities_bps);
  const overrides=document.createElement('div');overrides.className='fields';details.append(overrides);
  shell.radio_capacity_bps??={}; shell.capacity_assumptions??=[]; shell.terminal_assumptions??=[];
  for (const [key,label] of [['user_uplink_bps','整星用户上行覆盖值 Gbps'],['user_downlink_bps','整星用户下行覆盖值 Gbps'],
    ['feeder_uplink_bps','整星网关上行假设 Gbps'],['feeder_downlink_bps','整星网关下行假设 Gbps'],['feeder_combined_bps','整星网关双向共享假设 Gbps']]) {
    const wrap=document.createElement('label');wrap.textContent=label;
    const input=document.createElement('input');input.type='number';input.step='any';input.min='0';input.setAttribute('aria-label',label);
    input.value=key in shell.radio_capacity_bps?shell.radio_capacity_bps[key]/1e9:key in profile.capacities_bps?profile.capacities_bps[key]/1e9:'';
    input.placeholder=key in profile.capacities_bps?`默认 ${profile.capacities_bps[key]/1e9}`:'此版本未公布';
    input.oninput=input.onchange=()=>{if(input.value==='')delete shell.radio_capacity_bps[key];else shell.radio_capacity_bps[key]=Number(input.value)*1e9;document.dispatchEvent(new Event('modelchange'))};
    wrap.append(input);overrides.append(wrap);
  }
  for (const [key,label] of [['capacity_assumptions','RF 容量假设说明（填写覆盖值时必需）'],['terminal_assumptions','端口与单双向口径假设说明']]) {
    const wrap=document.createElement('label');wrap.textContent=label;const input=document.createElement('textarea');
    input.setAttribute('aria-label',label);input.value=shell[key].join('\n');
    input.oninput=input.onchange=()=>{shell[key]=input.value.split('\n').map(s=>s.trim()).filter(Boolean);document.dispatchEvent(new Event('modelchange'))};
    wrap.append(input);
    if(key==='capacity_assumptions')details.append(wrap);
    else{const assumptions=document.createElement('details'), title=document.createElement('summary');title.textContent='端口与单双向口径假设说明';assumptions.append(title,wrap);card.append(assumptions)}
  }
  renderLinkDefaults(card,shell,changed);
}
