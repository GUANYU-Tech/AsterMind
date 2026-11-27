import {field,group} from './studio_common.js';
import {renderISLSim} from './islsim_editor.js';
import {renderRadioCapacity} from './radio_editor.js';

const opticalLabels = {
  communication_isl:'通信星间', cross_layer:'计算星—通信星',
  cell_isl:'编队内星间', inter_cell:'编队间直连',
};
export const opticalKinds = Object.keys(opticalLabels);
const rfLabels = {
  user_uplink:'用户上行', user_downlink:'用户下行',
  feeder_uplink:'网关上行', feeder_downlink:'网关下行',
};
const rfKinds = Object.keys(rfLabels);

export function initializeLinkBudgets(config, hardware, {newModel=false}={}) {
  const network=config.network, profiles=network.link_profiles;
  const shells=config.constellations.communication_shells;
  const legacy=!network.optical_link_model;
  if (legacy) {
    // Seed from the first shell's effective link budget, retaining scene geometry.
    const shell=shells[0];
    const preset=hardware[shell?.radio_profile]?.simulation_defaults.hardware_link_profiles;
    const hardwareProfile=kind=>({...preset?.[kind],...shell?.hardware_link_profiles?.[kind]});
    const optical=hardwareProfile('communication_isl');
    const oldOptical=optical.bandwidth_bps ? optical : profiles[shell?.isl_profile];
    if (oldOptical) {
      const {setup_time_s,max_distance_km,sun_exclusion_angle_deg}=profiles.communication_isl ?? {};
      profiles.communication_isl={...profiles.communication_isl,...oldOptical,setup_time_s,max_distance_km,sun_exclusion_angle_deg};
    }
    const known=opticalKinds.map(k=>profiles[k]).filter(Boolean);
    network.optical_link_model=!newModel && known.length && known.every(p=>p.capacity_model==='fixed')?'fixed':'islsim';
    for (const kind of rfKinds) {
      const previous=hardwareProfile(kind);
      const direction=kind.endsWith('uplink')?'uplink_profile':'downlink_profile';
      const fallback=profiles[kind] ?? profiles[profiles.sgl?.[direction]] ?? profiles.sgl;
      profiles[kind]=structuredClone(previous.bandwidth_bps ? previous : fallback ?? preset?.[kind]);
    }
    for (const kind of opticalKinds) {
      profiles[kind]??=structuredClone(profiles.cross_layer ?? profiles.communication_isl ?? profiles.cell_isl);
      const p=profiles[kind];
      p.setup_time_s??=2;
      delete p.required_photons_per_bit;
      if (p.assumptions) p.assumptions=p.assumptions.filter(note=>!/photons?\s*\/\s*bit|photons? per bit|光子预算/i.test(note));
    }
  }
  for (const shell of shells) {
    delete shell.hardware_link_profiles;
    delete shell.isl_profile;
  }
  for (const edges of Object.values(network.cell_link_overrides ?? {})) {
    for (const edge of edges) edge.profile_id='cell_isl';
  }
  const used=new Set([...opticalKinds,...rfKinds,'sgl']);
  for (const key of Object.keys(profiles)) if (!used.has(key)) delete profiles[key];
  // Only geometry is shared between RF directions. Directional profiles carry the rate.
  const ground=profiles.sgl;
  profiles.sgl={capacity_model:'fixed',setup_time_s:ground.setup_time_s??0.1,
    max_distance_km:ground.max_distance_km,sun_exclusion_angle_deg:ground.sun_exclusion_angle_deg,
    bandwidth_bps:Math.max(...rfKinds.map(k=>profiles[k].bandwidth_bps))};
  for (const kind of rfKinds) {
    const profile=profiles[kind];
    profile.capacity_model??='fixed';
    profile.setup_time_s=profiles.sgl.setup_time_s;
    if (profile.capacity_model==='free_space_shannon') {
      profile.noise_temperature_k??=290;
      profile.implementation_efficiency??=1;
    }
    delete profiles[kind].uplink_profile;
    delete profiles[kind].downlink_profile;
  }
  setOpticalMode(network,network.optical_link_model);
  return legacy;
}

function setOpticalMode(network,mode) {
  network.optical_link_model=mode;
  for (const kind of opticalKinds) {
    const p=network.link_profiles[kind];
    p.capacity_model=mode;
    if (mode==='islsim') {
      p.islsim_config??={};
      (p.islsim_config.rx??={}).terminal_cap_gbps=null;
    }
  }
}

function scaled(parent,profile,key,label,scale=1,{optional=false,positive=true}={}) {
  const wrap=document.createElement('label'),input=document.createElement('input');
  wrap.textContent=label;
  input.type='number';input.step='any';input.required=!optional;
  input.dataset.designControl='number';input.setAttribute('aria-label',label);
  if (positive) input.min='0';
  input.value=profile[key]==null?'':profile[key]/scale;
  input.oninput=()=>{
    const n=input.valueAsNumber;
    input.setCustomValidity(positive && n<=0?'请输入大于零的数值':'');
    profile[key]=Number.isFinite(n)?n*scale:null;
    document.dispatchEvent(new Event('modelchange'));
  };
  wrap.append(input);parent.append(wrap);
}

function geometry(fields,p) {
  field(fields,p,'setup_time_s','建链时间 s').min='0';
  scaled(fields,p,'max_distance_km','最大距离 km',1,{optional:true});
  scaled(fields,p,'sun_exclusion_angle_deg','太阳避让角 °',1,{optional:true,positive:false});
}

export function renderLinkBudgets(host,config,hardware,render) {
  const net=config.network,profiles=net.link_profiles;
  const optical=group(host,'星间光链路');
  const model=field(optical.fields,net,'optical_link_model','星间链路计算模型','text',[
    ['islsim','ISLSim 激光链路模型（默认）'],['fixed','固定带宽'],
  ]);
  model.addEventListener('change',()=>{setOpticalMode(net,model.value);render()});
  for (const shell of config.constellations.communication_shells) {
    if (!(shell.optical_terminal_bandwidth_bps>0)) continue;
    const note=document.createElement('p'),label=document.createElement('span');
    label.textContent='光终端单向硬件上限';
    note.append(shell.shell_id+' · ',label,' · '+shell.optical_terminal_bandwidth_bps/1e9+' Gbps');
    optical.card.append(note);
  }
  const fixed=net.optical_link_model==='fixed';
  for (const kind of opticalKinds) {
    const p=profiles[kind],{card,fields}=group(optical.card,opticalLabels[kind]);
    card.dataset.linkProfile=kind;
    scaled(fields,p,'bandwidth_bps',fixed?'固定单向带宽（Gbps）':'单向速率上限（Gbps）',1e9);
    geometry(fields,p);
    if (!fixed) renderISLSim(fields,p);
  }
  const ground=group(host,'星地 RF');
  geometry(ground.fields,profiles.sgl);
  for (const kind of rfKinds) {
    const p=profiles[kind],{card,fields}=group(ground.card,rfLabels[kind]);
    card.dataset.linkProfile=kind;
    scaled(fields,p,'bandwidth_bps','单向速率上限（Gbps）',1e9);
    if (p.capacity_model!=='free_space_shannon') continue;
    const advanced=document.createElement('details'),heading=document.createElement('summary'),params=document.createElement('div');
    heading.textContent='射频链路参数';params.className='fields';
    advanced.append(heading,params);card.append(advanced);
    for (const [key,label,scale] of [
      ['frequency_hz','中心频率 GHz',1e9],['channel_bandwidth_hz','单载波带宽 MHz',1e6],
      ['transmit_power_w','发射功率 W',1],['noise_temperature_k','噪声温度 K',1],
      ['implementation_efficiency','实现效率',1],
    ]) scaled(params,p,key,label,scale);
    for (const [key,label] of [['transmit_aperture_m','发射有效孔径 m'],['receive_aperture_m','接收有效孔径 m']])
      scaled(params,p,key,label,1,{optional:true});
    for (const [key,label] of [['transmit_gain_db','发射增益 dB'],['receive_gain_db','接收增益 dB'],['system_loss_db','系统损耗 dB']]) {
      p[key]??=0;scaled(params,p,key,label,1,{positive:false});
    }
  }
  for (const shell of config.constellations.communication_shells) renderRadioCapacity(ground.card,shell,hardware);
}
