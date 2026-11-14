import {field,get} from '/assets/studio_common.js';

let defaults;
export async function initISLSim(){
  try{defaults=await get('/islsim-defaults')}catch{defaults=null}
}
export function nativeParameters(profile){
  const overrides=profile.islsim_config,base=defaults?.parameters;
  if(!base&&!['tx','rx','channel'].every(section=>overrides?.[section]))return;
  return Object.fromEntries(['tx','rx','channel'].map(section=>[section,{...base?.[section],...overrides?.[section]}]));
}

function modulationMode(pool){
  const orders=pool.map(item=>item[0]).join(',');
  return orders==='16,8,4'?'auto':['4','8','16','64'].includes(orders)?orders:'custom';
}

function renderModulation(fields,profile){
  const tx=profile.islsim_config.tx,wrap=document.createElement('label');
  wrap.textContent='ISLSim · 调制策略';
  const select=document.createElement('select');select.setAttribute('aria-label',wrap.textContent);
  for(const[value,text]of [['auto','自动适配：16QAM → 8QAM → QPSK'],['4','固定 QPSK'],['8','固定 8QAM'],['16','固定 16QAM'],['64','固定 64QAM'],['custom','自定义候选集合与顺序']]){
    const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);
  }
  select.value=modulationMode(tx.modulation_pool);
  const details=document.createElement('details'),summary=document.createElement('summary');
  summary.textContent='调制候选集合与顺序（高级）';details.append(summary);
  const label=document.createElement('label');label.textContent='ISLSim · 调制池 JSON';
  const pool=document.createElement('textarea');pool.setAttribute('aria-label',label.textContent);pool.value=JSON.stringify(tx.modulation_pool);
  pool.oninput=()=>{try{
    const value=JSON.parse(pool.value);
    if(!Array.isArray(value)||!value.length||value.some(row=>!Array.isArray(row)||row.length!==2||!Number.isInteger(row[0])||row[0]<4||!Number.isInteger(Math.log2(row[0]))||typeof row[1]!=='string'||!row[1]))throw Error();
    tx.modulation_pool=value;pool.setCustomValidity('');select.value=modulationMode(value);document.dispatchEvent(new Event('modelchange'));
  }catch{pool.setCustomValidity('请输入非空的 [调制阶数, 标签] 数组；阶数须为不小于 4 的 2 的幂')}};
  select.onchange=()=>{
    if(select.value==='custom'){details.open=true;return}
    tx.modulation_pool=(select.value==='auto'?[16,8,4]:[Number(select.value)]).map(order=>[order,(tx.dual_polarization?'DP-':'')+(order===4?'QPSK':order+'QAM')]);
    pool.value=JSON.stringify(tx.modulation_pool);pool.setCustomValidity('');document.dispatchEvent(new Event('modelchange'));
  };
  wrap.append(select);fields.append(wrap);label.append(pool);details.append(label);fields.append(details);
  const note=document.createElement('p');note.textContent='OrbitServe 提供允许的调制集合；ISLSim 按候选顺序选择首个满足 BER 门限的方式。固定模式也必须满足 BER，否则链路不可用。';fields.append(note);
}
export function renderISLSim(fields,profile,changed){
  profile.capacity_model??='islsim';
  const select=field(fields,profile,'capacity_model','星间链路计算模型','text',[
    ['islsim','ISLSim 激光链路模型（默认）'],
    ['optical_photon_budget','光子预算'],['fixed','固定带宽']]);
  select.addEventListener('change',changed);
  if(profile.capacity_model!=='islsim')return false;
  const parameters=nativeParameters(profile);
  if(parameters){profile.islsim_config=structuredClone(parameters);if(!profile.islsim_source_sha256&&defaults)profile.islsim_source_sha256=structuredClone(defaults.source_sha256)}
  const note=document.createElement('p');
  note.textContent=parameters?'由外部 ISLSim 原始模型计算接收功率、SNR、调制、带宽和传播时延；链路不满足 BER 门限时不可用。':'ISLSim 尚未配置。请配置外部源码目录，或显式选择其他模型。';fields.append(note);
  if(!parameters)return true;
  for(const[path,label]of [['tx.tx_power_w','发射功率 W'],['tx.tx_aperture_m','发射孔径 m'],['tx.wavelength_m','波长 m'],['tx.baud_rate_ghz','波特率 GBaud'],['tx.wdm_channels','WDM 信道数'],['tx.fec_overhead','FEC 开销比例'],['rx.rx_aperture_m','接收孔径 m'],['rx.preamp_gain_db','前置放大增益 dB（当前模型仅校验）'],['rx.preamp_nf_db','噪声系数 dB'],['rx.ber_threshold','BER 门限'],['channel.link_margin_db','链路余量 dB'],['channel.optical_efficiency','光学效率']])field(fields,profile,'islsim_config.'+path,'ISLSim · '+label);
  const gainNote=document.createElement('p');gainNote.textContent='当前外部 ISLSim 的高增益近似未使用 preamp_gain_db 计算 SNR；修改该值不会改变当前模型结果。';fields.append(gainNote);
  field(fields,profile,'islsim_config.tx.dual_polarization','ISLSim · 双偏振','checkbox');
  field(fields,profile,'islsim_config.tx.wdm_power_mode','ISLSim · WDM 功率分配','text',[
    ['total_power_fixed','总发射功率固定'],['per_channel_power_fixed','每信道功率固定']]);
  const capLabel=document.createElement('label');capLabel.textContent='ISLSim · 终端总速率上限 Gbps（留空为不限）';
  const cap=document.createElement('input');cap.type='number';cap.step='any';cap.min='0';cap.setAttribute('aria-label',capLabel.textContent);cap.value=profile.islsim_config.rx.terminal_cap_gbps??'';
  cap.oninput=()=>{profile.islsim_config.rx.terminal_cap_gbps=cap.value===''?null:Number(cap.value);document.dispatchEvent(new Event('modelchange'))};capLabel.append(cap);fields.append(capLabel);
  renderModulation(fields,profile);
  return true;
}
