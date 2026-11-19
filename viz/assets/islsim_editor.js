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

export function bindISLSim(profile){
  const parameters=nativeParameters(profile);
  if(!Array.isArray(parameters?.tx.baud_rate_options_ghz))return;
  profile.islsim_config=structuredClone(parameters);
  if(defaults)profile.islsim_source_sha256=structuredClone(defaults.source_sha256);
  return parameters;
}

function renderBaudRate(fields,tx){
  const fixedLabel=document.createElement('label');
  fixedLabel.textContent='固定波特率（GBaud，留空时自动选档）';
  const fixed=document.createElement('input');
  fixed.type='number';fixed.min='0';fixed.step='any';
  fixed.value=tx.baud_rate_ghz??'';
  fixedLabel.append(fixed);
  const candidatesLabel=document.createElement('label');
  candidatesLabel.textContent='候选波特率（GBaud，逗号分隔）';
  const candidates=document.createElement('input');
  candidates.type='text';candidates.required=true;
  candidates.value=tx.baud_rate_options_ghz.join(', ');
  candidatesLabel.append(candidates);
  const updateVisibility=()=>{
    candidatesLabel.hidden=tx.baud_rate_ghz!==null;
    candidates.disabled=candidatesLabel.hidden;
  };
  fixed.oninput=()=>{
    const value=fixed.value===''?null:fixed.valueAsNumber;
    fixed.setCustomValidity(value!==null&&!(value>0)?'请输入正波特率，或留空自动选档':'');
    if(!fixed.checkValidity())return;
    tx.baud_rate_ghz=value;
    updateVisibility();
    document.dispatchEvent(new Event('modelchange'));
  };
  candidates.oninput=()=>{
    const values=candidates.value.trim().split(/[,，\s]+/).map(Number);
    const valid=values.every(value=>Number.isFinite(value)&&value>0);
    candidates.setCustomValidity(valid?'':'请输入一个或多个正波特率');
    if(!valid)return;
    tx.baud_rate_options_ghz=values;
    document.dispatchEvent(new Event('modelchange'));
  };
  updateVisibility();
  fields.append(fixedLabel,candidatesLabel);
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
  if(!bindISLSim(profile)){
    const note=document.createElement('p');
    note.textContent='ISLSim 参数未加载，请检查服务后重试。';
    fields.append(note);
    return true;
  }
  renderBaudRate(fields,profile.islsim_config.tx);
  renderModulation(fields,profile);
  return true;
}
