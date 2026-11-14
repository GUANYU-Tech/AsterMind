import {field} from '/assets/studio_common.js';

export const preferenceOptions = [['nearest_distance','最近距离'],['longest_available','最长可用时间'],['highest_elevation','最高仰角'],['highest_capacity','最高双向瓶颈速率']];

export function nullableInteger(parent,obj,key,label){
  const wrap=document.createElement('label');wrap.textContent=label;
  const input=document.createElement('input');input.type='number';input.min='0';input.step='1';input.placeholder='仅受总配额限制';input.value=obj[key]??'';input.setAttribute('aria-label',label);
  input.onchange=()=>{if(input.value==='')delete obj[key];else obj[key]=Number(input.value);document.dispatchEvent(new Event('modelchange'))};wrap.append(input);parent.append(wrap);
}

export function quotaFields(parent,obj){
  nullableInteger(parent,obj,'max_user_links','每星同时接入用户节点上限');
  nullableInteger(parent,obj,'max_gateway_links','每星同时连接信关站上限');
}

export function sourceFields(parent,card,obj,compute,render,sync){
  const options=[['walker_delta','Walker 参数'],['dawn_dusk','Dawn-dusk · 晨昏太阳同步'],['tle','TLE · 实测轨道']];
  const source=field(parent,obj,'source.type','轨道来源','text',options);
  source.addEventListener('change',()=>{
    const s=obj.source;
    if(s.type==='dawn_dusk'){s.planes=1;s.ltan_hours??=6;s.plane_spread_deg??=0;s.u0_deg??=0}
    if(s.type==='tle'){s.tle_text??='';delete s.raan0_deg;delete s.u0_deg}
    if(s.type==='walker_delta'){s.inclination_deg??=53;s.walker_phase=0;s.raan0_deg??=0;s.u0_deg??=0}
    render();
  });
  if(obj.source.type!=='tle'){
    for(const [key,label]of [['altitude_km','高度 km'],['planes','轨道面数'],[compute?'clusters_per_plane':'satellites_per_plane',compute?'每面 cell 数':'每面卫星数'],['u0_deg','初始纬度幅角 °']])field(parent,obj,'source.'+key,label);
    if(obj.source.type==='walker_delta')for(const[key,label]of [['inclination_deg','倾角 °'],['walker_phase','Walker 相位'],['raan0_deg','初始 RAAN °']])field(parent,obj,'source.'+key,label);
    else{field(parent,obj,'source.ltan_hours','中心轨道升交点地方时','number',[[6,'06:00 · 晨侧'],[18,'18:00 · 昏侧']]);field(parent,obj,'source.plane_spread_deg','轨道面 RAAN 展宽 °（0–30）');const note=document.createElement('p');note.textContent='一阶 J2 太阳同步初始设计；各星独立 SGP4 传播，不含轨道保持。多面可在晨昏线附近展宽，是否入影由实际几何计算。';card.append(note)}
    return;
  }
  if(!compute)for(const[key,label]of [['planes','标称轨道面数（含缺失面）'],['satellites_per_plane','每面标称槽位数（含空位）']])field(parent,obj,'source.'+key,label);
  const upload=document.createElement('input');upload.type='file';upload.accept='.txt,.tle';upload.setAttribute('aria-label','导入 TLE 文件');
  const text=document.createElement('textarea');text.value=obj.source.tle_text||'';text.setAttribute('aria-label','TLE 文本');text.onchange=()=>{obj.source.tle_text=text.value;sync()};
  upload.onchange=async()=>{if(upload.files[0]){obj.source.tle_text=await upload.files[0].text();text.value=obj.source.tle_text;sync()}};card.append(upload,text);
  if(compute){
    const label=document.createElement('label');label.textContent='cell 成员分组：cell 名称 → 按成员顺序排列的 catalog 编号';
    const groups=document.createElement('textarea');groups.setAttribute('aria-label','计算 TLE cell 成员分组');groups.placeholder='{"cell-a":[12345,12346]}';groups.value=JSON.stringify(obj.source.cell_members||{},null,2);
    groups.onchange=()=>{try{obj.source.cell_members=JSON.parse(groups.value);groups.setCustomValidity('');sync()}catch(e){groups.setCustomValidity(e.message);groups.reportValidity()}};label.append(groups);card.append(label);
    const note=document.createElement('p');note.textContent='每颗导入卫星必须恰好归入一个 cell，成员数须与每 cell 卫星数一致。保留各星实测根数；不会按编队间距改写 TLE，间距/尺寸参数仅用于合成编队。';card.append(note);return;
  }
  const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='不规则 TLE 的槽位映射（可选）';details.append(summary);
  const mapping=document.createElement('textarea');mapping.setAttribute('aria-label','TLE catalog 到槽位映射');mapping.placeholder='{"12345": [0,0], "12346": [0,3]}';mapping.value=obj.source.grid_positions?JSON.stringify(obj.source.grid_positions,null,2):'';
  mapping.onchange=()=>{try{if(mapping.value.trim())obj.source.grid_positions=JSON.parse(mapping.value);else delete obj.source.grid_positions;mapping.setCustomValidity('');sync()}catch(e){mapping.setCustomValidity(e.message);mapping.reportValidity()}};details.append(mapping);card.append(details);
  const note=document.createElement('p');note.textContent='默认在共同历元按标称均匀网格拟合，偏差超过槽距 20% 的节点不参与 ISL 网格。已知槽位时请提供 catalog 编号→[面,槽]；省略编号保留为未分配，绝不补造卫星。该拟合不等同于 Ronchetti 的自动壳层/间距恢复。';card.append(note);
}

export function inferredScenario(config){const c=!!config.constellations.communication_shells.length,p=!!config.constellations.compute_formations.length;return c&&p?'mixed':p?'compute_only':'communication_only'}

export function applyScenario(config,mode,cache){
  const c=config.constellations;
  if(c.communication_shells.length)cache.shells=structuredClone(c.communication_shells);
  if(c.compute_formations.length)cache.cells=structuredClone(c.compute_formations);
  c.communication_shells=mode==='compute_only'?[]:c.communication_shells.length?c.communication_shells:structuredClone(cache.shells||[]);
  c.compute_formations=mode==='communication_only'?[]:c.compute_formations.length?c.compute_formations:structuredClone(cache.cells||[]);
  config.scenario_type=mode;
  if(mode==='compute_only'){config.network.ground_roles=['compute'];for(const f of c.compute_formations)if(!f.sgl_terminals)f.sgl_terminals=1}
  else config.network.ground_roles=['communication'];
  if(mode==='communication_only')config.route_probes=[];
}
