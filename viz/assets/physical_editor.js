import {renderRadio} from '/assets/radio_editor.js';
import {initISLSim,renderISLSim,bindISLSim} from '/assets/islsim_editor.js';
import {initTleImporter} from '/assets/tle_importer.js';
import {sourceFields,quotaFields,preferenceOptions,inferredScenario,applyScenario} from '/assets/network_options.js';
import {get,post,field,group,tabs} from '/assets/studio_common.js';
import {enhancePhysicalFields} from '/assets/design-controls.js';
import {ensureInterCellProfile,interCellProfileNote,missingInterCellProfile} from '/assets/link_profiles.js';
import {bindFormationSize} from '/assets/formation_size.js';
const $=id=>document.getElementById(id);let config,defaultShell,defaultCell,radioProfiles,scenarioCache={};tabs('data-tab');
for (const [containerId, buttonId] of [['shells','addShell'],['cells','addCell'],['grounds','addGround']]) {
  const heading = $(containerId).previousElementSibling;
  if (heading?.tagName !== 'H2') continue;
  const row = document.createElement('div'); row.className = 'physical-section-heading';
  heading.before(row); row.append(heading, $(buttonId));
  $(buttonId).setAttribute('aria-label', $(buttonId).textContent); $(buttonId).textContent = '+';
}
$('json').closest('details').open = true;
$('json').closest('details').classList.add('physical-json');
function stripRuntime(){for(const key of ['transport_model','faults','random_faults','fault_timing','failure_policy','max_transfer_retries'])delete config.network[key]}
function removeButton(card,action){const b=document.createElement('button');b.textContent='−';b.className='group-remove';b.setAttribute('aria-label','删除'+(card.querySelector('h3')?.textContent||'此项'));b.title=b.getAttribute('aria-label');b.onclick=()=>{action();render()};card.append(b)}
function render(){defaultShell??=structuredClone(config.constellations.communication_shells[0]||defaultShell);defaultCell??=structuredClone(config.constellations.compute_formations[0]||defaultCell);stripRuntime();ensureInterCellProfile(config.network);for(const g of config.ground_ingresses){g.terminal_count??=1;g.altitude_km??=0}for(const id of ['time','shells','cells','grounds','access','connections','budgets'])$(id).replaceChildren();
for(const[path,label,type]of [['epoch_utc','历元 UTC','text'],['duration_s','实验时长 s','number'],['step_s','几何时间片间隔 s','number']])field($('time'),config,'time.'+path,label,type);
config.constellations.communication_shells.forEach((shell,i)=>{
const {card,fields}=group($('shells'),'通信壳层 '+(i+1));field(fields,shell,'shell_id','壳层 ID','text');
sourceFields(fields,card,shell,false,render,sync);
field(fields,shell,'topology','星间拓扑预设','text',[['grid','Grid / 4 邻接'],['honeycomb','Honeycomb / 3 邻接']]);
shell.topology_repair??='none';field(fields,shell,'topology_repair','缺位处理','text',[['none','baseline · 缺位留空'],['ronchetti_intra_orbit','method1 · Ronchetti 同轨缺口修补']]);
for(const[k,label]of [['isl_terminals','ISL 终端数'],['sgl_terminals','每星地面关联总上限']])field(fields,shell,k,label);
quotaFields(fields,shell);
for(const[k,label]of [['wrap_planes','轨道面首尾相接'],['wrap_slots','轨内闭环']])field(fields,shell,k,label,'checkbox');
renderRadio(fields,card,shell,radioProfiles,render);removeButton(card,()=>config.constellations.communication_shells.splice(i,1))});
config.constellations.compute_formations.forEach((cell,i)=>{
const{card,fields}=group($('cells'),'计算编队 '+(i+1));field(fields,cell,'name','编队 ID','text');sourceFields(fields,card,cell,true,render,sync);
const countInput=field(fields,cell,'cluster.satellites_per_cluster','每 cell 卫星数');
const shapeInput=field(fields,cell,'cluster.grid_shape','初始尺寸（随卫星数联动）','text');
bindFormationSize(cell.cluster,countInput,shapeInput,sync,{synthetic:cell.source.type!=='tle'});
for(const[k,label]of [['formation_pitch_m','成员间距 m'],['d_keepout_m','安全间距 m'],['sigma_navigation_m','导航误差 σ m'],['sigma_control_m','控制误差 σ m']])field(fields,cell,'cluster.'+k,label);
field(fields,cell,'topology','cell 内拓扑','text',[['ring','Ring'],['fully_connected','Fully connected']]);field(fields,cell,'isl_terminals','每成员 ISL 终端数');field(fields,cell,'sgl_terminals','每成员地面关联总上限');quotaFields(fields,cell);
cell.egress_policy??=cell.ingress_members?.length===1?'single':'multi';const mode=field(fields,cell,'egress_policy','cell 对外出口模式','text',[['single','单星出口 · 指定一个成员'],['multi','多星出口 · 指定成员集合'],['free','自由连接 · 全体成员可对外']]);mode.addEventListener('change',()=>{if(cell.egress_policy==='single')cell.ingress_members=[cell.ingress_members?.[0]??0];render()});
const label=document.createElement('label');label.textContent='对外接入成员编号（从 0 开始，逗号分隔）';const input=document.createElement('input');input.value=(cell.ingress_members||[0]).join(',');input.disabled=cell.egress_policy==='free';input.onchange=()=>{cell.ingress_members=input.value.split(',').map(s=>s.trim()===''?NaN:Number(s));sync()};label.append(input);fields.append(label);removeButton(card,()=>config.constellations.compute_formations.splice(i,1))});
config.ground_ingresses.forEach((ground,i)=>{const{card,fields}=group($('grounds'),'地面站 '+(i+1));for(const[k,label,type]of [['ingress_id','地面站 ID','text'],['lat_deg','纬度 °','number'],['lon_deg','经度 °','number'],['altitude_km','高度 km','number'],['min_elevation_deg','最小仰角 °','number'],['terminal_count','终端数','number']])field(fields,ground,k,label,type);ground.ground_kind??='gateway';field(fields,ground,'ground_kind','地面节点用途','text',[['gateway','RF 网关回传'],['user','用户接入']]);removeButton(card,()=>config.ground_ingresses.splice(i,1))});
config.network.handover_policy??='retain_valid';config.network.cross_layer_access_policy??='nearest_distance';
field($('access'),config,'network.handover_policy','动态链路切换策略','text',[['retain_valid','尽量使用原链路 · 失效后才切换'],['quality_first','按质量重新选择 · 允许切换有效链路']]);
field($('access'),config,'network.cross_layer_access_policy','计算—通信候选评分','text',[['nearest_distance','最近距离 / 最低传播延迟'],['highest_capacity','最高双向瓶颈带宽'],['longest_available','最长剩余可用窗口']]);
const handoverNote=document.createElement('p');handoverNote.textContent='适用于星地与计算—通信动态链路。质量优先按下方候选评分重新选择；新链路均需等待建链完成。固定星间邻居不随此策略变化。';$('access').append(handoverNote);
field($('access'),config,'network.ground_access_policy','地面侧候选偏好','text',preferenceOptions);config.network.ground_matching??='greedy';config.network.satellite_access_policy??='nearest_distance';field($('access'),config,'network.ground_matching','双端关联算法','text',[['greedy','全局偏好排序 · 双端配额约束'],['deferred_acceptance','双向偏好 · 地面提议 / 卫星择优']]);field($('access'),config,'network.satellite_access_policy','卫星侧候选偏好（双向匹配生效）','text',preferenceOptions);const role=document.createElement('label');role.textContent='允许计算 cell 直接接地面';const check=document.createElement('input');check.type='checkbox';check.checked=config.network.ground_roles?.includes('compute');check.disabled=inferredScenario(config)==='compute_only';check.onchange=()=>{config.network.ground_roles=check.checked?['communication','compute']:['communication'];sync()};role.append(check);$('access').append(role);
const interCell=field($('connections'),config,'network.inter_cell_links','允许 cell 间直接连接','checkbox');
interCell.addEventListener('change',render);
const interCellNote=document.createElement('p');interCellNote.textContent=config.network.inter_cell_links&&!config.network.link_profiles?.inter_cell?missingInterCellProfile:interCellProfileNote;$('connections').append(interCellNote);
for(const[k,label]of [['communication_isl','通信星间'],['cross_layer','计算—通信'],['cell_isl','cell 内部'],['inter_cell','cell 间直连'],['sgl','星地']]){const p=config.network.link_profiles[k];if(!p)continue;const{card,fields}=group($('budgets'),label);card.dataset.linkProfile=k;for(const[key,text]of [['setup_time_s','建链时间 s'],['bandwidth_bps','每方向终端速率上限 bps'],['max_distance_km','最大距离 km']])field(fields,p,key,text);if(k!=='sgl')renderISLSim(fields,p,render);if(p.capacity_model==='optical_photon_budget'){for(const[key,text]of [['transmit_power_w','发射功率 W'],['transmit_aperture_m','发射孔径 m'],['receive_aperture_m','接收孔径 m'],['wavelength_m','波长 m'],['required_photons_per_bit','接收光子/bit'],['pointing_loss_db','指向损耗 dB']])field(fields,p,key,text)}else if(p.capacity_model==='free_space_shannon'){for(const[key,text]of [['frequency_hz','频率 Hz'],['channel_bandwidth_hz','信道带宽 Hz'],['transmit_power_w','发射功率 W'],['transmit_gain_db','发射增益 dB'],['receive_gain_db','接收增益 dB'],['noise_temperature_k','噪声温度 K'],['system_loss_db','系统损耗 dB']])field(fields,p,key,text)}field(fields,p,'sun_exclusion_angle_deg','太阳避让角 °')}
enhancePhysicalFields(document.querySelector('.physical-page'));sync()}
function sync(){stripRuntime();config.scenario_type=inferredScenario(config);$('scenarioType').value=config.scenario_type;$('next').disabled=config.scenario_type==='communication_only';$('scenarioNote').textContent=config.scenario_type==='communication_only'?'纯通信场景请使用物理演示与路由诊断；当前 LLM Runtime 需要计算节点。':'可保存物理模型、演示拓扑，或进入完整 Serving / Runtime 场景。';$('json').value=JSON.stringify(config,null,2);let comm=0,compute=0;for(const c of config.constellations.communication_shells)comm+=c.source.type!=='tle'?c.source.planes*c.source.satellites_per_plane:(c.source.tle_text||'').split('\n').filter(x=>x.startsWith('1 ')).length;for(const c of config.constellations.compute_formations)compute+=c.source.type==='tle'?(c.source.tle_text||'').split('\n').filter(x=>x.startsWith('1 ')).length:c.source.planes*c.source.clusters_per_plane*c.cluster.satellites_per_cluster;$('count').textContent=comm+' 颗通信卫星 · '+compute+' 颗计算卫星 · '+config.ground_ingresses.length+' 个地面站'}
document.addEventListener('modelchange',sync);
$('scenarioType').onchange=()=>{scenarioCache.shells??=defaultShell?[structuredClone(defaultShell)]:[];scenarioCache.cells??=defaultCell?[structuredClone(defaultCell)]:[];applyScenario(config,$('scenarioType').value,scenarioCache);render()};
$('applyJson').onclick=()=>{try{config=JSON.parse($('json').value);render()}catch(e){$('status').textContent=e.message}};
$('addShell').onclick=()=>{const row=structuredClone(config.constellations.communication_shells[0]||defaultShell);row.shell_id='shell-'+Date.now();config.constellations.communication_shells.push(row);render()};$('addCell').onclick=()=>{const row=structuredClone(config.constellations.compute_formations[0]||defaultCell);row.name='compute-'+Date.now();config.constellations.compute_formations.push(row);render()};$('addGround').onclick=()=>{config.ground_ingresses.push({ingress_id:'ground-'+Date.now(),lat_deg:0,lon_deg:0,altitude_km:0,min_elevation_deg:10,terminal_count:1});render()};
async function save(next){
  try{
    if(config.network.inter_cell_links&&!config.network.link_profiles?.inter_cell)throw Error(missingInterCellProfile);
    const invalid=document.querySelector(':invalid');
    if(invalid){invalid.reportValidity();throw Error('请先修正无效输入')}
    if(next===true&&inferredScenario(config)==='communication_only')throw Error('纯通信场景请使用物理演示与路由诊断');
    const profiles=[config.network.link_profiles,...config.constellations.communication_shells.map(shell=>shell.hardware_link_profiles)];
    for(const group of profiles){
      for(const profile of Object.values(group??{})){
        if(profile.capacity_model==='islsim'||(!profile.capacity_model&&profile.islsim_source_sha256))bindISLSim(profile);
      }
    }
    $('preview').disabled=$('next').disabled=$('save').disabled=true;
    const doc=await post('/save-physical',{name:$('name').value,physical_network:config});
    history.replaceState(null,'','/physical?id='+doc.id);
    $('status').textContent='物理模型已保存到本地：'+doc.name;
    if(next){
      $('status').textContent='正在生成完整物理数据：全时域位置、链路与双向预算…';
      const job=await post('/prepare-physical',{id:doc.id});
      location.href='/runs/'+job.id+'/progress'+(next===true?'?next=scene&model='+encodeURIComponent(doc.id):'');
    }
  }catch(e){$('status').textContent=e.message}
  finally{$('preview').disabled=$('save').disabled=false;$('next').disabled=inferredScenario(config)==='communication_only'}
}
$('save').onclick=()=>save(false);$('next').onclick=()=>save(true);$('preview').onclick=()=>save('preview');$('loadPreset').onclick=async()=>{try{config=await get('/templates/'+$('preset').value);render()}catch(e){$('status').textContent=e.message}};
(async()=>{await initISLSim();radioProfiles=await get('/radio-profiles');const initial=await get('/templates/starlink_550');defaultShell=structuredClone(initial.constellations.communication_shells[0]);defaultCell=structuredClone(initial.constellations.compute_formations[0]);const id=new URLSearchParams(location.search).get('id');if(id){const doc=await get('/documents/physical/'+id);config=doc.physical_network;$('name').value=doc.name}else config=initial;render();initTleImporter({readConfig:()=>config,readShell:()=>defaultShell,replaceConfig:value=>{config=value;scenarioCache={}},render});})().catch(e=>$('status').textContent=e.message);
