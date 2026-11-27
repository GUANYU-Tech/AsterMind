import {newPhysicalPreset,cellKeys,satelliteCount,cellGeometry,cellLayout,memberIds,physicalPayload,loadPhysicalCompute} from './physical-cells.js';
import {sceneCellDraft,sceneCellLayout} from './scene-cell-parameters.js';
import {renderIngressFields} from './ingress-fields.js';
import {showCellList} from './physical-cell-list.js';
import {renderConfigProfile} from './config-profile.js';
import {renderRadio} from '/assets/radio_editor.js';
import {initISLSim,bindISLSim} from '/assets/islsim_editor.js';
import {createFastLinkBudget} from './fast-link-budget.js';
import {initCommunicationDialog} from './communication-dialog.js';
import {groundDefaults,initGroundImport,parseGroundCsv} from './ground-import.js';
import {sourceFields,quotaFields,preferenceOptions,inferredScenario} from '/assets/network_options.js';
import {get,post,field,group,tabs} from '/assets/studio_common.js';
import {enhancePhysicalFields} from '/assets/design-controls.js';
import {initializeLinkBudgets,renderLinkBudgets,opticalKinds} from './scene-link-budgets.js';
const $=id=>document.getElementById(id);let config,defaultShell,defaultCell,radioProfiles;tabs('data-tab');
const sceneQuery=new URLSearchParams(location.search);
const sceneDraft=sceneQuery.get('scene_draft');
let capabilities, epStrategy='auto';
const linkBudget=createFastLinkBudget($('fastLinkBudget'));
$('computeMode').addEventListener('change',()=>sync());
const selectedCell=sceneQuery.get('cell');
const orbitPanels=new WeakMap();
const selectedFormation=()=>[...config.constellations.compute_formations].sort((a,b)=>b.name.length-a.name.length)
 .find(formation=>selectedCell?.startsWith(formation.name+'-'));
for (const [containerId, buttonId] of [['shells','addShell'],['cells','addCell']]) {
  const heading = $(containerId).previousElementSibling;
  if (heading?.tagName !== 'H2') continue;
  const row = document.createElement('div'); row.className = 'physical-section-heading';
  heading.before(row); row.append(heading, $(buttonId));
  $(buttonId).setAttribute('aria-label', $(buttonId).textContent);
}

$('json').closest('details').classList.add('physical-json');
function stripRuntime(){for(const key of ['transport_model','faults','random_faults','fault_timing','failure_policy','max_transfer_retries'])delete config.network[key]}
function removeButton(card,action){const b=document.createElement('button');b.textContent='−';b.className='group-remove';b.setAttribute('aria-label','删除'+(card.querySelector('h3')?.textContent||'此项'));b.title=b.getAttribute('aria-label');b.onclick=()=>{action();render()};card.append(b)}
const modelSource=$('modelSource');
let loadedSource, preparedExample;
function addSavedModel(doc,first=false){
  const group=$('savedModels'),value='?id='+encodeURIComponent(doc.id);
  group.querySelector('option[value=""]')?.remove();
  const option=new Option(doc.name+' · '+doc.saved_at.slice(0,19).replace('T',' '),value);
  option.translate=false;
  if(first)group.prepend(option);
  else group.append(option);
  return value;
}
function selectLoadedSource(value){
  loadedSource=value;
  modelSource.value=value;
  $('loadModel').disabled=true;
  if(value!=='?example=case_study_iii'){
    preparedExample=undefined;
    $('caseStudyExperiment').hidden=true;
    $('caseStudyDurationNote')?.setAttribute('hidden','');
  }
}
modelSource.onchange=()=>{$('loadModel').disabled=saving||modelSource.value===loadedSource};
$('loadModel').onclick=()=>{
  const target=new URL('/physical'+modelSource.value,location.origin);
  if(sceneDraft)target.searchParams.set('scene_draft',sceneDraft);
  if(document.querySelector('[data-tab="compute"]').getAttribute('aria-selected')==='true')target.hash='cells';
  location.assign(target);
};
function render(){
  const previousType=config.scenario_type;
  const type=inferredScenario(config);
  if(type)config.scenario_type=type;
  else delete config.scenario_type;
  if(type==='compute_only')config.network.ground_roles=['compute'];
  if(type==='communication_only')config.network.ground_roles=['communication'];
  if(previousType==='compute_only'&&type==='mixed'){
    config.network.ground_roles=['communication','compute'];
  }
  stripRuntime();for(const g of config.ground_ingresses){g.terminal_count??=1;g.altitude_km??=0}for(const id of ['time','shells','cells','grounds','access','connections','budgets'])$(id).replaceChildren();
for(const[path,label,type]of [['epoch_utc','实验开始时间（UTC）','text'],['duration_s','实验时长 s','number'],['step_s','几何时间片间隔 s','number']]){
 const input=field($('time'),config,'time.'+path,label,type);
 if(path==='duration_s'&&preparedExample)input.setAttribute('aria-describedby','caseStudyDurationNote');
}
if(preparedExample){
 const note=document.createElement('small');
 note.id='caseStudyDurationNote';
 note.textContent='PS：Case Study III 示例实验中，只有前 600s 请求到达，后面一个60s时间片仅留给请求执行';
 $('time').append(note);
}
config.constellations.communication_shells.forEach((shell,i)=>{
const {card,fields}=group($('shells'),'通信壳层 '+(i+1));field(fields,shell,'shell_id','壳层 ID','text');
sourceFields(fields,card,shell,false,render,sync);
field(fields,shell,'topology','星间拓扑预设','text',[['grid','Grid / 4 邻接'],['honeycomb','Honeycomb / 3 邻接'],...(shell.source.type==='tle'?[['explicit','显式链路（默认无固定星间链路）']]:[])]);
if(shell.source.type==='tle'){
  const details=document.createElement('details'),caption=document.createElement('summary');
  caption.textContent='实测壳层拓扑';details.append(caption);card.append(details);
  shell.topology_repair??='none';
  field(details,shell,'topology_repair','缺位处理','text',[['none','保留空位'],['ronchetti_intra_orbit','Ronchetti 同轨缺口修补']]);
  field(details,shell,'wrap_planes','轨道面首尾相接','checkbox');
  field(details,shell,'wrap_slots','轨内闭环','checkbox');
}else{
  shell.wrap_planes=true;shell.wrap_slots=true;
}
quotaFields(fields,shell);
renderRadio(fields,card,shell,radioProfiles,render);removeButton(card,()=>config.constellations.communication_shells.splice(i,1))});
config.constellations.compute_formations.forEach((cell,i)=>{
const {card,fields}=group($('cells'),'计算星座 '+(i+1)+' · '+cell.name);
card.dataset.formationName=cell.name;
const list=document.createElement('div');list.className='formation-cell-list';card.insertBefore(list,fields);
let advanced=orbitPanels.get(cell);
if(!advanced){
  advanced=document.createElement('details');
  orbitPanels.set(cell,advanced);
}
const caption=document.createElement('summary');caption.textContent='星座轨道与接入参数';advanced.replaceChildren(caption,fields);card.append(advanced);
const identity=field(fields,cell,'name','星座 ID','text');identity.required=true;
identity.addEventListener('change',()=>{
 const old=card.dataset.formationName;
 if(cell.cell_overrides)cell.cell_overrides=Object.fromEntries(Object.entries(cell.cell_overrides).map(([id,v])=>[cell.name+id.slice(old.length),v]));
 render();
});
sourceFields(fields,advanced,cell,true,()=>{
  const restoreFocus=document.activeElement===advanced.querySelector('[data-orbit-source]');
  render();
  if(restoreFocus)advanced.querySelector('[data-orbit-source]').focus({preventScroll:true});
},render);
const refreshList=()=>showCellList(list,cell,capabilities,()=>({
 mode:$('computeMode').value, epStrategy, linkProfile:config.network.link_profiles.cell_isl,
  cellLinkOverrides:config.network.cell_link_overrides,
}),render,selectedCell);
refreshList();
for(const label of ['轨道面数','每面 cell 数']){
 const input=fields.querySelector('[aria-label="'+label+'"]');
 if(input){input.min='1';input.step='1';input.required=true;input.addEventListener('change',refreshList)}
}
for(const[k,label]of [['d_keepout_m','安全间距 m'],['sigma_navigation_m','导航误差 σ m'],['sigma_control_m','控制误差 σ m']])field(fields,cell,'cluster.'+k,label);
field(fields,cell,'sgl_terminals','每成员地面关联总上限');quotaFields(fields,cell);
cell.egress_policy??=cell.ingress_members?.length===1?'single':'multi';const mode=field(fields,cell,'egress_policy','cell 对外出口模式','text',[['single','单星出口 指定一个成员'],['multi','多星出口 指定成员集合'],['free','自由连接 全体成员可对外']]);mode.addEventListener('change',()=>{if(cell.egress_policy==='single')cell.ingress_members=[cell.ingress_members?.[0]??0];delete cell.ingress_isl_terminals;render()});
renderIngressFields(fields,cell,sync);
removeButton(card,()=>config.constellations.compute_formations.splice(i,1))});
config.ground_ingresses.forEach((ground,i)=>{const{card,fields}=group($('grounds'),'地面站 '+(i+1));for(const[k,label,type]of [['ingress_id','地面站 ID','text'],['lat_deg','纬度 °','number'],['lon_deg','经度 °','number'],['altitude_km','高度 km','number'],['min_elevation_deg','最小仰角 °','number'],['terminal_count','终端数','number']])field(fields,ground,k,label,type);ground.ground_kind??='gateway';field(fields,ground,'ground_kind','地面节点用途','text',[['gateway','RF 网关回传'],['user','用户接入']]);removeButton(card,()=>config.ground_ingresses.splice(i,1))});
config.network.handover_policy??='retain_valid';config.network.cross_layer_access_policy??='nearest_distance';
field($('access'),config,'network.handover_policy','动态链路切换策略','text',[['retain_valid','尽量使用原链路 失效后才切换'],['quality_first','按质量重新选择 允许切换有效链路']]);
field($('access'),config,'network.cross_layer_access_policy','计算—通信候选评分','text',[['nearest_distance','最近距离 / 最低传播延迟'],['highest_capacity','最高双向瓶颈带宽'],['longest_available','最长剩余可用窗口']]);
const handoverNote=document.createElement('p');handoverNote.textContent='适用于星地与计算—通信动态链路。质量优先按下方候选评分重新选择；新链路均需等待建链完成。固定星间邻居不随此策略变化。';$('access').append(handoverNote);
field($('access'),config,'network.ground_access_policy','地面侧候选偏好','text',preferenceOptions);config.network.ground_matching??='greedy';config.network.satellite_access_policy??='nearest_distance';field($('access'),config,'network.ground_matching','双端关联算法','text',[['greedy','全局偏好排序 双端配额约束'],['deferred_acceptance','双向偏好 地面提议 / 卫星择优']]);field($('access'),config,'network.satellite_access_policy','卫星侧候选偏好（双向匹配生效）','text',preferenceOptions);const role=document.createElement('label');role.textContent='允许计算 cell 直接接地面';const check=document.createElement('input');check.type='checkbox';check.checked=config.network.ground_roles?.includes('compute');check.disabled=type!=='mixed';check.onchange=()=>{config.network.ground_roles=check.checked?['communication','compute']:['communication'];sync()};role.append(check);$('access').append(role);
const interCell=field($('connections'),config,'network.inter_cell_links','允许 cell 间直接连接','checkbox');
interCell.addEventListener('change',render);
renderLinkBudgets($('budgets'),config,radioProfiles,render);
enhancePhysicalFields(document.querySelector('.physical-page'));sync()}
const physicalLabels={schema_version:'配置版本',time:'时间',constellations:'星座资源',ground_ingresses:'地面站',network:'通信网络',route_probes:'路由探测',scenario_type:'场景类型',compute_formations:'算力星座',communication_shells:'通信星座'};
let saving=false;
function updateActions(){
  const type=inferredScenario(config);
  $('loadModel').disabled=saving||modelSource.value===loadedSource;
  $('save').disabled=$('preview').disabled=saving||!type;
  $('next').disabled=saving||!config.constellations.compute_formations.length;
  const note=$('scenarioNote');
  let message='';
  if(!type)message='请先添加通信壳层或计算编队。';
  else if(type==='communication_only')message='可以保存和预览通信拓扑。要运行推理，请添加计算编队。';
  else if(type==='compute_only'&&config.constellations.compute_formations.every(f=>f.sgl_terminals===0)){
    message='所有计算编队的星地终端数均为 0。要接收地面请求，请在计算星座中设置星地终端。';
  }
  note.textContent=message;
  note.hidden=!note.textContent;
}
function sync(){
  stripRuntime();
  updateActions();
  const note=$('caseStudyDurationNote');
  if(note)note.hidden=!preparedExample||config.time.duration_s!==660||config.time.step_s!==60;
  linkBudget.update(config,$('computeMode').value==='fast_mode');
  renderConfigProfile($('json'),config,physicalLabels);let comm=0,compute=0;for(const c of config.constellations.communication_shells)comm+=c.source.type!=='tle'?c.source.planes*c.source.satellites_per_plane:(c.source.tle_text||'').split('\n').filter(x=>x.startsWith('1 ')).length;for(const c of config.constellations.compute_formations)compute+=satelliteCount(c);$('count').textContent=comm+' 颗通信卫星 '+compute+' 颗计算卫星 '+config.ground_ingresses.length+' 个地面站'}
document.addEventListener('modelchange',()=>{if(!$('addCommunication').open)sync()});
const addDialog=$('addFormation'), addForm=addDialog.querySelector('form');
const nameInput=addForm.elements.formationName;
$('addCell').onclick=()=>{
 let n=1;while(config.constellations.compute_formations.some(row=>row.name==='compute-'+n))n++;
 nameInput.value='compute-'+n;nameInput.setCustomValidity('');addForm.elements.cellCount.value=2;addDialog.showModal();
};
$('cancelFormation').onclick=()=>addDialog.close();
nameInput.oninput=()=>nameInput.setCustomValidity('');
addForm.onsubmit=event=>{
 event.preventDefault();
 const name=nameInput.value.trim();
 if(!name || config.constellations.compute_formations.some(row=>row.name===name)){
  nameInput.setCustomValidity('请填写唯一的星座 ID');nameInput.reportValidity();return;
 }
 const row=structuredClone(defaultCell);
 row.name=name;row.source.planes=1;row.source.clusters_per_plane=Number(addForm.elements.cellCount.value);
 if(row.source.type==='walker_delta')row.source.walker_phase=0;
 if(!config.constellations.communication_shells.length)row.sgl_terminals=Math.max(1,row.sgl_terminals||0);
 delete row.cell_overrides;delete row.compute_parameters;
 config.constellations.compute_formations.push(row);addDialog.close();render();
 $('cells').lastElementChild.scrollIntoView({block:'start'});
};
const computeTab=document.querySelector('[data-tab="compute"]');
function showComputeTab(){
 computeTab.click();
 document.querySelector('[data-panel="compute"]').scrollIntoView({block:'start'});
}
$('addGround').onclick=()=>{config.ground_ingresses.push({ingress_id:'ground-'+Date.now(),lat_deg:0,lon_deg:0,...groundDefaults});render()};
function physicalFormState(){
  return JSON.stringify([$('name').value,$('computeMode').value,epStrategy,linkBudget.factors(),config]);
}
async function save(next){
  try{
    const invalid=document.querySelector('main input:invalid:not(dialog input),main select:invalid:not(dialog select),main textarea:invalid');
    if(invalid){invalid.reportValidity();throw Error('请先修正无效输入')}
    if(!inferredScenario(config))throw Error('请先添加通信壳层或计算编队。');
    if(next===true&&!config.constellations.compute_formations.length)throw Error('可以保存和预览通信拓扑。要运行推理，请添加计算编队。');
    if(next===true&&preparedExample?.canReuse&&preparedExample.formState===physicalFormState()){
      location.assign('/scene?model='+encodeURIComponent(preparedExample.modelId));
      return;
    }
    if(config.network.optical_link_model==='islsim'){
      for(const kind of opticalKinds){
        if(!bindISLSim(config.network.link_profiles[kind]))throw Error('ISLSim 参数未加载，请检查服务后重试。');
      }
    }
    saving=true;updateActions();
    const doc=await post('/save-physical',{name:$('name').value,...physicalPayload(config,$('computeMode').value,epStrategy,capabilities,linkBudget.factors())});
    selectLoadedSource(addSavedModel(doc,true));
    history.replaceState(null,'','/physical?id='+doc.id+(sceneDraft?'&scene_draft='+encodeURIComponent(sceneDraft):'')+location.hash);
    $('status').textContent='物理模型已保存到本地：'+doc.name;
    if(next){
      $('status').textContent='正在生成完整物理数据：全时域位置、链路与双向预算…';
      const job=await post('/prepare-physical',{id:doc.id,independent:true});
      location.href='/runs/'+job.id+'/progress'+(next===true?'?next=scene&model='+encodeURIComponent(doc.id)
        +(sceneDraft?'&scene_draft='+encodeURIComponent(sceneDraft):''):'');
    }
  }catch(e){$('status').textContent=e.message}
  finally{saving=false;updateActions()}
}
$('save').onclick=()=>save(false);$('next').onclick=()=>save(true);$('preview').onclick=()=>save('preview');
(async()=>{
 const catalog=get('/physical-catalog').catch(error=>{
   $('savedModels').firstElementChild.textContent='读取配置失败：'+error.message;
   return [];
 });
 capabilities=await get('/cell/api/capabilities');$('computeMode').replaceChildren(...capabilities.compute_modes.map(row=>new Option(row.name,row.id)));$('computeMode').value='detail_mode';
 await initISLSim();radioProfiles=await get('/radio-profiles');const initial=newPhysicalPreset(await get('/templates/starlink_550'));
 defaultShell=structuredClone(initial.constellations.communication_shells[0]);defaultCell=structuredClone(initial.constellations.compute_formations[0]);
 const models=await catalog;
 const preset=models.find(model=>model.case_study_iii_scene_id);
 const presetSource='?example=case_study_iii';
 if(preset)$('examples').append(new Option('Case Study III（600s 仿真）',presetSource));
 for(const model of models)if(model!==preset)addSavedModel(model);
 const fromPreset=sceneQuery.get('example')==='case_study_iii';
 const id=fromPreset?preset?.id:sceneQuery.get('id');
 if(fromPreset&&!id)throw Error('尚未安装 Case Study III 示例数据');
 if(id){const doc=await get('/documents/physical/'+encodeURIComponent(id));loadPhysicalCompute(doc);config=doc.physical_network;$('name').value=doc.name;
 selectLoadedSource(preset?.id===id?presetSource:models.some(model=>model.id===id)?'?id='+encodeURIComponent(id):addSavedModel(doc,true));
 $('computeMode').value=doc.compute_configuration?.mode||'detail_mode';epStrategy=doc.compute_configuration?.ep_strategy||'auto';linkBudget.load(doc.compute_configuration?.link_bandwidth_factors);
 const saved=sceneDraft && sessionStorage.getItem('orbitserve-scene-geometry-'+sceneDraft);
 const draft=saved && JSON.parse(saved);
 if(draft && draft.physical_model_id===doc.id){
  $('computeMode').value=draft.mode;epStrategy=draft.ep_strategy;
  for(const formation of config.constellations?.compute_formations||[]) for(const key of cellKeys(formation)){
   const layout=draft.cell_parameters?.[formation.name+'-'+key];
   if(layout){
    const members=memberIds(formation,key,cellGeometry(formation,key));
    const effective=cellLayout(formation,key,members)||sceneCellLayout(sceneCellDraft(capabilities,members,draft.mode),capabilities,members);
    if(JSON.stringify(layout)!==JSON.stringify(effective)){
     const settings=formation.compute_parameters??={cells:{}};settings.cells[key]=layout;
    }
   }
  }
 }}else{
   const example=sceneQuery.get('example')||'starlink_550';
   const option=Array.from($('examples').children).find(row=>row.value==='?example='+example);
   if(!option)throw Error('未知的物理场景示例');
   config=example==='starlink_550'?initial:newPhysicalPreset(await get('/templates/'+example));
   selectLoadedSource(option.value);$('name').value=option.textContent;
 }
 const migrated=initializeLinkBudgets(config,radioProfiles,{newModel:!id});
 if(id && migrated)$('budgetMigration').hidden=false;
 if(preset?.id===id&&!sceneDraft&&!selectedCell)preparedExample={modelId:id,canReuse:!migrated};
 // New shells use the scene budgets, including shells added after the initial render.
 delete defaultShell.hardware_link_profiles;
 delete defaultShell.isl_profile;
 render();modelSource.disabled=$('addCell').disabled=false;
 if(preparedExample){
   preparedExample.formState=physicalFormState();
   const entry=$('caseStudyExperiment');
   entry.querySelector('a').href='/scene?id='+encodeURIComponent(preset.case_study_iii_scene_id);
   entry.hidden=false;
 }
 initCommunicationDialog(()=>config,defaultShell,render);
 initGroundImport(text=>{
   const rows=parseGroundCsv(text,config.ground_ingresses);
   config.ground_ingresses=config.ground_ingresses.concat(rows);
   render();return rows.length;
 });
 $('groundFile').disabled=$('addGround').disabled=false;
 for(const tab of document.querySelectorAll('[data-tab]'))tab.disabled=false;
 if(location.hash==='#links')document.querySelector('[data-tab="links"]').click();
 if(location.hash==='#cells'||sceneDraft||selectedCell)showComputeTab();
 if(sceneDraft||selectedCell){
  const match=selectedFormation();
  const card=match&&[...$('cells').children].find(node=>node.dataset.formationName===match.name);
  (card||$('cells')).scrollIntoView({block:'start'});
  const edit=card && [...card.querySelectorAll('.formation-cell-row button')].find(button=>button.getAttribute('aria-label')==='编辑编队 '+selectedCell);
  edit?.click();
  if(sceneDraft)$('status').textContent=match
   ? '已定位 '+selectedCell+'。点击「编辑编队」修改，生成数据后返回场景。'
   : '请编辑目标算力编队，生成新物理数据后会返回当前场景草稿。';
 }
})().catch(e=>$('status').textContent=e.message);
