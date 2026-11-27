import {modelDisplayName} from '../cell/config-labels.mjs';
import {renderConfigProfile} from './config-profile.js';
import {get,post} from '/assets/studio_common.js';
import {normalizeRandomFaults,renderRandomFaults} from '/assets/random_faults.js';
import {createCellTopology} from '/assets/cell_topology.js';
import {createRequestBatch} from '/assets/request_batch.js';
import {startBatchExperiment} from '/assets/experiment_flow.js';
import {deploymentFromPhysical,removeLegacyCellWorkload} from './physical-cells.js';
import {layoutFromExperiment, preserveOtherCellTopologies, sceneCellDraft, sceneCellLayout} from './scene-cell-parameters.js';
const $=id=>document.getElementById(id);
let experiment=null,modelId=null,physicalTime={},runtime={},busy=false,needsBuild=true,ready=false;
let validatedSignature='';
let capabilities=null;
let membersByCell={}, cellParameters={}, effectiveRequestKey='';
let physicalOptions=null, cellNames={}, linkBandwidthFactors;
const cellCaption=id=>cellNames[id]||id;
const restoreKey='orbitserve-scene-geometry-';
let requestPage=0;
const requestPageSize=25;
let initialTopologyOwner=new URLSearchParams(location.search).get('satellite');
const signature=()=>JSON.stringify(payload());
const number=id=>Number($(id).value);
const topologySection=document.createElement('section');topologySection.className='card topology-panel';topologySection.hidden=true;
const topologyHeading=document.createElement('h2');topologyHeading.textContent='计算节点拓扑';const topologyHost=document.createElement('div');topologySection.append(topologyHeading,topologyHost);$('requests').closest('section').before(topologySection);
const topologyEditor=createCellTopology(topologyHost);
function autoContextCells(){
 return Object.entries(cellParameters).filter(([,layout])=>{
  const draft=layout.configuration;
  const limit=draft
   ? draft.scheduler[draft.deployment.worker_pools[0].name].vllm.max_model_len
   : layout.deployment.max_model_len;
  return limit==null;
 }).map(([id])=>id);
}
const requestBatch=createRequestBatch({getExperiment:()=>experiment,autoContextCells,apply:requests=>{experiment.requests=requests;requestPage=0;validatedSignature='';renderRequests();writeJson();state();$('status').textContent=`已应用 ${requests.length.toLocaleString()} 条业务请求`}});
for(const [id,kind]of [['importRequests','import'],['generateRequests','generate']])$(id).onclick=()=>{try{requestBatch.open(kind)}catch(error){$('status').textContent=error.message}};
function state(){
 if($('cellSelection').dataset.mode!==$('computeMode').value)setCellSelection(experiment?.cell_selection);
 $('runtimeConfig').disabled=!ready; $('sceneFields').disabled=busy;
 $('dataset').disabled=busy; $('name').disabled=busy;
 for(const id of ['save','run','liveRun'])$(id).disabled=busy||!experiment||needsBuild;
 if(experiment?.fast_orbital_route_frames){
  $('liveRun').disabled=true;
  $('liveRun').title='已导入的路由帧用于后台实验，完成后可查看场景回放。';
 }else $('liveRun').removeAttribute('title');
 $('buildStatus').textContent=needsBuild?'部署参数待生成':'';
 $('build').hidden=!needsBuild;

 for(const id of ['addRequest','importRequests','generateRequests'])$(id).disabled=!experiment||needsBuild||busy;
}
const resourceKeys=['physical','physical_link_bindings','ground_owners','satellite_roles','physical_dataset','physical_network'];
function writeJson(){
 const config=experiment && Object.fromEntries(Object.entries(experiment).filter(([key])=>!resourceKeys.includes(key)));
 renderConfigProfile($('json'),config,{cells:'算力编队',requests:'业务请求',communication:'通信',routing:'路由',migrations:'KV 迁移',cell_selection:'Cell 选择策略'},experiment?[
  ['仿真时长',experiment.horizon_ns/1e9+' 秒'],['算力编队',experiment.cells.length+' 个'],
  ['业务请求',experiment.requests.length+' 条'],['最多重试次数',experiment.communication?.retry_limit??8],
 ]:[]);
 if(validatedSignature&&validatedSignature!==signature())$('jsonStatus').textContent='配置已修改，启动前将重新校验。';
}
function payload(){return{name:$('name').value.trim(),physical_model_id:modelId,experiment,cell_parameters:cellParameters}}
function validFields(){
 for(const input of document.querySelectorAll('input,select')){
  if(input.closest('dialog')||input.disabled||input.checkValidity())continue;
  input.reportValidity();throw Error(input.validationMessage||'请修正无效的表单参数');
 }
 if(!$('name').value.trim())throw Error('请填写场景名称');
}
function initializeCatalog(value){
 capabilities=value;
 $('computeMode').replaceChildren(...value.compute_modes.map(row=>option(row.id,row.name)));
}
function renderCellList(){
 const root=$('cellDeploymentList');const table=document.createElement('table');
 const head=table.createTHead().insertRow();
 for(const text of ['Cell','成员','GPU 部署']){const cell=document.createElement('th');cell.textContent=text;head.append(cell)}
 const body=table.createTBody();
 for(const [id,members] of Object.entries(membersByCell)){
  const row=body.insertRow();
  row.insertCell().textContent=cellCaption(id);
  row.insertCell().textContent=members.length+' 颗卫星';
  const cell=experiment?.cells.find(row=>row.id===id), options=cellParameters[id]?.deployment;
  const model=options?.model||cell?.model;
  const hardware=options?.hardware_id||cell?.hardware_id;
  const deployment=row.insertCell();
  const parts=model?[
   capabilities.su_profiles[hardware]?.name||hardware, (options?.gpus_per_satellite??1)+' 卡/星',modelDisplayName(model.split('/').at(-1).replace(/\.json$/,'')),
   cell?'TP '+cell.parallel.tp+' / PP '+cell.parallel.pp+' / DP '+cell.parallel.dp:'',
  ].filter(Boolean):['待生成默认部署'];
  parts.forEach((part,index)=>{
   const label=document.createElement('span');label.textContent=part;
   deployment.append(index ? ' · ' : '',label);
  });
 }
 root.replaceChildren(table);
}
$('physicalLink').onclick=event=>{
 if(!modelId)return;
 event.preventDefault();
 const returnId=crypto.randomUUID();
 try{
  sessionStorage.setItem(restoreKey+returnId,JSON.stringify({name:$('name').value,experiment,
   physical_model_id:modelId,cell_parameters:cellParameters,mode:$('computeMode').value,
   ep_strategy:$('epStrategy').value}));
 }catch(error){
  $('status').textContent='浏览器草稿空间不足，请先保存场景后从物理模型页编辑。';
  return;
 }
 const target=new URL($('physicalLink').href);
 target.searchParams.set('scene_draft',returnId);
 target.hash='cells';
 location.assign(target.href);
};
const metricNames={load:'正在处理的请求最少',distance:'传输路径最短',arrival:'预计数据最早到达'};
$('cellPrimary').replaceChildren(...Object.entries(metricNames).map(([id,name])=>option(id,name)));
function updateSecondary(preferred){
 const choices=Object.entries(metricNames).filter(([id])=>id!==$('cellPrimary').value);
 $('cellSecondary').replaceChildren(option('','直接按编队编号决定'),...choices.map(([id,name])=>option(id,name)));
 $('cellSecondary').value=choices.some(([id])=>id===preferred)?preferred:'';
}
updateSecondary('distance');
const legacyPriorities={
 least_loaded:{primary:'load',secondary:'distance'},
 nearest:{primary:'distance',secondary:'load'},
 estimated_arrival:{primary:'load',secondary:'arrival'},
};
const detailedCellChoices=[['least_loaded','负载优先'],['nearest','距离优先'],
 ['estimated_arrival','负载相同时优先选传输更快的编队'],['custom','自定义优先顺序']];
const fastCellChoices=[['custom','自定义优先顺序'],['nearest_per_turn','最近距离'],
 ['least_loaded_per_turn','最小负载'],['session_fixed','会话固定']];
function setCellSelection(policy){
 const select=$('cellSelection'),mode=$('computeMode').value,fast=mode==='fast_mode';
 const choices=fast?fastCellChoices:detailedCellChoices;
 if(select.dataset.mode!==mode){
  select.replaceChildren(...choices.map(([value,label])=>option(value,label)));
  select.dataset.mode=mode;
 }
 const preserved=fast&&typeof policy==='string'&&Object.hasOwn(legacyPriorities,policy)?legacyPriorities[policy]:policy;
 if(preserved&&typeof preserved==='object'){
  select.value='custom';$('cellPrimary').value=preserved.primary;updateSecondary(preserved.secondary);
 }else{
  select.value=choices.some(([value])=>value===preserved)?preserved:fast?'nearest_per_turn':'least_loaded';
 }
 showPriority();
}
function selectionPolicy(){
 return $('cellSelection').value==='custom'
   ? {primary:$('cellPrimary').value,secondary:$('cellSecondary').value||null}
   : $('cellSelection').value;
}
function showPriority(){ $('cellPriority').hidden=$('cellSelection').value!=='custom'; }
function changePriority(){showPriority();if(experiment){experiment.cell_selection=selectionPolicy();writeJson()}}
$('cellPrimary').onchange=()=>{updateSecondary($('cellSecondary').value);changePriority()};
$('cellSecondary').onchange=changePriority;

function syncRuntime(){delete runtime.failure_policy;delete runtime.max_transfer_retries;if(experiment){experiment.communication??={};experiment.communication.retry_limit=number('retryCount');}runtime.fault_timing='continuous';runtime.faults??=[];if(experiment){experiment.runtime_network=runtime;experiment.routing=routingConfig();experiment.cell_selection=selectionPolicy()}}
function routingConfig(){const selected=$('routingPolicy').value;const policy=selected==='custom'?$('customRoutingPolicy').value.trim():selected;if(!/^[a-z][a-z0-9_-]{0,127}$/.test(policy))throw Error('请填写有效的已注册策略名称');return {policy}}
$('cellSelection').onchange=changePriority;
function reflectRouting(){setCellSelection(experiment?.cell_selection);const policy=experiment?.routing?.policy||'min_hops';const custom=!['min_hops','min_propagation'].includes(policy);$('routingPolicy').value=custom?'custom':policy;$('customRoutingField').hidden=!custom;$('customRoutingPolicy').required=custom;$('customRoutingPolicy').value=custom?policy:''}
$('routingPolicy').onchange=$('customRoutingPolicy').oninput=()=>{const custom=$('routingPolicy').value==='custom';$('customRoutingField').hidden=!custom;$('customRoutingPolicy').required=custom;try{if(experiment){experiment.routing=routingConfig();writeJson()}else routingConfig();$('status').textContent=''}catch(error){$('status').textContent=error.message}};
function renderRuntime(){runtime.faults??=[];runtime.random_faults=normalizeRandomFaults(runtime.random_faults,physicalTime);$('retryCount').value=experiment?.communication?.retry_limit??8;renderRandomFaults($('runtimeFaults'),runtime.random_faults,physicalTime)}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o}
function renderRequests(){
 const host=$('requests');host.replaceChildren();$('requestEmpty').hidden=!!experiment?.requests.length;
 const count=experiment?.requests.length||0;requestPage=Math.min(requestPage,Math.max(0,Math.ceil(count/requestPageSize)-1));$('requestCount').textContent=`${count.toLocaleString()} 条请求`;
 $('requestPagination').hidden=count<=requestPageSize;$('previousRequestPage').disabled=requestPage===0;$('nextRequestPage').disabled=(requestPage+1)*requestPageSize>=count;$('requestPage').textContent=`${requestPage*requestPageSize+1}–${Math.min(count,(requestPage+1)*requestPageSize)} / ${count.toLocaleString()}`;
 $('requestEmpty').textContent=experiment?'暂无业务请求。':'生成部署配置后添加业务请求。';
 if(!experiment)return;
 const ground=new Set(experiment.ground_owners),endpoints=experiment.physical.devices.filter(d=>ground.has(d.satellite));
 const offset=requestPage*requestPageSize;
 for(const [localIndex,request]of experiment.requests.slice(offset,offset+requestPageSize).entries()){
  const index=offset+localIndex;
  const row=document.createElement('tr');
  for(const key of ['id','user_device','cell','arrival_ns','input_tokens','output_tokens']){
   const td=document.createElement('td'),select=['user_device','cell'].includes(key),input=document.createElement(select?'select':'input');
   input.setAttribute('aria-label',`${request.id} ${key}`);
   if(select){if(key==='cell'){input.append(option('','自动选择'));for(const id of Object.keys(membersByCell))input.append(option(id,cellCaption(id)))}else for(const d of endpoints)input.append(option(d.id,d.satellite));input.value=request[key]||''}
   else{input.type=key==='id'?'text':'number';input.required=true;if(key!=='id'){input.min=key==='arrival_ns'?'0':'1';input.step=key==='arrival_ns'?'any':'1';if(key==='arrival_ns')input.max=String(experiment.horizon_ns/1e9-1e-9)}input.value=key==='arrival_ns'?request[key]/1e9:request[key]}
   input.oninput=()=>{request[key]=key==='id'?input.value:key==='cell'?input.value||null:key==='user_device'?input.value:key==='arrival_ns'?Math.round(Number(input.value)*1e9):Number(input.value);writeJson()};td.append(input);row.append(td);
  }
  const td=document.createElement('td'),remove=document.createElement('button');remove.type='button';remove.textContent='移除';remove.onclick=()=>{experiment.requests.splice(index,1);renderRequests();writeJson()};td.append(remove);row.append(td);host.append(row);
 }
}
function changeRequestPage(delta){for(const input of $('requests').querySelectorAll('input,select'))if(!input.checkValidity()){input.reportValidity();return}requestPage+=delta;renderRequests()}
$('previousRequestPage').onclick=()=>changeRequestPage(-1);$('nextRequestPage').onclick=()=>changeRequestPage(1);
function reflectConfig(){
 reflectRouting();
 $('deploymentSummary').textContent=experiment.cells.length+' 个 Cell · '+experiment.physical.devices.filter(d=>d.kind==='gpu').length+' 张 GPU';
 runtime=experiment.runtime_network||{};renderRuntime();renderRequests();writeJson();renderCellList();state();
 topologySection.hidden=false;topologyEditor.update(experiment,null,initialTopologyOwner);initialTopologyOwner=null;
}
async function loadDataset(id,usePhysical=true){
 ready=false;state();$('physicalLink').href='/physical?id='+encodeURIComponent(id);
 const data=await get('/physical-data/'+encodeURIComponent(id));if(!data.ready)throw Error(data.reason||'物理数据尚未就绪，请先完成物理模型生成');
 const [doc,catalog]=await Promise.all([get('/documents/physical/'+encodeURIComponent(id)),get('/native-scene-catalog?physical_model_id='+encodeURIComponent(id))]);
 physicalTime=doc.physical_network.time;membersByCell=catalog.cell_satellites;
 physicalOptions=doc.compute_configuration;
 linkBandwidthFactors=usePhysical
  ?physicalOptions?.link_bandwidth_factors
  :experiment?.communication?.link_bandwidth_factors??physicalOptions?.link_bandwidth_factors;
 cellNames=Object.fromEntries(Object.entries(physicalOptions?.formations||{}).flatMap(([name,settings])=>
   Object.entries(settings.cell_names||{}).map(([key,caption])=>[name+'-'+key,caption])));
 if(usePhysical){
  cellParameters=deploymentFromPhysical(doc,membersByCell);
  $('computeMode').value=physicalOptions?.mode||'detail_mode';$('epStrategy').value=physicalOptions?.ep_strategy||'auto';
 }
 modelId=id;$('dataset').value=id;
 $('datasetStatus').textContent='物理拓扑已就绪 / '+physicalTime.duration_s+' s';ready=true;renderCellList();renderRuntime();state();
}
function requestSchedulingKey(){
 return JSON.stringify(experiment?.requests.map(row=>[row.cell,row.input_tokens,row.output_tokens])||[]);
}
async function build(updated=cellParameters, changed=new Set(Object.keys(membersByCell))){
 validFields();syncRuntime();
 const mode=$('computeMode').value;
 const submitted=Object.fromEntries(Object.entries(membersByCell).map(([id,members])=>{
  const saved=updated[id];
  const sameMembers=saved&&members.length===Object.keys(saved.satellite_gpus).length&&members.every(sat=>Object.hasOwn(saved.satellite_gpus,sat));
  if(sameMembers && (!saved.configuration || saved.configuration.compute_mode===mode)){
   const row=structuredClone(saved);
   if(mode!=='fast_mode'){
    row.deployment.memory_policy_file=null;row.deployment.fast_glm_dsa_score=null;
   }
   return [id,row];
  }
  const draft=sceneCellDraft(capabilities,members,mode,saved);
  return [id,{...saved,...sceneCellLayout(draft,capabilities,members)}];
 }));
 const communication={...experiment?.communication,retry_limit:number('retryCount')};
 delete communication.cell_policies;
 if(mode==='fast_mode')communication.link_bandwidth_factors=linkBandwidthFactors??{};
 else delete communication.link_bandwidth_factors;
 if($('epStrategy').value==='auto')delete communication.ep_strategy;
 else communication.ep_strategy=$('epStrategy').value;
 const value=await post('/native-scene-config',{physical_model_id:modelId,deployment:{compute_mode:mode,communication},
  cell_deployments:submitted,requests:experiment?.requests||[],migrations:experiment?.migrations||[],
  runtime_network:runtime,routing:routingConfig(),cell_selection:selectionPolicy()});
 experiment=experiment?preserveOtherCellTopologies(experiment,value,changed):value;
 cellParameters=submitted;needsBuild=false;effectiveRequestKey=requestSchedulingKey();validatedSignature='';
 reflectConfig();$('legacyNotice').hidden=true;$('build').textContent='重新生成 GPU 部署';$('status').textContent='GPU 部署已生成，可添加业务请求。';
}
$('build').onclick=()=>perform(()=>build());
for(const id of ['computeMode','epStrategy'])$(id).onchange=()=>{needsBuild=true;validatedSignature='';state()};
$('addRequest').onclick=()=>{const owners=new Set(experiment.ground_owners),device=experiment.physical.devices.find(d=>owners.has(d.satellite));let i=1;while(experiment.requests.some(r=>r.id==='request-'+i))i++;experiment.requests.push({id:'request-'+i,arrival_ns:0,user_device:device.id,input_tokens:32,output_tokens:4,input_bytes:128,output_bytes_per_token:4,cell:null});requestPage=Math.floor((experiment.requests.length-1)/requestPageSize);renderRequests();writeJson()};
$('retryCount').oninput=()=>{try{syncRuntime();writeJson();$('status').textContent=''}catch(e){$('status').textContent='重试配置：'+e.message}};
document.addEventListener('modelchange',()=>{if(experiment){experiment.runtime_network=runtime;writeJson()}});
async function validate(){
 if(needsBuild)throw Error('部署参数已修改，请先生成 GPU 部署配置');validFields();syncRuntime();
 if(Object.keys(cellParameters).length&&effectiveRequestKey!==requestSchedulingKey())await build(cellParameters,new Set());
 writeJson();
 if(validatedSignature===signature())return experiment;
 $('jsonStatus').textContent='正在校验通信与部署…';
 const response=await post('/validate-scene',payload());
 experiment=response.document.experiment;reflectConfig();validatedSignature=signature();$('jsonStatus').textContent='校验通过 配置未修改时可直接启动';return experiment;
}
async function perform(action,rethrow=false){if(busy){if(rethrow)throw Error('请等待当前操作完成');return}try{$('status').textContent='';validFields();busy=true;state();return await action()}catch(e){$('status').textContent=e.message;if(rethrow)throw e}finally{busy=false;state()}}
async function save(kind){
 await validate();
 $('status').textContent=kind?'正在提交实验…':'正在保存…';
 const doc=await post('/save-scene',payload());
 history.replaceState(null,'','/scene?id='+doc.id);
 $('status').textContent='场景已保存';
 if(kind==='batch'){
  $('status').textContent='正在启动后台实验，随后进入运行进度…';
  await startBatchExperiment(post,doc.id,url=>location.assign(url));
 }else if(kind==='live'){
  const job=await post('/live/start',{scene_id:doc.id});
  location.assign(job.url+'?autostart=1');
 }
}
$('save').onclick=()=>perform(()=>save());$('run').onclick=()=>perform(()=>save('batch'));$('liveRun').onclick=()=>perform(()=>save('live'));
$('dataset').onchange=()=>perform(async()=>{experiment=null;needsBuild=true;runtime={};cellParameters={};membersByCell={};writeJson();topologySection.hidden=true;topologyEditor.clear();renderRequests();const id=$('dataset').value;if(!id){ready=false;return}await loadDataset(id);await build()});
(async()=>{
 const [datasets,catalog]=await Promise.all([get('/physical-datasets'),get('/cell/api/capabilities')]);
 initializeCatalog(catalog);for(const d of datasets)$('dataset').append(option(d.id,d.name));
 const q=new URLSearchParams(location.search);let doc=null;
 const draftId=q.get('scene_draft');
 const restored=draftId&&sessionStorage.getItem(restoreKey+draftId);
 if(restored)doc=JSON.parse(restored);
 else if(q.has('id'))doc=await get('/documents/scenes/'+encodeURIComponent(q.get('id')));
 else if(q.has('run'))doc={experiment:await get('/runs/'+encodeURIComponent(q.get('run'))+'/input'),name:'复用场景'};
 if(doc){
  $('name').value=doc.name;
  modelId=q.get('model')||doc.physical_model_id;
  if(!modelId&&doc.experiment?.physical_dataset)for(const d of datasets){
   const info=await get('/physical-data/'+encodeURIComponent(d.id));if(info.physical_dataset===doc.experiment.physical_dataset){modelId=d.id;break}
  }
  if([1,'orbitserve.native_cells.v1'].includes(doc.experiment?.schema_version))experiment=doc.experiment;
  else if(doc.experiment){
   $('legacyNotice').hidden=false;
   $('legacyDownload').href=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));
   runtime=structuredClone(doc.experiment.physical_network?.network||{});
   for(const key of Object.keys(runtime))if(!['faults','random_faults','fault_timing','failure_policy','max_transfer_retries'].includes(key))delete runtime[key];
  }
  cellParameters=doc.cell_parameters||{};
  Object.values(cellParameters).forEach(removeLegacyCellWorkload);
 }
 modelId??=q.get('model');
 const revised=!!restored && doc.physical_model_id!==modelId;
 if(modelId)await loadDataset(modelId,!doc || revised);else $('datasetStatus').textContent=datasets.length?'选择物理数据后配置实验。':'暂无完整物理数据，请先生成物理模型。';
 const mode=(revised||!doc)?physicalOptions?.mode:doc?.mode||experiment?.compute_mode;
 $('computeMode').value=['fast_mode','batch_roofline'].includes(mode)?'fast_mode':'detail_mode';
 $('epStrategy').value=((revised||!doc)?physicalOptions?.ep_strategy:doc?.ep_strategy||experiment?.communication?.ep_strategy)||'auto';
 if(experiment){
  const sameTopology=!restored||doc.physical_model_id===modelId;
  if(sameTopology)for(const cell of experiment.cells){
   if(!membersByCell[cell.id])continue;
   const saved=cellParameters[cell.id];
   if(!saved){
    cellParameters[cell.id]=layoutFromExperiment(experiment,cell,membersByCell[cell.id]);
   }else if(cell.fast_kv_pool_blocks!=null){
    const settings=saved.configuration??saved.deployment;
    if(!Object.hasOwn(settings,'fast_kv_pool_blocks'))settings.fast_kv_pool_blocks=cell.fast_kv_pool_blocks;
   }
  }
  // Previously saved scenes carry effective fields. Only form-owned fields are posted on future builds.
  if(!revised && !doc?.cell_parameters)for(const [id,layout] of Object.entries(cellParameters)){
   const draft=sceneCellDraft(capabilities,membersByCell[id],$('computeMode').value,layout);
   cellParameters[id]=sceneCellLayout(draft,capabilities,membersByCell[id]);
  }
  cellParameters=Object.fromEntries(Object.entries(cellParameters).filter(([id])=>membersByCell[id]));
  needsBuild=!sameTopology;effectiveRequestKey=requestSchedulingKey();
  reflectConfig();if(q.has('satellite'))topologySection.scrollIntoView({block:'start'});
 }
 if(ready && (!experiment||revised))await perform(()=>build(),true);
 if(restored)sessionStorage.removeItem(restoreKey+draftId);
 state();
})().catch(e=>{$('status').textContent=e.message;state()});
