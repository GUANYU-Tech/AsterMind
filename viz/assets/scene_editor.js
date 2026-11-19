import {get,post} from '/assets/studio_common.js';
import {normalizeRandomFaults,renderRandomFaults} from '/assets/random_faults.js';
import {createCellTopology} from '/assets/cell_topology.js';
import {createRequestBatch} from '/assets/request_batch.js';
import {startBatchExperiment} from '/assets/experiment_flow.js';
const $=id=>document.getElementById(id);
let experiment=null,modelId=null,physicalTime={},runtime={},jsonDirty=false,baseline='',busy=false,needsBuild=true,ready=false;
let validatedSignature='';
let capabilities=null;
let minCellMembers=Infinity;
let requestPage=0;
const requestPageSize=25;
let initialTopologyOwner=new URLSearchParams(location.search).get('satellite');
const signature=()=>JSON.stringify(payload());
const deploymentFields=['model','hardware','gpus','parallel','concurrency','context','gpuBandwidth','gpuLatency','batchTokens','kvBlocks','expertSeed'];
const number=id=>Number($(id).value);
const topologySection=document.createElement('section');topologySection.className='card';topologySection.hidden=true;
const topologyHeading=document.createElement('h2');topologyHeading.textContent='计算节点拓扑';const topologyHost=document.createElement('div');topologySection.append(topologyHeading,topologyHost);$('requests').closest('section').before(topologySection);
const topologyEditor=createCellTopology(topologyHost,{edit:op=>perform(async()=>{if(jsonDirty||needsBuild)throw Error('请先应用 JSON 草稿或生成当前 GPU 部署');syncRuntime();const value=await post('/edit-scene-topology',{experiment,edit:op});experiment=value;validatedSignature='';reflectConfig();$('jsonStatus').textContent='拓扑已修改，启动前将重新校验。';return value},true)});
const requestBatch=createRequestBatch({getExperiment:()=>experiment,apply:requests=>{experiment.requests=requests;requestPage=0;validatedSignature='';renderRequests();writeJson();state();$('status').textContent=`已应用 ${requests.length.toLocaleString()} 条业务请求`}});
for(const [id,kind]of [['importRequests','import'],['generateRequests','generate']])$(id).onclick=()=>{try{requestBatch.open(kind)}catch(error){$('status').textContent=error.message}};
function state(){
 $('runtimeConfig').disabled=!ready; $('sceneFields').disabled=busy||jsonDirty;
 $('dataset').disabled=busy||jsonDirty; $('name').disabled=busy;
 for(const id of ['save','run','liveRun'])$(id).disabled=busy||(!jsonDirty&&(!experiment||needsBuild));
 $('apply').disabled=busy||!experiment;$('discardJson').hidden=!jsonDirty;
 $('buildStatus').textContent=needsBuild?'部署参数待生成':'';
 for(const id of ['addRequest','importRequests','generateRequests'])$(id).disabled=!experiment||needsBuild||jsonDirty||busy;
}
const resourceKeys=['physical','physical_link_bindings','ground_owners','satellite_roles'];
function writeJson(){if(jsonDirty)return;const editable=experiment?Object.fromEntries(Object.entries(experiment).filter(([key])=>!resourceKeys.includes(key))):null;baseline=editable?JSON.stringify(editable,null,2):'';$('json').value=baseline;if(validatedSignature&&validatedSignature!==signature())$('jsonStatus').textContent='配置已修改，启动前将重新校验。'}
function payload(){return{name:$('name').value.trim(),physical_model_id:modelId,experiment}}
const selectedModel=()=>capabilities?.models.find(row=>row.id===$('model').value);
const choices=()=>selectedModel()?.choices[$('hardware').value]||[];
const fitsCell=row=>row.tp*row.pp<=minCellMembers*number('gpus');
function hardwareDefaults(){const row=capabilities.hardware.find(item=>item.id===$('hardware').value);if(row?.gpu_link_bandwidth_gbps!=null)$('gpuBandwidth').value=row.gpu_link_bandwidth_gbps;$('gpuLatency').value=row?.gpu_link_latency_ns??0}
function deploymentValidity(){const row=choices().find(row=>`${row.tp},${row.pp}`===$('parallel').value);$('parallel').setCustomValidity(!row?'当前模型、硬件与并行部署组合不受支持':!fitsCell(row)?'并行部署所需计算卡超过当前 Cell 容量':'')}
function refreshChoices(preferred=$('parallel').value,restore=false){
 const model=selectedModel(),hardware=$('hardware'),previousHardware=hardware.value;
 for(const item of hardware.options)item.disabled=!(model?.choices[item.value]?.length);
 if(!restore&&hardware.selectedOptions[0]?.disabled)hardware.value=[...hardware.options].find(item=>!item.disabled)?.value||'';
 if(!restore&&previousHardware!==hardware.value)hardwareDefaults();
 const rows=choices(),available=rows.filter(fitsCell);$('parallel').replaceChildren(...rows.map(row=>{const item=option(`${row.tp},${row.pp}`,row.expert_parallel?`TP ${row.tp} × EP ${row.tp}`:`TP ${row.tp} × PP ${row.pp}`);item.disabled=!fitsCell(row);if(item.disabled)item.textContent+=`（需 ${row.tp*row.pp} 卡）`;return item}));
 if(available.some(row=>`${row.tp},${row.pp}`===preferred))$('parallel').value=preferred;
 else if(restore){if(![...$('parallel').options].some(item=>item.value===preferred))$('parallel').append(option(preferred,preferred.replace(',',' × ')));$('parallel').value=preferred}
 else if(available.some(row=>row.tp===capabilities.defaults.tp&&row.pp===capabilities.defaults.pp))$('parallel').value=`${capabilities.defaults.tp},${capabilities.defaults.pp}`;
 else if(available.length)$('parallel').value=`${available[0].tp},${available[0].pp}`;
 $('expertSeedField').hidden=!model?.experts;$('expertSeed').disabled=!model?.experts;
 $('context').max=String(model?.max_context||131072);deploymentValidity();
}
function initializeCatalog(value){
 capabilities=value;
 $('model').replaceChildren(...value.models.map(model=>{const item=option(model.id,model.name+(model.available?' / BF16':' / 未接入'));item.disabled=!model.available;item.title=model.reason;return item}));
 $('hardware').replaceChildren(...value.hardware.map(row=>option(row.id,row.name)));
 $('model').value=value.defaults.model;$('hardware').value=value.defaults.hardware;refreshChoices(`${value.defaults.tp},${value.defaults.pp}`);hardwareDefaults();
}
function validFields(){deploymentValidity();for(const input of document.querySelectorAll('input,select'))if(!input.closest('dialog')&&!input.disabled&&!input.checkValidity()){input.reportValidity();throw Error(input.validationMessage||'请修正无效的表单参数')}if(!$('name').value.trim())throw Error('请填写场景名称')}
function deployment(){const [tp,pp]=$('parallel').value.split(',').map(Number),hardwareId=$('hardware').value,hardware=capabilities.hardware.find(row=>row.id===hardwareId)?.file,choice=choices().find(row=>row.tp===tp&&row.pp===pp);return{model:$('model').value,hardware,hardware_id:hardwareId,gpus_per_satellite:number('gpus'),tp,pp,expert_parallel:!!choice?.expert_parallel,expert_routing_seed:choice?.expert_parallel?number('expertSeed'):null,max_requests:number('concurrency'),max_model_len:number('context'),max_batched_tokens:number('batchTokens'),num_kv_blocks:number('kvBlocks'),gpu_link_bandwidth_gbps:number('gpuBandwidth'),gpu_link_latency_ns:number('gpuLatency'),compute_mode:'midlayer',communication:{retry_limit:number('retryCount'),intra_mode:'IntraSatelliteRoofline'}}}
function syncRuntime(){if($('failurePolicy').value!=='v1_retry')throw Error('V1.0 使用原生通信等待与重试；旧 failure_policy 不能直接映射。');delete runtime.failure_policy;delete runtime.max_transfer_retries;if(experiment){experiment.communication??={};experiment.communication.retry_limit=number('retryCount');}runtime.fault_timing='continuous';runtime.faults=JSON.parse($('faultList').value);if(experiment){experiment.runtime_network=runtime;experiment.routing=routingConfig();experiment.cell_selection=$('cellSelection').value}}
function routingConfig(){const selected=$('routingPolicy').value;const policy=selected==='custom'?$('customRoutingPolicy').value.trim():selected;if(!/^[a-z][a-z0-9_-]{0,127}$/.test(policy))throw Error('请填写有效的已注册策略名称');return {policy}}
$('cellSelection').onchange=()=>{if(experiment){experiment.cell_selection=$('cellSelection').value;writeJson()}};
function reflectRouting(){$('cellSelection').value=experiment?.cell_selection||'least_loaded';const policy=experiment?.routing?.policy||'min_hops';const custom=!['min_hops','min_propagation'].includes(policy);$('routingPolicy').value=custom?'custom':policy;$('customRoutingField').hidden=!custom;$('customRoutingPolicy').required=custom;$('customRoutingPolicy').value=custom?policy:''}
$('routingPolicy').onchange=$('customRoutingPolicy').oninput=()=>{const custom=$('routingPolicy').value==='custom';$('customRoutingField').hidden=!custom;$('customRoutingPolicy').required=custom;try{if(experiment){experiment.routing=routingConfig();writeJson()}else routingConfig();$('status').textContent=''}catch(error){$('status').textContent=error.message}};
function renderRuntime(){runtime.faults??=[];runtime.random_faults=normalizeRandomFaults(runtime.random_faults,physicalTime);$('failurePolicy').value='v1_retry';$('retryCount').value=experiment?.communication?.retry_limit??8;$('faultList').value=JSON.stringify(runtime.faults,null,2);renderRandomFaults($('runtimeFaults'),runtime.random_faults,physicalTime)}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o}
function renderRequests(){
 const host=$('requests');host.replaceChildren();$('requestEmpty').hidden=!!experiment?.requests.length;
 const count=experiment?.requests.length||0;requestPage=Math.min(requestPage,Math.max(0,Math.ceil(count/requestPageSize)-1));$('requestCount').textContent=`${count.toLocaleString()} 条`;
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
   if(select){if(key==='cell'){input.append(option('','自动选择'));for(const c of experiment.cells)input.append(option(c.id,c.id))}else for(const d of endpoints)input.append(option(d.id,d.satellite));input.value=request[key]||''}
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
 const c=experiment.cells[0];$('deploymentSummary').textContent=`${experiment.cells.length} 个 Cell · ${experiment.physical.devices.filter(d=>d.kind==='gpu').length} 个 GPU`;
 $('concurrency').value=c.max_requests;$('context').value=c.max_model_len;$('batchTokens').value=c.max_batched_tokens;$('kvBlocks').value=c.num_kv_blocks;$('expertSeed').value=c.expert_routing_seed??0;
 for(const [id,value]of [['model',c.model],['hardware',c.hardware_id]]){if(![...$(id).options].some(o=>o.value===value))$(id).append(option(value,value));$(id).value=value}
 const gpus=experiment.physical.devices.filter(d=>d.kind==='gpu'),owner=gpus[0]?.satellite;$('gpus').value=gpus.filter(d=>d.satellite===owner).length||1;
 refreshChoices(`${c.parallel.tp},${c.parallel.pp}`,true);
 const port=experiment.physical.links?.find(l=>l.source===gpus[0]?.id);if(port){$('gpuBandwidth').value=port.bandwidth_gbps;$('gpuLatency').value=port.latency_ns||0}
 runtime=experiment.runtime_network||{};renderRuntime();renderRequests();writeJson();needsBuild=false;state();
 topologySection.hidden=false;topologyEditor.update(experiment,null,initialTopologyOwner);initialTopologyOwner=null;
}
async function loadDataset(id){
 ready=false;state();$('physicalLink').href='/physical?id='+encodeURIComponent(id);
 const data=await get('/physical-data/'+encodeURIComponent(id));if(!data.ready)throw Error(data.reason||'物理数据尚未就绪，请先完成物理模型生成');
 const [doc,catalog]=await Promise.all([get('/documents/physical/'+id),get('/native-scene-catalog?physical_model_id='+encodeURIComponent(id))]);physicalTime=doc.physical_network.time;minCellMembers=Math.min(...Object.values(catalog.cell_members));refreshChoices();modelId=id;$('dataset').value=id;$('physicalLink').href='/physical?id='+id;
 $('datasetStatus').textContent=`物理拓扑已就绪 / ${physicalTime.duration_s} s`;ready=true;renderRuntime();state();
}
async function build(){
 if(jsonDirty)throw Error('请先应用或还原 JSON 草稿');validFields();syncRuntime();
 const value=await post('/native-scene-config',{physical_model_id:modelId,deployment:deployment(),requests:experiment?.requests||[],migrations:experiment?.migrations||[],runtime_network:runtime,routing:routingConfig(),cell_selection:$('cellSelection').value});
 experiment=value;reflectConfig();$('legacyNotice').hidden=true;$('build').textContent='重新生成 GPU 部署';$('status').textContent='GPU 部署已生成，可添加请求或编辑高级配置。';
}
$('build').onclick=()=>perform(build);
for(const id of deploymentFields)$(id).oninput=()=>{if(['model','hardware','gpus'].includes(id))refreshChoices();if(id==='hardware')hardwareDefaults();deploymentValidity();needsBuild=true;validatedSignature='';state()};
$('addRequest').onclick=()=>{const owners=new Set(experiment.ground_owners),device=experiment.physical.devices.find(d=>owners.has(d.satellite));let i=1;while(experiment.requests.some(r=>r.id==='request-'+i))i++;experiment.requests.push({id:'request-'+i,arrival_ns:0,user_device:device.id,input_tokens:32,output_tokens:4,input_bytes:128,output_bytes_per_token:4,cell:null});requestPage=Math.floor((experiment.requests.length-1)/requestPageSize);renderRequests();writeJson()};
$('failurePolicy').onchange=$('retryCount').oninput=$('faultList').oninput=()=>{try{syncRuntime();writeJson();$('status').textContent=''}catch(e){$('status').textContent='故障列表 JSON：'+e.message}};
document.addEventListener('modelchange',()=>{if(experiment){experiment.runtime_network=runtime;writeJson()}});
$('json').oninput=()=>{jsonDirty=$('json').value!==baseline;$('jsonStatus').textContent=jsonDirty?'草稿待校验。通过后会应用到场景。':'与当前配置一致。';state()};
$('discardJson').onclick=()=>{jsonDirty=false;writeJson();state();$('jsonStatus').textContent='草稿已还原。'};
async function validate(){
 if(!jsonDirty){if(needsBuild)throw Error('部署参数已修改，请先生成 GPU 部署配置');validFields();syncRuntime();writeJson()}
 if(!jsonDirty&&validatedSignature===signature())return experiment;
 const raw=$('json').value,draft=JSON.parse(raw);if(![1,'orbitserve.native_cells.v1'].includes(draft.schema_version))throw Error('此入口需要原生 GPU 场景配置');
 $('jsonStatus').textContent='正在检查物理数据、GPU 映射、模型、请求与通信范围…';
 const request=payload();delete request.experiment;request.experiment_json=raw;request.resource_context=Object.fromEntries(resourceKeys.map(key=>[key,experiment[key]]));
 const response=await post('/validate-scene',request);if($('json').value!==raw)throw Error('校验期间草稿改变，请重新校验');
 experiment=response.document.experiment;jsonDirty=false;reflectConfig();validatedSignature=signature();$('jsonStatus').textContent='校验通过 · 配置未修改时可直接启动';return experiment;
}
async function perform(action,rethrow=false){if(busy){if(rethrow)throw Error('请等待当前操作完成');return}try{$('status').textContent='';if(!jsonDirty)validFields();busy=true;state();return await action()}catch(e){$('status').textContent=e.message;$('jsonStatus').textContent=jsonDirty?'草稿未应用：'+e.message:$('jsonStatus').textContent;if(rethrow)throw e}finally{busy=false;state()}}
$('apply').onclick=()=>perform(validate);
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
$('dataset').onchange=()=>perform(async()=>{experiment=null;needsBuild=true;baseline='';runtime={};$('json').value='';topologySection.hidden=true;topologyEditor.clear();renderRequests();const id=$('dataset').value;if(!id){ready=false;return}await loadDataset(id)});
(async()=>{
 const [datasets,catalog]=await Promise.all([get('/physical-datasets'),get('/native-scene-catalog')]);initializeCatalog(catalog);for(const d of datasets)$('dataset').append(option(d.id,d.name));
 const q=new URLSearchParams(location.search);let doc=null;if(q.has('id'))doc=await get('/documents/scenes/'+q.get('id'));else if(q.has('run'))doc={experiment:await get('/runs/'+q.get('run')+'/input'),name:'复用场景'};
 if(doc){$('name').value=doc.name;modelId=doc.physical_model_id||q.get('model');if(!modelId&&doc.experiment.physical_dataset){for(const d of datasets){const info=await get('/physical-data/'+d.id);if(info.physical_dataset===doc.experiment.physical_dataset){modelId=d.id;break}}}
  if([1,'orbitserve.native_cells.v1'].includes(doc.experiment.schema_version))experiment=doc.experiment;
  else{$('legacyNotice').hidden=false;$('legacyDownload').href=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));runtime=structuredClone(doc.experiment.physical_network?.network||{});for(const key of Object.keys(runtime))if(!['faults','random_faults','fault_timing','failure_policy','max_transfer_retries'].includes(key))delete runtime[key]}}
 modelId??=q.get('model');if(modelId)await loadDataset(modelId);else $('datasetStatus').textContent=datasets.length?'选择物理数据后配置实验。':'暂无完整物理数据，请先生成物理模型。';
 if(experiment){reflectConfig();if(q.has('satellite'))topologySection.scrollIntoView({block:'start'})}state();
})().catch(e=>{$('status').textContent=e.message;state()});
