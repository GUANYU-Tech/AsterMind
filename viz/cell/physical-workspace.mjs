import {buildWorkspace,renderComputeSummary} from './workspace-ui.mjs?v=profile-1';
import {groups} from './config-layout.mjs';
import {renderConfigForm,parseConfiguration} from './config-form.mjs';
import {rememberEmptyLinkMode} from './link-mode.mjs';
import {renderParameterGroup,renderSchedulingReadout} from './parameter-panel.mjs';
import {editorChoices,editDependencies,formationPairs} from './config-edit.mjs';
import {configurationErrors} from './config-validation.mjs';
import {renderCellConfigProfile} from '../assets/config-profile.js';
import {mappingDraft} from './deployment-mapping.mjs';
import {modelOptions} from './compute-config.mjs';
import {synchronizeManualResources as synchronizeResources,clearDependentDrafts} from './resource-sync.mjs';
import {AsterMindScene,sceneState,syncSceneConfiguration} from './scene.js?v=profile-1';
import {sceneCellDraft,sceneCellLayout} from '../assets/scene-cell-parameters.js';
import {memberIds,inheritCellBudget} from '../assets/physical-cells.js';
import {formationNames,labels} from './config-labels.mjs';

const $=id=>document.getElementById(id);
const session=window.frameElement?.formationSession;
if(!session) throw Error('请从物理场景选择一个 Cell');
const {formation,key,capabilities,mode,geometry}=session;
const observed=formation.source.type==='tle';
const fixed=formation.cluster.motion_model==='reference_fixed_lvlh';
const members=()=>memberIds(formation,key,geometry);
let draft=sceneCellDraft(capabilities,members(),mode,session.layout);
// The full scene supplies the physical topology and accepts single-member Cells.
draft.topology.source='file';
draft.ep_strategy=session.epStrategy;
delete draft.fast_link_contention;
const profile=session.linkProfile || {};
const opticalMode=profile.capacity_model==='fixed'?'fixed':'islsim';
let explicitPairs=session.explicitPairs ?? (session.layout?.custom_links
  ? draft.topology.inter_satellite.links.map(({source,target})=>[source,target]) : null);
function inheritLinks() {
  rememberEmptyLinkMode(draft,opticalMode);
  const pairs=explicitPairs ?? formationPairs(members(),geo.layout??geo.topology);
  inheritCellBudget(draft,profile,pairs.map(([source,target])=>({source,target})));
}
const drafts=new Map();
const synthetic=!observed && !fixed;
const geo={satellite_count:members().length,[synthetic?'layout':'topology']:geometry.topology};
if(!observed)Object.assign(geo,{spacing_km:geometry.formation_pitch_m/1000,altitude_km:geometry.altitude_km});
const geoSchema={properties:{
  ...capabilities.schema.$defs.FormationConfig.properties,
  ...(fixed?{spacing_km:{type:'number',exclusiveMinimum:0},altitude_km:{type:'number',exclusiveMinimum:0}}:{}),
  satellite_count:{type:'integer',minimum:1},topology:{type:'string',enum:Object.keys(formationNames)},
}};
buildWorkspace({sceneCell:true});
const cellSchema={...capabilities.schema,required:capabilities.schema.required.filter(key=>key!=='requests')};
document.body.classList.add('physical-workspace');
document.querySelector('.studio-nav-label').textContent='算力编队编辑';
document.querySelector('.studio-nav-hint').textContent=session.id;
document.querySelectorAll('.studio-topbar a,.studio-sidefoot,.cell-overview,.cell-home,[data-stage-target="analysis"],.panel-close,#restore-defaults,#tle-input,#topology-preview').forEach(node=>node.remove());
document.querySelector('.studio-topbar').remove();
const section=(stage,index)=>$('form-'+stage+'-'+index);
const fields=[['formation',0,['hardware.default_hardware_id','hardware.accelerators_per_satellite']],
  ...Object.entries(groups).filter(([stage])=>stage!=='formation').flatMap(([stage,sections])=>
    sections.map(([,paths],index)=>[stage,index,paths.filter(path=>path!=='compute_mode')]).filter(([,index])=>section(stage,index)))];
// The physical scene owns all inter-satellite rates. Keep the intra-satellite fabric editable.
inheritLinks();
section('network',1).closest('details').remove();
section('network',2).closest('details').remove();
const budgetNote=document.createElement('p');
budgetNote.textContent='编队内星间链路使用「连接与链路预算」中的设置。';
section('network',0).closest('details').after(budgetNote);
$('run-simulation').remove();
const geometryHost=section('formation',1);
if(synthetic && geometry.motion_model!=='static_formation'){
  const note=document.createElement('p');note.className='upload-status';
  note.textContent='当前草稿采用网格排布。应用后，此 Cell 将按所选静态构型随参考轨道运动。';
  geometryHost.after(note);
}
const report=message=>session.status(message);
let jsonDirty=false;
function jsonLock(value){
  jsonDirty=value;
  for(const panel of document.querySelectorAll('.basic-fields'))panel.disabled=value;
  if(!value)renderParameters();
  else schedulePreflight();
}
$('advanced-config').oninput=()=>jsonLock(true);
$('discard-json').onclick=()=>jsonLock(false);
$('apply-json').onclick=()=>{try{
  const next=parseConfiguration($('advanced-config').value);
  if(Object.hasOwn(next,'requests'))throw Error('请在实验场景的「业务请求」中配置请求。');
  if(Object.hasOwn(next,'compute_mode') && next.compute_mode!==mode)throw Error('请在物理场景中统一设置计算模式。');
  next.compute_mode=mode;
  if(mode==='fast_mode' && capabilities.fast_roofline_defaults)next.fast_roofline??=structuredClone(capabilities.fast_roofline_defaults);
  if(JSON.stringify(next.hardware.satellite_ids)!==JSON.stringify(members()) || next.topology.source!=='file' ||
     JSON.stringify(next.constellation)!==JSON.stringify(draft.constellation))
    throw Error('请在星座几何中设置全场景的编队成员');
  const problem=configurationErrors(next,cellSchema)[0];
  if(problem)throw Error(problem.msg);
  if(JSON.stringify(next.fso)!==JSON.stringify(draft.fso) ||
     JSON.stringify(next.topology.inter_satellite)!==JSON.stringify(draft.topology.inter_satellite))
    throw Error('请在连接与链路预算中修改星间链路');
  draft=next;drafts.clear();jsonLock(false);updatePreview();report('');
}catch(e){report(e.message)}};
$('export-config').onclick=()=>{
  const link=document.createElement('a');
  link.href=URL.createObjectURL(new Blob([JSON.stringify(draft,null,2)],{type:'application/json'}));
  link.download=session.id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
const previewCaption=observed?'TLE 成员拓扑示意':'编队构型示意';
const memberPrefix=(observed?formation.name:session.id)+'-';
function showMember(index){
  const id=index==null?null:draft.hardware.satellite_ids[index];
  const caption=$('scene-links'), text=id||previewCaption;
  if(caption.textContent!==text)caption.textContent=text;
}
let scene;
try {
  const noop=()=>{};
  scene=new AsterMindScene($('scene'),{onSelect:noop,onHover:noop,onHoverStation:noop,onFormationNodeSelect:showMember,onHardwareNodeSelect:showMember,onTerminalNodeSelect:showMember,
    memberLabel:(id,i)=>id.startsWith(memberPrefix)?id.slice(memberPrefix.length):String(i+1),
    onMemberHover:showMember,onLinkSelect:noop,onLinkRemove:noop});
  scene.prepareConfigAnchor();
} catch { $('scene-caption').hidden=true; }
function updatePreview() {
  const ids=draft.hardware.satellite_ids;
  const layout=geo.layout??geo.topology;
  $('scene-summary').textContent=formationNames[layout]+' '+ids.length+' 颗卫星';
  showMember(null);
  if(!scene)return;
  const links=draft.topology.inter_satellite.links;
  const previewConfig={
    hardware:draft.hardware,
    constellation:{formation:{layout,spacing_km:geo.spacing_km,altitude_km:geo.altitude_km}},
    topology:{source:'formation',inter_satellite:{links}},
  };
  let positions=null;
  if(fixed){
    const offsets=formation.cluster.member_offsets_m,extent=Math.max(1,...offsets.flat().map(Math.abs));
    positions=offsets.map(row=>row.map(x=>x/extent));
  }
  syncSceneConfiguration(scene,previewConfig,null,positions);
}
function renderGeometry(){
  renderConfigForm(geometryHost,geo,Object.keys(geo),geoSchema,()=>{},{
    labels:{topology:'编队内部连接',satellite_count:'编队卫星数'},
    drafts,onEdit(path,value){
      if(path[0]==='satellite_count' && !observed && !fixed){
        geometry.satellites_per_cluster=value;
        draft.hardware.satellite_count=value;draft.hardware.satellite_ids=members();
        synchronizeResources(draft,capabilities);renderGeometry();renderParameters();
      }
      if (path[0]==='satellite_count' || path[0]==='layout' || path[0]==='topology') {
        explicitPairs=null;
        inheritLinks();
        renderParameters();
      }
      updatePreview();
    },
  });
  if(observed || fixed) geometryHost.querySelectorAll('[data-field-path=\'["satellite_count"]\'] input').forEach(input=>input.disabled=true);
}
function renderParameters(){
  const choices=editorChoices(draft,capabilities);
  let refresh=false;
  Object.assign(choices,{drafts,onDraftChange:schedulePreflight,onEdit:(path,value,previous)=>{
    if(['hardware','model'].includes(path[0]) || path[0]==='deployment' && ['worker_pools','mode'].includes(path[1])) clearDependentDrafts(drafts);
    refresh=editDependencies(draft,path,value,capabilities,previous);
    if(path[0]==='deployment' && path[1]==='placements')refresh=true;
    if(path[0]==='topology')updatePreview();
  }});
  for(const [stage,index,paths] of fields) if(section(stage,index)) renderParameterGroup(section(stage,index),draft,paths,capabilities,path=>{
    if(refresh || path?.[0]==='deployment')renderParameters();
    else renderJson();
    schedulePreflight();
  },{...choices,stage});
  renderJson();
  renderComputeSummary(draft,capabilities,()=>{renderJson();schedulePreflight()},choices);
  schedulePreflight();
}
function renderJson(){
  const {compute_mode,fast_link_contention,...editable}=draft;
  $('advanced-config').value=JSON.stringify(editable,null,2);
  renderCellConfigProfile($('config-profile'),draft,labels);
}
$('generate-mapping').onclick=()=>{
  try {
    draft.deployment.placements=mappingDraft(draft,{reset:true});
    renderParameters();
    report('');
  } catch(e){report(e.message)}
};
function stage(name){
  document.body.dataset.stage=name;
  document.querySelectorAll('.config-panel').forEach(panel=>{panel.hidden=panel.dataset.panel!==name;panel.classList.toggle('is-active',!panel.hidden)});
  document.querySelectorAll('#chapter-rail button').forEach(button=>{const active=button.dataset.stageTarget===name;button.classList.toggle('is-current',active);button.setAttribute('aria-current',active?'step':'false')});
  if(scene){sceneState.stage=({network:'terminal',deployment:'hardware',runtime:'inference',workload:'inference'})[name]||name;scene.setStage(sceneState.stage)}
  schedulePreflight();
}
document.querySelectorAll('[data-stage-target]').forEach(button=>button.onclick=()=>stage(button.dataset.stageTarget));
$('reset-view').onclick=()=>scene?.resetConfigView();
function currentLayout(){
  if(jsonDirty)throw Error('请先应用或放弃 JSON 编辑。');
  const pending=drafts.values().next().value;
  if(pending)throw Error(pending.error || '请修正输入');
  if(!observed){
    const invalid=geometryHost.querySelector('input:invalid,select:invalid');
    if(invalid){invalid.reportValidity();throw Error(invalid.validationMessage)}
    if(geo.satellite_count>1 && geo.spacing_km*1000<=formation.cluster.d_keepout_m)throw Error('编队距离必须大于安全间距');
    if(formation.egress_policy!=='free' && formation.ingress_members?.some(n=>n>=geo.satellite_count))throw Error('编队卫星数不足以容纳当前对外接入成员');
    Object.assign(geometry,{satellites_per_cluster:geo.satellite_count,formation_pitch_m:geo.spacing_km*1000,altitude_km:geo.altitude_km});
  }
  if(synthetic){
    geometry.motion_model='static_formation';
    delete geometry.grid_shape;
  }
  geometry.topology=geo.layout??geo.topology;
  if(!modelOptions(draft,capabilities).includes(draft.model))throw Error('请选择当前计算模式与卡型支持的模型');
  const problem=configurationErrors(draft,cellSchema)[0];
  if(problem)throw Error(problem.msg);
  const layout=sceneCellLayout(draft,capabilities,members());
  layout.custom_links=explicitPairs!==null;
  return layout;
}
async function validateLayout(layout) {
  const response=await fetch('/cell/api/scene/validate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:session.id,compute_mode:mode,layout})});
  const result=await response.json();if(!response.ok)throw Error(result.detail||'编队配置校验失败');
  return result;
}
let preflightTimer,preflightVersion=0;
function schedulePreflight() {
  const version=++preflightVersion;
  clearTimeout(preflightTimer);
  if(document.body.dataset.stage!=='runtime')return;
  const output=$('configuration-check');
  renderSchedulingReadout(document,null);
  output.textContent='正在进行后端预检…';
  preflightTimer=setTimeout(async()=>{
    try {
      const result=await validateLayout(currentLayout());
      if(version!==preflightVersion)return;
      renderSchedulingReadout(document,result.cell);
      output.textContent=result.launch_blocker||'编队配置校验通过';
    } catch(error) {if(version===preflightVersion)output.textContent=error.message}
  },250);
}
renderGeometry();renderParameters();updatePreview();stage('formation');
session.ready(async all=>{
  const layout=currentLayout();
  await validateLayout(layout);
  session.apply(geometry,layout,all);
});
