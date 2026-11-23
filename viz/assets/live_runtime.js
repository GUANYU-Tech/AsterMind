import {get,post} from '/assets/studio_common.js';
import {acceptsLiveState,requestLiveState,livePlaybackState,shouldAdvanceLive,liveWaitingFeedback,liveAdvancePlan} from '/assets/live_playback_state.js';
import {eventGroups,eventDescription,renderEventRows} from '/assets/event_view.js';
import {updateLiveScene,inspectLiveEvent,viewingLive,hasPendingLiveEvents,nextLiveEvent,showCurrentLiveEvent,setLivePlayback,setLivePhysicalSource,advanceLiveWatch,streamingLive,bufferedWatchSeconds,watchedThrough,executedThroughTime,latestCommunicationTime,watchDidMove,watchingSelectedPath,selectedPathComplete,liveWatchError,resumeLiveWatch} from '/assets/live_scene.js';
import '/assets/fault_state.js';
import {createCellTopology} from '/assets/cell_topology.js';
import {formatTime} from '/assets/communication_clock.js';
const $=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
const observeOnly=new URLSearchParams(location.search).get('observe')==='1';
const recordedScene=new URLSearchParams(location.search).get('recorded')==='1';
if(recordedScene){
 document.documentElement.classList.add('recorded-scene');
 document.querySelector('h1').textContent='场景回放';
 document.title='场景回放，已保存记录';
}
let identity=location.pathname.split('/')[2]||null,state=null,version=null,playing=new URLSearchParams(location.search).get('autostart')==='1',busy=false,selectedNode='',selectedLink='',pendingConnections=new Set();
const pageOpenedAt=Date.now()/1000;
let commandPending=false,advanceInFlight=Promise.resolve();
let pollError=null;
let nextDisplayAt=0,singleStep=false;
let nativeConfig=null;
const topologyHost=document.createElement('div');$('selection').after(topologyHost);
const editTopology=document.createElement('a');editTopology.className='button';editTopology.textContent='修改拓扑并另存场景 →';editTopology.hidden=true;topologyHost.after(editTopology);
const cellTopology=createCellTopology(topologyHost,{command:async value=>{const result=await command(value);if(result?.error)throw Error(result.error);return null},onSatellite:id=>selectNode(id)});
if(identity)get('/runs/'+encodeURIComponent(identity)+'/input').then(value=>{if([1,'orbitserve.native_cells.v1'].includes(value.schema_version)){nativeConfig=value;setLivePhysicalSource(value.physical_dataset);topology()}}).catch(()=>{});
const sourceRun=new URLSearchParams(location.search).get('from_run');
if(identity&&new URLSearchParams(location.search).has('autostart')){const params=new URLSearchParams(location.search);params.delete('autostart');history.replaceState(null,'',location.pathname+(params.size?'?'+params:''))}
document.querySelector('main').classList.toggle('has-session',!!identity);
document.querySelector('main').classList.toggle('awaiting-live',!identity);
const labels={request_arrival:'请求到达',route_plan:'路线规划',route_recovery:'恢复选路',route_wait:'等待可行路线',fault_injected:'故障生效',fault_restored:'故障恢复',topology_command:'拓扑操作',transfer_queued:'本跳入队',transmit_start:'开始发送',transmit_complete:'发送完成',hop_arrival:'到达下一跳',transfer_cancelled:'传输中断',bandwidth_update:'带宽分配',link_state:'链路状态',compute_start:'计算开始',prefill_complete:'Prefill 完成',compute_complete:'计算完成',request_complete:'请求完成',request_failed:'请求失败',first_delivered:'首 token 交付',output_delivered:'输出交付',retry_transfer:'执行重试'};
const stateLabels={preparing:'准备场景',paused:'当前时刻已暂停',advancing:'正在执行',completed:'已完成',failed:'执行失败',cancelled:'已结束',finalizing:'保存执行日志'};
function trafficStatus(flows,label){const host=$('trafficState');host.replaceChildren();for(const [status,text]of [['transmitting','正在传输'],['queued','排队'],['propagating','传播']]){const item=document.createElement('span'),name=document.createElement('span'),value=document.createElement('strong');item.className='traffic-stat';name.textContent=text;value.textContent=flows.filter(f=>f.status===status).length;item.append(name,value);host.append(item)}const event=document.createElement('span');event.className='traffic-event';event.textContent=label;host.append(event)}
let noticeMessage=$('notice').textContent;
function notice(message,type=''){
 const host=$('notice'),classes='notice '+type;
 if(noticeMessage!==message){noticeMessage=message;host.textContent=message}
 if(host.className!==classes)host.className=classes;
}
function controlText(id,text){if($(id).textContent!==text)$(id).textContent=text}
function displayClock(){
 const seconds=value=>Number(value).toFixed(6).replace(/\.?0+$/,''),view=window.networkReplayState?.()?.runtime,time=view?.time_s??state?.time_s??0;
 const exact=$('livePlaybackMode')?.value==='inspect'&&view?.event?.time_ticks!=null;
 const clock=(exact?formatTime(view.event):seconds(time))+' s',info=state?.nodes?`后端已执行至 ${seconds(Math.max(state.time_s||0,executedThroughTime()))} s`:'等待场景初始化';
 if($('clock').textContent!==clock)$('clock').textContent=clock;
 if($('stepInfo').textContent!==info)$('stepInfo').textContent=info;
}
function el(tag,attrs={},text){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n}
function option(host,value,text){const n=document.createElement('option');n.value=value;n.textContent=text;host.append(n)}
function color(event){if(event.kind==='fault_restored')return '#2b9b78';if(event.source==='manual')return '#dc4c64';if(event.source==='random'||/fault/.test(event.kind))return '#d89319';if(event.reason==='sun_exclusion')return '#8b63c8';return '#315bea'}
function describe(event){if(/fault_(injected|restored)/.test(event.kind)&&event.start_step!=null)return `第 ${event.start_step} → ${event.end_step} 片，持续 ${event.duration_steps} 片 / ${event.duration_s} s，${event.source||'scheduled'}${event.right_censored?'，计划恢复超出时域，结束时仍故障':''}`;return event.message||event.reason||event.link_id||(event.source_node_id?event.source_node_id+' → '+event.target_node_id:'')||event.stage||''}
function inspectEvent(event,sync=true){$('eventDetail').textContent=`${event.time_s.toFixed(6)} s，${event.event_label||labels[event.kind]||event.kind}，${event.request_id||event.target_id||''}\n${eventDescription(event)}`;if(sync)inspectLiveEvent(event)}
function visibleEvents(){return [...(state?.event_groups?.recent.business||[]),...(state?.event_groups?.recent.network||[]).filter(e=>/fault|topology_command/.test(e.kind)||e.reason==='sun_exclusion')].sort((a,b)=>a.event_order-b.event_order)}
function timeline(){
 const events=visibleEvents(),width=Math.max(600,$('timeline').clientWidth),left=82,right=width-18,now=state?.time_s||0,span=Math.max(20,(state?.step_s||5)*4),from=Math.max(0,now-span),to=Math.max(from+span,now+1),scale=t=>left+(t-from)/(to-from)*(right-left);
 const svg=el('svg',{viewBox:`0 0 ${width} 230`,role:'group','aria-label':'实际执行事件时间线'}),lanes=['故障 / 拓扑','请求 / 路由','排队 / 传输','计算 / 交付'];
 lanes.forEach((label,i)=>{const y=47+i*42;svg.append(el('text',{x:0,y:y+4,fill:'#91a0b8','font-size':10},label),el('line',{x1:left,y1:y,x2:right,y2:y,stroke:'#e8edf5'}))});
 for(let i=0;i<=6;i++){const t=from+(to-from)*i/6,x=scale(t);svg.append(el('line',{x1:x,y1:22,x2:x,y2:194,stroke:'#edf0f7','stroke-dasharray':'3 5'}),el('text',{x,y:216,fill:'#9ca8ba','font-size':10,'text-anchor':'middle'},t.toFixed(1)+' s'))}
 const open=new Map(),intervals=[];
 for(const e of events){const transfer=/^transmit_/.test(e.kind)||e.kind==='transfer_cancelled',key=transfer?e.transfer_id:'compute:'+e.request_id;if(e.kind==='transmit_start'||e.kind==='compute_start')open.set(key,e);if(['transmit_complete','transfer_cancelled','compute_complete'].includes(e.kind)&&open.has(key)){intervals.push([open.get(key),e.time_s]);open.delete(key)}if(e.kind==='request_failed')for(const[k,start]of open)if(start.request_id===e.request_id){intervals.push([start,e.time_s]);open.delete(k)}}
 for(const start of open.values())intervals.push([start,now]);
 for(const[start,end]of intervals){const low=Math.max(from,start.time_s),high=Math.min(now,end);if(high>low){const row=start.kind==='compute_start'?3:2;svg.append(el('rect',{x:scale(low),y:42+row*42,width:Math.max(1,scale(high)-scale(low)),height:10,rx:3,fill:color(start),opacity:.15}))}}
 const stack=new Map();for(const event of events.filter(e=>e.time_s>=from&&e.time_s<=now)){
  const row=/fault|topology_command|cancelled|link_state/.test(event.kind)?0:/route|request_arrival/.test(event.kind)?1:/transfer|transmit|hop_arrival|bandwidth/.test(event.kind)?2:3,x=scale(event.time_s),key=row+'|'+Math.round(x/5),n=stack.get(key)||0;stack.set(key,n+1);
  const point=el('circle',{cx:x,cy:47+row*42+(n%5-2)*4,r:4,fill:color(event),opacity:.85,'data-event':event.event_order,tabindex:0,role:'button','aria-label':(event.event_label||labels[event.kind]||event.kind)+' '+event.time_s+'秒'});point.append(el('title',{},event.time_s.toFixed(6)+' s，'+(event.event_label||labels[event.kind]||event.kind)+'，'+describe(event)));point.onclick=()=>inspectEvent(event);point.onkeydown=e=>{if(e.key==='Enter')inspectEvent(event)};svg.append(point)
 }
 if(state){svg.append(el('line',{x1:scale(now),y1:17,x2:scale(now),y2:195,stroke:'#315bea','stroke-width':1.5}),el('text',{x:Math.max(left,scale(now)-5),y:10,fill:'#315bea','font-size':9,'text-anchor':'end'},'当前 '+now.toFixed(3)+' s'))}
 $('timeline').replaceChildren(svg);$('timelineInfo').textContent=`已执行至 ${now.toFixed(3)} s，当前显示 ${from.toFixed(1)}–${to.toFixed(1)} s，无预计算业务结果`;
}
function tables(){
 const filter=$('eventFilter').value,group=eventGroups[filter],events=state?.event_groups?.recent[filter]||[];
 $('eventGroupNote').textContent=`${group.note} 累计 ${(state?.event_groups?.counts[filter]||0).toLocaleString()} 条；此处保留最近 300 条。`;
 renderEventRows($('eventRows'),events,inspectEvent);
}
function openInspector(){ $('liveInspector').classList.add('is-open');$('liveInspector').scrollTop=0; }
$('openInspector').onclick=openInspector;
$('closeInspector').onclick=()=>$('liveInspector').classList.remove('is-open');
function selectNode(id){selectedNode=id;selectedLink='';$('linkSelect').value='';$('nodeSelect').value=id;openInspector();window.selectGlobeObject?.({nodeId:id});inspectSelection();topology()}
window.openComputeTopology=id=>{const node=nativeConfig&&state?.nodes?.find(n=>n.cell_id===id&&n.role==='compute');if(!node)return false;selectNode(node.node_id);return true};
function selectLink(id){selectedLink=id;selectedNode='';$('nodeSelect').value='';$('linkSelect').value=id;const link=state?.links.find(e=>e.link_id===id);if(link){$('endpointA').value=link.endpoint_a;$('endpointB').value=link.endpoint_b}openInspector();window.selectGlobeObject?.({linkId:id});inspectSelection();topology()}
function connectionError(a=$('endpointA').value,b=$('endpointB').value){
 if(!a||!b)return '选择两个不同的卫星节点。';
 if(a===b)return '两个端点不能是同一节点。';
 if([a,b].some(id=>!state?.nodes.some(n=>n.node_id===id)))return '请选择有效的节点。';
 const existing=state.links.some(e=>e.endpoint_a===a&&e.endpoint_b===b||e.endpoint_a===b&&e.endpoint_b===a);
 if(!existing&&[a,b].some(id=>state.nodes.find(n=>n.node_id===id).role==='ground'))return '新建链路请选择两个卫星节点。';
 return '';
}
function inspectSelection(){
 const active=['paused','advancing'].includes(state?.state),inspection=active||viewingLive()?state:window.networkReplayState?.()?.snapshot||state;
 const node=inspection?.nodes?.find(n=>n.node_id===selectedNode),link=inspection?.links?.find(e=>e.link_id===selectedLink),editable=active&&!commandPending;
 const hint=!active?'实验已结束，仅可查看。':commandPending?'正在执行操作…':!viewingLive()?'提交时返回后端当前时刻并暂停。':playing||state.state==='advancing'?'提交操作时自动暂停。':'选择对象后设置故障或调整连接。';
 const roles={communication:'通信卫星',compute:'计算卫星',ground:'地面站'};
 $('selection').textContent=(node?`${node.node_id}\n${roles[node.role]||node.role}，${node.state}`:link?`${link.endpoint_a} ↔ ${link.endpoint_b}\n${link.state}${link.reason?'，'+link.reason:''}`:'点击节点或链路')+'\n'+hint;
 if(node?.cell_id){const members=new Set(inspection.nodes.filter(n=>n.cell_id===node.cell_id).map(n=>n.node_id)),external=inspection.links.filter(e=>members.has(e.endpoint_a)!==members.has(e.endpoint_b));$('selection').textContent+=`\nCell 对外已连 ${external.filter(e=>e.state==='UP').length}，建链 ${external.filter(e=>e.state==='SETTING_UP').length}`}
 if(link?.state==='DOWN')$('selection').textContent+='\n未连接，仅显示所选链路位置';
 for(const id of ['nodeFail','nodeRestore'])$(id).disabled=!editable||!node;
 for(const id of ['linkFail','linkRestore','disconnect'])$(id).disabled=!editable||!link;
 $('reconnect').disabled=!editable||!link||['UP','SETTING_UP'].includes(link.state);
 for(const id of ['useEndpointA','useEndpointB'])$(id).disabled=!node||node.role==='ground';
 const error=connectionError();$('connect').disabled=!editable||!!error;
 $('connectionHint').textContent=error||'由后端校验距离、终端与链路状态。';
 $('restartInteractive').hidden=!['completed','failed','cancelled'].includes(state?.state);
 $('restartInteractive').href='/live?from_run='+encodeURIComponent(identity);
}
function topology(){const node=state?.nodes?.find(n=>n.node_id===selectedNode);if(nativeConfig&&node?.role==='compute'){cellTopology.update(nativeConfig,state,selectedNode);editTopology.hidden=false;editTopology.href='/scene?run='+encodeURIComponent(identity)+'&satellite='+encodeURIComponent(selectedNode)}else{cellTopology.clear();editTopology.hidden=true}}
function menus(){
 for(const [id,rows,label,key]of [['nodeSelect',state.nodes,'选择节点','node_id'],['endpointA',state.nodes.filter(n=>n.role!=='ground'),'选择卫星 A','node_id'],['endpointB',state.nodes.filter(n=>n.role!=='ground'),'选择卫星 B','node_id'],['linkSelect',state.links,'选择链路','link_id']]){const host=$(id),signature=rows.map(row=>row[key]+'|'+(row.state||'')).join('\n');if(host.dataset.menuSignature===signature)continue;const prior=host.value,fragment=document.createDocumentFragment();option(fragment,'',label);for(const row of rows)option(fragment,row[key],row[key]+(row.state?'，'+row.state:''));host.replaceChildren(fragment);host.dataset.menuSignature=signature;host.value=prior}
}
function render(){
 document.querySelector('h1').textContent=recordedScene?'场景回放':'在线仿真';
 $('launcher').hidden=!!identity;
 $('viewResults').hidden=state.state!=='completed';$('viewResults').href=`/results?run=${encodeURIComponent(identity)}`;
 $('liveBackend').textContent=state.backend?state.backend.toUpperCase():'';
 const replay=state.replay,flows=replay?RuntimeVisual.stateAt(replay.initial,replay.events,replay.events.length-1).activeTransfers:[];
 const counts=status=>flows.filter(flow=>flow.status===status).length;
 $('runningTraffic').hidden=!state.nodes;
 if(state.nodes)trafficStatus(flows,'已完成请求 '+(state.completed??0));
 $('fullReplay').hidden=state.state!=='completed';$('fullReplay').href=state.execution_model==='native_gpu_operator_graph'?`/runs/${encodeURIComponent(identity)}/native-replay/index.html`:`/runs/${encodeURIComponent(identity)}/runtime?view=playback`;
 $('fullReplay').textContent=state.execution_model==='native_gpu_operator_graph'?'GPU 算子与通信回放 →':'完整 3D 回放 →';
 displayClock();
 $('stop').disabled=!['preparing','paused','advancing'].includes(state.state)||commandPending;
 if(!state.nodes){playbackControls();if(['failed','cancelled'].includes(state.state))notice(state.message,'error');return}
 $('eventCount').textContent=(state.event_groups?.counts.business||0).toLocaleString();$('upCount').textContent=state.links.filter(e=>e.state==='UP').length.toLocaleString();$('linkCount').textContent=`${state.links.length.toLocaleString()} 条候选链路，${state.nodes.length} 个节点`;
 const faults=FaultVisual.collect(state);$('faultCount').textContent=faults.filter(f=>f.type==='node').length+' / '+faults.filter(f=>f.type==='link').length;$('requests').textContent=state.completed+' / '+state.failed;$('requestCount').textContent=state.request_count+' 个实际业务请求'+(state.communication_label?'，'+state.communication_label:'');$('revision').textContent='网络状态修订号 '+state.topology_revision;
 menus();updateLiveScene(state,{onNode:selectNode,onLink:selectLink,onEvent:inspectEvent,onMode:inspectSelection,onPresent:presentEvent,onHistory:pauseForHistory,onWatchPlay:()=>{playing=true;playbackControls()},onWatchEnd:terminal=>{playing=false;playbackControls();notice(terminal==='completed'?'所选传输已到达终点':terminal==='failed'?'所选传输已失败':'已到达当前已执行的时刻')},label:labels});inspectSelection();topology();timeline();tables();playbackControls();
 for(const id of pendingConnections){const edge=state.links.find(e=>e.link_id===id);if(edge?.state==='UP'){notice('连接成功：'+id,'success');pendingConnections.delete(id)}}
 if(['failed','cancelled'].includes(state.state))notice(state.message,'error');
}
function presentEvent(runtime){
 displayClock();
 const flows=runtime.activeTransfers||[],count=status=>flows.filter(f=>f.status===status).length;
 trafficStatus(flows,runtime.event.event_label||labels[runtime.event.kind]||'等待通信');
 playbackControls();
}
function pauseForHistory(){playing=false;singleStep=false;playbackControls();if(!observeOnly&&state?.state==='advancing')post(`/live/${identity}/pause`,{}).catch(error=>notice(error.message,'error'))}
function playbackControls(){
 displayClock();
 const pending=hasPendingLiveEvents(),playback=livePlaybackState(state,{playing,pending,stream:streamingLive(),through:watchedThrough(),observeOnly}),disabled=commandPending||singleStep;
 const feedback=liveWaitingFeedback(state,{waiting:playback.waiting,pending,error:liveWatchError(),observeOnly,playing});
 setLivePlayback(playing,Number($('eventRate').value),feedback);
 controlText('sessionState',pollError?'连接恢复中':state?.state==='preparing'||!state?'正在准备场景':playing?'在线运行中':stateLabels[state.state]||state.state);
 $('advance').disabled=!playback.ready||disabled;$('play').disabled=!playback.canToggle||commandPending;
 if($('returnLive'))$('returnLive').hidden=viewingLive();
 if($('previousTime'))$('previousTime').disabled=!streamingLive()||commandPending||singleStep;
 controlText('play',playback.label);
 return {...playback,feedback};
}
async function advanceEvent(batch=false){
 if(observeOnly)return;
 busy=true;playbackControls();
 try{const replay=state.replay,flows=replay?RuntimeVisual.stateAt(replay.initial,replay.events,replay.events.length-1).activeTransfers:[];
  const payload=liveAdvancePlan(state,{stream:batch,activeTransfers:flows.some(flow=>['transmitting','queued','propagating'].includes(flow.status)),singleStep}),before=state.version;
  advanceInFlight=post(`/live/${identity}/advance`,payload);const reply=await advanceInFlight;if(state.version===before){state.state=reply.state;state.version=reply.version??before+1}version=null}
 finally{busy=false;playbackControls()}
}
async function refresh(){
 if(!identity)return;
 try{
  const update=['completed','cancelled','failed'].includes(state?.state)?{unchanged:true}:await requestLiveState(`/live/${encodeURIComponent(identity)}/state`+(version==null?'':'?after='+version));
  pollError=null;
  if(acceptsLiveState(state,update)){state=update;version=update.version??null;render()}
 }catch(error){pollError=error.name==='TimeoutError'?'状态读取超时，正在自动重试，无需刷新页面':error.message;notice(pollError,'error')}
 finally{setTimeout(refresh,document.hidden?2000:state?.state==='preparing'?1000:playing?500:1000)}
}
// A slow state snapshot or advance acknowledgement must not block rendering
// communication that has already arrived in the independent page buffer.
function requestAdvance(batch){if(!busy&&!commandPending)advanceEvent(batch).catch(error=>{version=null;notice(error.message,'error')})}
async function playbackTick(){
 try{
  if(singleStep&&!commandPending&&!busy){
   const stream=await advanceLiveWatch(performance.now(),1,true);
   if(stream&&watchDidMove()){singleStep=false;playbackControls()}
   else if(!stream&&hasPendingLiveEvents()){nextLiveEvent();singleStep=false;playbackControls()}
   else if(!observeOnly&&state?.state==='paused'&&state.time_s<state.end_s)requestAdvance(stream);
   else if(['completed','cancelled','failed'].includes(state?.state)){singleStep=false;playbackControls()}
  }
  if(playing&&!commandPending){
   const inspect=$('livePlaybackMode')?.value==='inspect';
   const due=performance.now()>=nextDisplayAt;
   const stream=await advanceLiveWatch(performance.now(),inspect&&!due?0:Number($('eventRate').value),inspect&&due);
   if(inspect&&performance.now()>=nextDisplayAt)nextDisplayAt=performance.now()+600/Number($('eventRate').value);
   if(!stream&&hasPendingLiveEvents()&&performance.now()>=nextDisplayAt){nextLiveEvent();nextDisplayAt=performance.now()+600/Number($('eventRate').value)}
   // The backend executes every causal event in a bounded look-ahead batch.
   // The viewer consumes lossless communication pages on its independent clock.
   if(!observeOnly&&shouldAdvanceLive(state,{stream,pending:hasPendingLiveEvents(),remaining:bufferedWatchSeconds(),selectedComplete:watchingSelectedPath()&&selectedPathComplete()}))requestAdvance(stream);
  }
  const playback=playbackControls();
  if(pollError){
   notice(pollError,'error');
  }else if(playback.drained){
   playing=false;singleStep=false;playbackControls();notice(state.message||'仿真已结束，可查看实验结果',state.state==='failed'?'error':'');
  }else if(playback.feedback?.phase==='preparing'){
   const waited=Math.max(0,Math.floor(Date.now()/1000-(state?.preparation_started_at||pageOpenedAt)));
   notice(`${playback.feedback.title}：${state?.message||'初始化仿真环境'}，已等待 ${waited} 秒。${playback.feedback.detail}`,'working');
  }else if(playing&&!['finalizing','failed'].includes(state?.state)){
   notice(liveWatchError()||'正在播放通信，数据就绪后会自动继续。',liveWatchError()?'error':'');
  }else if(playback.waiting||playback.feedback){
   notice(liveWatchError()||(state?.state==='failed'?state.message:playback.feedback?.title+'，'+playback.feedback?.detail),liveWatchError()||state?.state==='failed'?'error':'working');
  }else if($('notice').className==='notice working'||state?.nodes&&noticeMessage==='等待任务运行。'){
   notice(playing?'正在播放通信传输':'已暂停观看，可检查下一时刻');
  }
 }catch(error){notice(error.message,'error')}
 finally{setTimeout(playbackTick,playing||singleStep?40:250)}
}
async function command(value){
 if(!state||commandPending)return;
 let applied=false;
 const error=!['paused','advancing'].includes(state.state)?'实验已结束，仅可查看。':value.action==='connect'?connectionError(value.endpoint_a,value.endpoint_b):!value.target_id?'请先选择对象。':'';
 if(error){$('operationStatus').textContent=error;$('operationStatus').className='error';return{error}}
 playing=false;showCurrentLiveEvent();commandPending=true;busy=true;playbackControls();inspectSelection();$('play').disabled=true;
 $('operationStatus').textContent='正在暂停并执行操作…';$('operationStatus').className='';
 try{
  await advanceInFlight.catch(()=>{});
  const current=await get(`/live/${identity}/state`);
  if(!['paused','advancing'].includes(current.state))throw Error('实验已结束，仅可查看。');
  const reply=await post(`/live/${identity}/command`,{...value,topology_revision:current.topology_revision});
  if(value.action==='connect'&&reply.message!=='连接成功')pendingConnections.add(reply.target_id);
  state=await get(`/live/${identity}/state`);version=state.version;
  applied=true;
  const message=reply.message+'，已暂停，可继续运行';notice(message,'success');$('operationStatus').textContent=message;$('operationStatus').className='success';
  return reply;
 }catch(error){notice(error.message,'error');$('operationStatus').textContent=error.message;$('operationStatus').className='error';version=null;return{error:error.message}}
 finally{commandPending=false;busy=false;render();if(applied){showCurrentLiveEvent();playbackControls()}}
}
$('nodeSelect').onchange=()=>selectNode($('nodeSelect').value);$('linkSelect').onchange=()=>selectLink($('linkSelect').value);$('eventFilter').onchange=tables;
for(const [id,action]of [['nodeFail','node_fail'],['nodeRestore','node_restore'],['linkFail','link_fail'],['linkRestore','link_restore'],['disconnect','disconnect']])$(id).onclick=()=>command({action,target_id:action.startsWith('node')?selectedNode:selectedLink});
$('connect').onclick=()=>command({action:'connect',endpoint_a:$('endpointA').value,endpoint_b:$('endpointB').value});
$('reconnect').onclick=()=>{const link=state?.links.find(e=>e.link_id===selectedLink);if(link)command({action:'connect',endpoint_a:link.endpoint_a,endpoint_b:link.endpoint_b})};
$('endpointA').onchange=$('endpointB').onchange=inspectSelection;
for(const endpoint of ['A','B'])$('useEndpoint'+endpoint).onclick=()=>{$('endpoint'+endpoint).value=selectedNode;inspectSelection()};
$('advance').onclick=()=>{if(commandPending||singleStep)return;playing=false;singleStep=true;playbackControls()};
if($('previousTime'))$('previousTime').onclick=async()=>{playing=false;try{await advanceLiveWatch(performance.now(),1,-1);playbackControls()}catch(error){notice(error.message,'error')}};
if($('returnLive'))$('returnLive').onclick=()=>{playing=false;showCurrentLiveEvent();playbackControls()};
$('play').onclick=()=>{playing=!playing;singleStep=false;nextDisplayAt=0;if(playing)resumeLiveWatch();playbackControls();if(!observeOnly&&!playing&&state.state==='advancing')post('/live/'+identity+'/pause',{}).catch(error=>notice(error.message,'error'));notice(playing?'正在播放通信传输':'已暂停观看，可检查下一时刻')};
$('eventRate').onchange=()=>{nextDisplayAt=performance.now()+1000/Number($('eventRate').value);playbackControls()};
$('stop').onclick=async()=>{playing=false;playbackControls();try{await post(`/live/${identity}/stop`,{});version=null}catch(error){notice(error.message,'error')}};
let selectedSceneReady=false;
async function start(){
 if(!selectedSceneReady){$('launchStatus').textContent='请选择场景';return}
 $('startScene').disabled=true;$('launchStatus').textContent='正在启动仿真…';
 try{const reply=await post('/live/start',{scene_id:$('sceneSelect').value});location.href=reply.url+'?autostart=1'}
 catch(error){$('launchStatus').textContent=error.message;$('startScene').disabled=!selectedSceneReady}
}
$('startScene').onclick=start;
$('startFromRun').onclick=async()=>{const button=$('startFromRun');button.disabled=true;$('launchStatus').textContent='正在启动仿真…';try{const reply=await post('/live/start',{run_id:sourceRun});location.href=reply.url+'?autostart=1'}catch(error){$('launchStatus').textContent=error.message;button.disabled=false}};
window.addEventListener('resize',()=>{timeline();topology()});
if(identity){refresh();playbackTick()}else if(sourceRun){
 $('startScene').hidden=$('launchChoices').hidden=true;
 $('startFromRun').hidden=false;$('startFromRun').disabled=true;
 $('launchDescription').textContent='使用原配置重新运行。';
 get('/runs/'+encodeURIComponent(sourceRun)+'/input').then(value=>{const ready=[1,'orbitserve.native_cells.v1'].includes(value.schema_version)?value.physical_dataset&&value.physical&&value.cells?.length:value.physical_network;if(!ready)throw Error('原运行缺少完整实验配置');$('launchDescription').textContent='原配置已就绪';$('startFromRun').disabled=false;timeline()}).catch(error=>$('launchStatus').textContent=error.message);
}else get('/scene-catalog').then(rows=>{for(const row of rows)option($('sceneSelect'),row.id,row.name);if(!rows.length)$('launchStatus').textContent='暂无场景，先新建任务。';timeline()}).catch(error=>$('launchStatus').textContent=error.message);
if(!identity&&!sourceRun){
 document.querySelector('main').classList.add('awaiting-live');
 $('launchDescription').textContent='启动后推进真实仿真，并可干预节点与链路。';
 $('launchStatus').textContent='历史回放和分析位于实验结果。';
 $('browseResults').hidden=false;
 get('/current-experiment').then(async current=>{if(!current.id)return;const job=await get('/runs/'+encodeURIComponent(current.id)+'/status');if(job.interactive&&job.state==='running'&&job.runtime_url)location.replace(job.runtime_url)}).catch(()=>{});
}

$('sceneSelect').onchange=async()=>{
 const id=$('sceneSelect').value;selectedSceneReady=false;$('startScene').disabled=true;
 $('sceneParameters').hidden=true;$('launchStatus').textContent='';
 $('launchDescription').textContent='选择场景，开始仿真。';
 if(!id)return;
 try{const doc=await get('/documents/scenes/'+encodeURIComponent(id));if(id!==$('sceneSelect').value)return;
  $('launchDescription').textContent=doc.name;$('editSceneFaults').href='/scene?id='+encodeURIComponent(id);
  $('sceneParameters').hidden=false;selectedSceneReady=true;$('startScene').disabled=false;
 }catch(error){if(id===$('sceneSelect').value)$('launchStatus').textContent=error.message}
};
$('startScene').disabled=true;
