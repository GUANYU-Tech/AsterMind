import '/assets/visual_state.js';
import '/assets/runtime_state.js';
import '/assets/constellation_layout.js';
import {createLiveEventCursor,isPlaybackEvent} from '/assets/live_event_cursor.js';
import {eventsForPath} from '/assets/live_traffic_model.js';
import {LiveCommunicationWatch} from '/assets/live_watch.js';
import {loadReplayJson} from '/assets/replay_page_loader.js';
import {formatTime} from '/assets/communication_clock.js';

const $=id=>document.getElementById(id),liveCursor=createLiveEventCursor();
let latest=null,presented=null,cursor=-1,history=false,timer=null,loading=null,callbacks={};
let livePlaying=false,liveRate=1,historyPlaying=false,liveWaiting=null;
let pathReplay=null;
let watchPlayer=null,watchView=null,watchPollAt=0,watchRevision=0,watchLast=null,watchBusy=false,watchIndexRequest=null;
let watchMoved=false;
let detachedWatch=false;
const watchFrames=new Map(),recentTransfers=new Map();
export function streamingLive(){return !!watchPlayer?.index}
export function bufferedWatchSeconds(){return watchPlayer?.remaining||0}
export function watchedThrough(){return watchPlayer?.index?.through_order||0}
export function executedThroughTime(){return Math.max(latest?.time_s||0,watchPlayer?.index?.end_s||0)}
export function latestCommunicationTime(){
 const end=watchPlayer?.index?.busy_intervals?.at(-1)?.[1];
 const kinds=new Set(['transfer_queued','transmit_start','transmit_complete','hop_arrival','transfer_cancelled','transfer_dropped','transfer_retried']);
 const recent=(latest?.events||latest?.replay?.events||[]).findLast(event=>kinds.has(event.kind)&&event.source_node_id&&event.target_node_id&&event.source_node_id!==event.target_node_id)?.time_s;
 const times=[end,recent].filter(Number.isFinite);return times.length?Math.max(...times):null;
}
export function liveWatchError(){return watchPlayer?.streamError||null}
export function watchDidMove(){return watchMoved}
export function watchingSelectedPath(){return !!watchPlayer?.selected}
export function selectedPathComplete(){return !!pathReplay?.path?.terminal}
export function resumeLiveWatch(){watchLast=null;if(watchPlayer?.selected&&selectedPathComplete()&&!watchPlayer.pending){watchPlayer.time=watchPlayer.selected.toSimulation(0);watchPlayer.exact=null}}
async function streamIndex(){
 if(watchIndexRequest)return watchIndexRequest;
 if(!latest?.id||performance.now()<watchPollAt)return;
 watchPollAt=performance.now()+200;
 watchIndexRequest=(async()=>{
 const response=await fetch(`/runs/${encodeURIComponent(latest.id)}/native-replay/live-communication.json`,{cache:'no-store'});
 if(!response.ok)return;
 const index=await response.json();
 if(!watchPlayer)watchPlayer=new LiveCommunicationWatch(name=>loadReplayJson(`/runs/${encodeURIComponent(latest.id)}/native-replay/${name}`,{events:true}));
 watchPlayer.update(index);
 })();
 try{await watchIndexRequest}finally{watchIndexRequest=null}
}
function renderWatch(view){
 if(!latest?.nodes||!view)return;
 const runtime=RuntimeVisual.stateAt(view.initial,view.events,view.cut);
 runtime.event=view.event;runtime.time_s=view.time;runtime.observed_order=view.events[view.cut]?.event_order??view.event.event_order;runtime.time_group=true;
 runtime.activeTransfers=runtime.activeTransfers.filter(e=>e.source_node_id!==e.target_node_id);
 const now=performance.now();for(const [id,row]of recentTransfers)if(now-row.shownAt>500)recentTransfers.delete(id);
 for(const event of view.arriving||[])recentTransfers.set(event.transfer_id,{...event,shownAt:now});
 const frame=[...watchFrames.values()].filter(f=>f.time_s<=view.time).at(-1)||view.geometry||[...watchFrames.values()][0]||latest;
 const links=new Map(frame.links.map(link=>[link.link_id,{...link}]));
 for(const row of [...(view.initial.link_states||[]),...view.events.slice(0,view.cut+1)]){
  if(row.kind==='link_state'&&links.has(row.link_id))Object.assign(links.get(row.link_id),{state:row.state,reason:row.reason});
  if(row.kind==='topology_command'&&links.has(row.target_id)){if(row.action==='disconnect')links.get(row.target_id).state='DOWN';if(row.action==='connect'&&row.link_state)Object.assign(links.get(row.target_id),row.link_state)}
 }
 const paths={...pathContext,paths:[...new Map([...(pathContext?.paths||[]),...searchedPaths,...(pathReplay?.path?[pathReplay.path]:[])].map(p=>[p.id,p])).values()]};
 presented={snapshot:RuntimeVisual.applyFaults({...frame,links:[...links.values()],time_s:view.time},runtime.faults),data:{sun_positions:latest.sun_positions||[],network:{nodes:latest.nodes,faults:runtime.faults,inter_cell_links:latest.inter_cell_links},time:{step_s:latest.step_s,duration_s:latest.end_s}},runtime,pathContext:paths,record:{request:{request_id:runtime.request_id}},route:{path:runtime.activeTransfers},recentTransfers:[...recentTransfers.values()],kind:'all',revision:'watch-'+(++watchRevision)};
 syncPlayback();callbacks.onPresent?.(runtime);callbacks.onMode?.();
 $('visualTime').textContent=formatTime(watchPlayer.exact??{time_s:view.time})+' s';
 $('geometryStatus').textContent=runtime.event.event_label||callbacks.label?.[runtime.event.kind]||'在线通信';
 $('visualMode').textContent='';
 if(!loading)loading=import('/assets/globe.js').catch(error=>{$('geometryStatus').textContent=error.message;loading=null});
}
export async function advanceLiveWatch(now,rate=1,inspect=false){
 if(watchBusy)return false;watchBusy=true;
 watchMoved=false;
 try{const update=streamIndex();if(!streamingLive())await update;else update.catch(()=>{});if(!streamingLive())return false;
  const elapsed=watchLast===null ? .08 : Math.max(0,(now-watchLast)/1000);watchLast=now;
  const view=inspect?await watchPlayer.next(typeof inspect==='number'?inspect:1):await watchPlayer.advance(elapsed,rate);
  if(view){
   if(typeof inspect==='number'&&inspect<0)detachedWatch=true;
   if(physicalSource&&![...watchFrames.values()].some(frame=>frame.time_s<=view.time)){
    const base='/runs/'+encodeURIComponent(physicalSource)+'/';
    if(!physicalIndex){const response=await fetch(base+'environment-index');if(response.ok)physicalIndex=await response.json()}
    const descriptor=physicalIndex?.artifacts.frames.filter(frame=>frame.time_s<=view.time).at(-1);
    if(descriptor){if(!physicalFrames.has(descriptor.artifact)){const response=await fetch(base+descriptor.artifact);if(response.ok)physicalFrames.set(descriptor.artifact,await response.json());while(physicalFrames.size>8)physicalFrames.delete(physicalFrames.keys().next().value)}view.geometry=physicalFrames.get(descriptor.artifact)}
   }
   watchMoved=true;watchView=view;renderWatch(view)
  }
  else if(watchPlayer.selected&&pathReplay?.path?.terminal&&!inspect)callbacks.onWatchEnd?.(pathReplay.path.terminal);
  return true;
 }finally{watchBusy=false}
}
let pathRequest=0;
let physicalSource=null,physicalIndex=null;
const physicalFrames=new Map();
export function setLivePhysicalSource(path){const id=String(path||'').replaceAll('\\','/').split('/').filter(Boolean).at(-1);if(id&&/^[a-zA-Z0-9_-]+$/.test(id))physicalSource=id}
const replayData=()=>pathReplay||latest?.replay||{events:latest?.events||[],initial:{},frames:latest?[latest]:[]};
const displayEvent=e=>isPlaybackEvent(e)||!!pathReplay&&e.kind==='cell_transfer_terminal';
let pathContext=null,pathPollAt=0,pathLoading=false;
let searchedPaths=[],pathSearchSerial=0,pathSearchRevision=0,pathSearchTimer=null;
window.queryRuntimePaths=query=>{
 const serial=++pathSearchSerial;clearTimeout(pathSearchTimer);searchedPaths=[];pathSearchRevision++;render();
 if(!query.trim()||!latest?.id||latest.execution_model!=='native_gpu_operator_graph')return;
 pathSearchTimer=setTimeout(async()=>{
  try{const response=await fetch(`/runs/${encodeURIComponent(latest.id)}/flow-paths?query=${encodeURIComponent(query.trim())}&through=${latest.event_count||0}`);
   if(response.ok){const data=await response.json();if(serial===pathSearchSerial){searchedPaths=data.paths||[];pathSearchRevision++;render()}}
  }catch{}
 },250);
};
async function loadPaths(){
 if(!latest?.id||latest.execution_model!=='native_gpu_operator_graph'||pathLoading||performance.now()<pathPollAt||pathContext?.through_event_count>=latest.event_count)return;
 pathPollAt=performance.now()+3000;pathLoading=true;
 try{const response=await fetch(`/runs/${encodeURIComponent(latest.id)}/native-replay/live-paths.json`,{cache:'no-store'});
  if(response.ok){const data=await response.json();if(data.schema==='orbitserve.live_paths.v1'&&data.through_event_count!==pathContext?.through_event_count){pathContext=data;if(pathReplay?.stream){const path=data.paths.find(p=>p.id===pathReplay.path.id);if(path){pathReplay.path=path;watchPlayer.select(path,{restart:false})}}render()}}
 }catch{}finally{pathLoading=false}
}
setInterval(loadPaths,3000);
window.runtimeGlobeMode=true;
window.runtimeLiveMode=true;
window.networkReplayState=()=>presented;
window.inspectNetworkNode=id=>callbacks.onNode?.(id);
window.inspectNetworkLink=id=>callbacks.onLink?.(id);
window.inspectRuntimePath=async path=>{
 if(streamingLive()){
  callbacks.onHistory?.();history=false;pathReplay={path,stream:true};
  watchPlayer.select(path);watchView=await watchPlayer.at(watchPlayer.time);renderWatch(watchView);
  callbacks.onWatchPlay?.();return;
 }
 const request=++pathRequest;
 const events=eventsForPath(path,Math.min(latest.event_count||Infinity,pathContext?.through_event_count||Infinity)-1);
 if(!events.length)return;
 stop();callbacks.onHistory?.();history=true;cursor=-1;
 let frames=latest.replay.frames;
 if(events[0].time_s<frames[0].time_s){
  if(!physicalSource){$('geometryStatus').textContent='这条传输早于当前回看窗口，缺少对应几何快照';return}
  try{
   const base='/runs/'+encodeURIComponent(physicalSource)+'/';
   if(!physicalIndex){const response=await fetch(base+'environment-index');if(!response.ok)throw Error('读取物理场景失败');physicalIndex=await response.json()}
   const descriptors=physicalIndex.artifacts.frames,first=Math.max(0,descriptors.findLastIndex(f=>f.time_s<=events[0].time_s));
   const needed=descriptors.slice(first).filter(f=>f.time_s<=events.at(-1).time_s);frames=[];
   for(const descriptor of needed){if(!physicalFrames.has(descriptor.artifact)){const response=await fetch(base+descriptor.artifact);if(!response.ok)throw Error('读取路径对应时刻失败');physicalFrames.set(descriptor.artifact,await response.json())}frames.push(physicalFrames.get(descriptor.artifact))}
  }catch(error){$('geometryStatus').textContent=error.message;return}
 }
 if(request!==pathRequest)return;
 pathReplay={initial:{},events,frames,path};
 $('visualRate').value=String(liveRate);startHistory();
};

function syncPlayback(){if(presented)presented.playback={playing:history?historyPlaying:livePlaying,rate:history?Number($('visualRate').value):liveRate,waiting:history?null:liveWaiting}}
export function setLivePlayback(playing,rate,waiting=null){livePlaying=playing;liveRate=Number(rate)||1;liveWaiting=waiting;syncPlayback()}
function stop(){if(timer)clearTimeout(timer);timer=null;historyPlaying=false;syncPlayback();$('visualPlay').textContent='播放已执行事件'}
function render(){
 if(!latest?.nodes)return;
 if(watchView&&(!history||pathReplay?.stream)){renderWatch(watchView);return}
 const replay=replayData(),events=replay.events;
 cursor=Math.max(-1,Math.min(cursor,events.length-1));
 const terminal=['completed','cancelled','failed'].includes(latest.state);
 const displayIndex=cursor<0&&!history&&!liveCursor.pending?events.length-1:cursor;
 const runtime=RuntimeVisual.stateAt(replay.initial,events,displayIndex);
 const time=displayIndex>=0?runtime.time_s:events[0]?.time_s??latest.time_s;
 runtime.time_s=time;
 const held=replay.frames.filter(f=>f.event_order==null?f.time_s<=time:f.event_order<=(runtime.event.event_order??-1)).at(-1)||replay.frames[0];
 if(!held)return;
 // The live snapshot has backend geometry already. Rendering must not wait
 // for environment-index exports, which may only become available on finish.
 const snapshot=history||liveCursor.pending?{...held,time_s:time}: {time_s:time,physical_time_s:latest.geometry_sample_s,nodes:latest.nodes,links:latest.links};
 const combinedPaths={...pathContext,paths:[...new Map([...(pathContext?.paths||[]),...searchedPaths,...(pathReplay?.path?[pathReplay.path]:[])].map(p=>[p.id,p])).values()]};
 presented={snapshot,data:{sun_positions:latest.sun_positions||[],network:{nodes:latest.nodes,faults:runtime.faults,inter_cell_links:latest.inter_cell_links},time:{step_s:latest.step_s,duration_s:latest.end_s}},
  runtime,pathContext:combinedPaths,record:{request:{request_id:runtime.request_id}},route:{path:runtime.activeTransfers},kind:'all',revision:latest.version+'|'+cursor+'|'+history+'|'+(pathContext?.through_event_count||0)+'|'+pathSearchRevision};
 syncPlayback();
 const label=displayEvent(runtime.event)?runtime.event.event_label||callbacks.label?.[runtime.event.kind]||runtime.event.kind:terminal?'仿真已结束':'等待下一事件';
 runtime.event={...runtime.event,event_label:label};
 $('visualCursor').max=Math.max(0,events.length-1);$('visualCursor').value=Math.max(0,cursor);
 $('visualTime').textContent=`${time.toFixed(9)} s，${label}`;
 $('geometryStatus').textContent=`${history?'已执行事件回看':'逐事件展示'}，${time.toFixed(9)} s，${label}`;
 $('visualLive').disabled=!history&&!liveCursor.pending;$('visualPrev').disabled=!events.length||cursor<=0;$('visualNext').disabled=!events.length||cursor>=events.length-1;
 $('visualPlay').disabled=!events.length;$('visualCursor').disabled=!events.length;
 $('visualMode').textContent=streamingLive()?'':pathReplay?'回放所选传输，不改变后台仿真':history?'回看已执行事件':liveCursor.pending?`还有 ${liveCursor.pending} 条已执行事件待展示`:terminal?'实验已结束，可回看事件':'已显示当前事件，可操作节点与链路';
 callbacks.onMode?.();callbacks.onPresent?.(runtime);
 if(!loading)loading=import('/assets/globe.js').catch(error=>{$('geometryStatus').textContent='3D 加载失败：'+error.message;loading=null});
}
function seek(index){stop();callbacks.onHistory?.();history=true;cursor=index;render();const event=replayData().events[cursor];if(event)callbacks.onEvent?.(event,false)}
$('visualPrev').onclick=()=>seek(cursor-1);$('visualNext').onclick=()=>seek(cursor+1);
$('visualCursor').oninput=()=>seek(Number($('visualCursor').value));
$('visualRate').onchange=syncPlayback;
$('visualLive').onclick=()=>showCurrentLiveEvent();
function startHistory(){
 if(cursor>=replayData().events.length-1)cursor=-1;
 historyPlaying=true;$('visualPlay').textContent='暂停事件回看';
 const tick=()=>{const next=replayData().events.findIndex((e,i)=>i>cursor&&displayEvent(e));if(next<0){stop();return}cursor=next;render();timer=setTimeout(tick,1000/Number($('visualRate').value))};tick();
}
$('visualPlay').onclick=()=>{if(timer){stop();return}callbacks.onHistory?.();history=true;startHistory()};
export function updateLiveScene(state,handlers){
 callbacks=handlers;latest=state;liveCursor.update(state.replay?.events||[]);
 for(const frame of state.replay?.frames||[])watchFrames.set(frame.event_order??frame.time_s,frame);
 const ordered=[...watchFrames.entries()].sort((a,b)=>a[1].time_s-b[1].time_s);while(ordered.length>12){const [key]=ordered.shift();watchFrames.delete(key)}
 if(!history)cursor=liveCursor.index;
 render();
 if(!watchPlayer?.index)streamIndex().then(async()=>{if(watchPlayer?.index&&!watchView&&!watchBusy&&watchPlayer.time===null){watchView=await watchPlayer.at(watchPlayer.clock.toSimulation(0));if(watchView)renderWatch(watchView)}}).catch(()=>{});
 loadPaths();
}
export function hasPendingLiveEvents(){return streamingLive()?watchPlayer.pending:liveCursor.pending>0}
export function showCurrentLiveEvent(){pathRequest++;stop();pathReplay=null;history=false;detachedWatch=false;watchView=null;if(watchPlayer){watchPlayer.time=latest.time_s;watchPlayer.select(null)}liveCursor.catchUp();cursor=liveCursor.index;render()}
export function nextLiveEvent(){
 pathRequest++;stop();pathReplay=null;history=false;const next=liveCursor.next();if(next===null)return false;
 cursor=next;render();callbacks.onEvent?.(latest.replay.events[cursor],false);return true;
}
export function inspectLiveEvent(event){
 pathRequest++;
 pathReplay=null;
 const index=latest?.replay?.events.findIndex(e=>e.event_order===event.event_order)??-1;if(index>=0)seek(index);
}
// Buffered live playback is still live mode. Only explicit history/path/back
// navigation detaches the view; new solver batches never reveal Return Live.
export function viewingLive(){return !history&&!pathReplay&&!detachedWatch}
