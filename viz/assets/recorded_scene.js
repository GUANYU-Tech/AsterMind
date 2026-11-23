import {get} from '/assets/studio_common.js';
import {loadReplayJson} from '/assets/replay_page_loader.js';
import {isTransmissionEvent,watchTimeline,communicationPositionAt,countAt,compareTime,formatTime} from '/assets/communication_clock.js';
import {renderEventRows} from '/assets/event_view.js';
import '/assets/visual_state.js';
import '/assets/runtime_state.js';
import '/assets/constellation_layout.js';

export async function mountRecordedScene(main,identity){
const $=id=>main.querySelector('#'+id),base='/runs/'+encodeURIComponent(identity)+'/';
main.classList.add('scene-record');
main.innerHTML=`<div class="recorded-controls" aria-label="场景回放控制">
 <button id="recordPlay" disabled>播放回放</button><button id="recordPrevious" disabled>上一时刻</button><button id="recordNext" disabled>下一时刻</button><button id="recordRestart" disabled>重新开始</button>
 <label>回放<select id="recordMode"><option value="smooth">流畅观看</option><option value="inspect">逐时刻检查</option></select></label>
 <label>观看倍速<select id="recordRate"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="5">5×</option></select></label>
 <output id="recordClock">0.000000 s</output><input id="recordSeek" type="range" min="0" max="0" step="any" value="0" aria-label="仿真时间" disabled>
 <span id="recordPosition"></span></div><p id="recordStatus" role="status">正在读取回放…</p><div id="globe"></div><p id="hopInfo"></p><p id="nodeInfo"></p><p id="recordSelection"></p>
 <details class="recorded-log"><summary>通信记录</summary><table><thead><tr><th>时间 / s</th><th>通信状态</th><th>请求 / 对象</th><th>链路</th></tr></thead><tbody id="recordEvents"></tbody></table></details>`;
const style=document.createElement('link');style.rel='stylesheet';style.href='/assets/recorded_scene.css';document.head.append(style);
if(main.tagName==='MAIN'){
 document.title='场景回放 已保存记录';
 const heading=document.createElement('h1');heading.textContent='场景回放';main.prepend(heading);
}
window.runtimeGlobeMode=true;window.runtimeLiveMode=true;window.runtimeRecordedMode=true;
let catalog,model,presented,current=0,time=0,running=false,timer=null,revision=0,seekVersion=0,epoch=0,ready=false,globeReady=false,pagePaths=[],globalClock,selectedClock=null;
let exactTime=null;
let selected={},searchedPaths=[],searchRevision=0;
const pages=new Map(),frames=new Map();
const recent=new Map();
const allPaths=()=>[...new Map([...pagePaths,...searchedPaths].map(path=>[path.id,path])).values()];
window.networkReplayState=()=>presented;
function pause(){running=false;clearTimeout(timer);epoch++;seekVersion++;const end=selectedClock?selectedClock.toSimulation(selectedClock.end):catalog?.end_s;$('recordPlay').textContent=time>=end?'重新播放':'播放回放';if(presented)presented.playback.playing=false}
function inspect(){const node=presented?.snapshot.nodes.find(n=>n.node_id===selected.nodeId),link=presented?.snapshot.links.find(l=>l.link_id===selected.linkId);$('recordSelection').textContent=node?node.node_id+' '+node.state:link?link.endpoint_a+' ↔ '+link.endpoint_b+' '+link.state:''}
window.inspectNetworkNode=id=>{selected={nodeId:id};window.selectGlobeObject?.(selected);inspect()};
window.inspectNetworkLink=id=>{selected={linkId:id};window.selectGlobeObject?.(selected);inspect()};
window.openComputeTopology=id=>{const node=presented?.snapshot.nodes.find(n=>n.cell_id===id);if(node)window.inspectNetworkNode(node.node_id);return true};
window.queryRuntimePaths=async query=>{
 const serial=++searchRevision;
 if(!query.trim()){searchedPaths=[];if(presented){presented.pathContext={paths:pagePaths};presented.revision=++revision}return}
 try{const result=await get(base+'flow-paths?query='+encodeURIComponent(query.trim())+'&through='+((presented?.runtime.event.event_order??0)+1));if(serial===searchRevision&&presented){searchedPaths=result.paths||[];presented.pathContext={paths:allPaths()};presented.revision=++revision}}catch{}
};
window.inspectRuntimePath=async path=>{
 pause();recent.clear();const hops=path.hops.filter(h=>h.source_node_id!==h.target_node_id);
 const intervals=path.time_semantics==='whole_path_shared_resources'?[[path.start_s,path.terminal_s??catalog.end_s]]:hops.map(h=>[h.sent_s??h.start_s,h.arrived_s??h.cancelled_s??h.sent_s??h.start_s]);
 if(intervals.length&&path.terminal_s>intervals.at(-1)[1])intervals.at(-1)[1]=path.terminal_s;
 selectedClock=watchTimeline(intervals,.35);
 await seekTime(hops[0]?.start_s??path.start_s);start();
};
async function loadPage(index){
 if(!pages.has(index)){
  const page=await loadReplayJson(base+'native-replay/'+catalog.pages[index].file,{events:true});
  pages.set(index,page);while(pages.size>3)pages.delete(pages.keys().next().value);
 }
 return pages.get(index);
}
async function physical(at){
 const descriptors=model.artifacts.frames,index=Math.max(0,descriptors.findLastIndex(f=>f.time_s<=at)),descriptor=descriptors[index];
 if(!frames.has(descriptor.artifact)){frames.set(descriptor.artifact,await loadReplayJson(base+descriptor.artifact));while(frames.size>3)frames.delete(frames.keys().next().value)}
 return frames.get(descriptor.artifact);
}
async function pageAt(at){
 return communicationPositionAt(catalog.pages,loadPage,at);
}
async function arrivals(from,to,includeStart=false){
 const result=[];
 for(let i=0;i<catalog.pages.length;i++){
  const descriptor=catalog.pages[i];if(descriptor.end_s<from||descriptor.start_s>to)continue;
  const page=await loadPage(i);
  for(const e of page.events)if(['hop_arrival','transfer_cancelled'].includes(e.kind)&&isTransmissionEvent(e)&&(e.time_s>from||includeStart&&e.time_s===from)&&e.time_s<=to)result.push(e);
 }
 return result;
}
async function seekTime(at,animate=false){
 const own=++seekVersion,target=typeof at==='object'?at:null;at=Math.max(0,Math.min(Number(target?.time_s??at),catalog.end_s));
 const {page,pageIndex,visible,cut}=await pageAt(target??at),event=page.events[page.visible[visible]]||{kind:'waiting',event_label:'等待通信',time_s:at,event_order:-1};
 const runtime=RuntimeVisual.stateAt(page.initial,page.events,cut,event.request_id),held=await physical(at);
 runtime.event=event;runtime.time_s=at;runtime.time_ticks=target?.time_ticks;runtime.observed_order=page.events[cut]?.event_order??-1;runtime.time_group=true;
 runtime.activeTransfers=runtime.activeTransfers.filter(e=>e.source_node_id!==e.target_node_id);
 const arriving=await arrivals(animate?time:at,at,!animate),now=performance.now();
 if(own!==seekVersion)return;
 if(!animate)recent.clear();
 for(const [id,row]of recent)if(now-row.shownAt>500)recent.delete(id);
 for(const event of arriving)recent.set(event.transfer_id,{...event,shownAt:now});
 while(recent.size>16384)recent.delete(recent.keys().next().value);
 const links=new Map(held.links.map(link=>[link.link_id,{...link}]));
 const apply=e=>{
  if(e.kind==='link_state'&&e.time_s>=held.time_s&&links.has(e.link_id))Object.assign(links.get(e.link_id),{state:e.state,reason:e.reason});
  if(e.kind==='topology_command'&&links.has(e.target_id)){const link=links.get(e.target_id);if(e.action==='disconnect')Object.assign(link,{state:'DOWN',manual_disconnected:true});if(e.action==='connect')Object.assign(link,{manual_disconnected:false,...e.link_state})}
 };
 (page.link_states||[]).forEach(apply);(page.commands||[]).forEach(apply);page.events.slice(0,cut+1).forEach(apply);
 const snapshot=RuntimeVisual.applyFaults({...held,time_s:at,links:[...links.values()]},runtime.faults);
 current=catalog.pages[pageIndex].offset+visible;time=at;exactTime=target;pagePaths=page.paths||[];presented={snapshot,data:model,runtime,recentTransfers:[...recent.values()],route:{path:runtime.activeTransfers},record:{request:{request_id:runtime.request_id}},pathContext:{paths:allPaths()},kind:'all',revision:++revision,playback:{playing:running,rate:Number($('recordRate').value)}};
 const completed=target?.time_ticks&&catalog.completion_ticks?countAt(catalog.completion_ticks,target):catalog.completion_times?countAt(catalog.completion_times,at):(catalog.pages[pageIndex].completed_before||0)+page.events.slice(0,cut+1).filter(e=>e.kind==='external_flow_done').length;
 const failed=target?.time_ticks&&catalog.failure_ticks?countAt(catalog.failure_ticks,target):countAt(catalog.failure_times||[],at);
 $('recordSeek').value=at;$('recordClock').textContent=($('recordMode').value==='inspect'?formatTime(target??at):at.toFixed(6))+' s';$('recordClock').title=formatTime(target??at)+' s'+(catalog.clock?' 统一时钟 1 as':' 旧时钟记录');$('recordPosition').textContent=catalog.transfer_count?(catalog.unit?'完成'+catalog.unit+' ':'完成传输 ')+completed.toLocaleString()+' / '+catalog.transfer_count.toLocaleString()+(failed?' 失败 '+failed.toLocaleString():''):'通信回放';
 $('recordStatus').textContent=event.event_label||event.kind;$('recordPrevious').disabled=current<=0;$('recordNext').disabled=current>=catalog.event_count-1;
 if(!running)$('recordPlay').textContent=at>=catalog.end_s?'重新播放':'播放回放';
 renderEventRows($('recordEvents'),page.events.slice(0,cut+1).filter(isTransmissionEvent).slice(-50));inspect();
 if(!globeReady){
  globeReady=true;await import('/assets/globe.js');
  const links=$('showLinks')?.parentElement;if(links)$('recordClock').before(links);
 }
}
async function adjacent(direction){
 const target=exactTime??time,entry=await pageAt(target);let i=entry.pageIndex,local=entry.visible+direction;
 while(i>=0&&i<catalog.pages.length){
  const page=await loadPage(i);
  for(;local>=0&&local<page.visible.length;local+=direction){const event=page.events[page.visible[local]],order=compareTime(event,target);if(direction>0?order>0:order<0)return {time_s:event.time_s,time_ticks:event.time_ticks}}
  i+=direction;if(i>=0&&i<catalog.pages.length)local=direction>0?0:(await loadPage(i)).visible.length-1;
 }
 return null;
}
function start(){
 if(!ready||running)return;running=true;const token=++epoch;$('recordPlay').textContent='暂停回放';
 const clock=selectedClock||globalClock;let watch=time>=clock.toSimulation(clock.end)?0:clock.toWatch(time),last=performance.now();
 const tick=async()=>{try{
  const speed=Number($('recordRate').value),now=performance.now(),inspect=$('recordMode').value==='inspect';
  watch+=Math.min(.5,(now-last)/1000)*speed;last=now;
  const next=inspect?await adjacent(1):clock.toSimulation(watch);
  if(next===null){pause();return}await seekTime(next,true);if(token!==epoch)return;
  if(inspect?compareTime(exactTime??{time_s:time},{time_s:catalog.end_s,time_ticks:catalog.end_ticks})>=0:watch>=clock.end){pause();return}
  timer=setTimeout(tick,inspect?600/speed:80);
 }catch(error){pause();$('recordStatus').textContent=error.message}};
 if(presented)presented.playback.playing=true;timer=setTimeout(tick,80);
}
const move=async at=>{pause();selectedClock=null;try{await seekTime(at)}catch(error){$('recordStatus').textContent=error.message}};
const step=async direction=>{pause();selectedClock=null;try{const at=await adjacent(direction);if(at!==null)await seekTime(at)}catch(error){$('recordStatus').textContent=error.message}};
$('recordPlay').onclick=()=>running?pause():start();$('recordPrevious').onclick=()=>step(-1);$('recordNext').onclick=()=>step(1);$('recordRestart').onclick=()=>move(0);
$('recordSeek').oninput=()=>move($('recordSeek').value);$('recordRate').onchange=()=>{if(presented)presented.playback.rate=Number($('recordRate').value)};
$('recordMode').onchange=()=>{pause();$('recordClock').textContent=($('recordMode').value==='inspect'?formatTime(exactTime??time):time.toFixed(6))+' s'};
window.addEventListener('message',event=>{if(event.origin===location.origin&&event.data?.type==='analysis-visibility'&&!event.data.visible)pause()});
main.addEventListener('analysis-visibility',event=>{if(!event.detail.visible)pause()});
try{
 model=await get(base+'environment-index');
 const response=await fetch(base+'native-replay/transmission-recording.json');
 if(response.ok)catalog=await response.json();
 else{
  const log=await get(base+'scene-events'),visible=log.events.flatMap((e,i)=>isTransmissionEvent(e)?[i]:[]);
  catalog={event_count:visible.length,end_s:log.events[visible.at(-1)]?.time_s||0,transfer_count:log.events.filter(e=>e.kind==='external_flow_done').length,pages:[{offset:0,count:visible.length,start_s:0,end_s:log.end_s,last_order:log.events.at(-1)?.event_order}]};
  pages.set(0,{initial:{},events:log.events,visible});
 }
 if(!catalog.event_count)throw Error('此记录没有通信传输');
 globalClock=watchTimeline(catalog.busy_intervals?.length?catalog.busy_intervals:[[0,catalog.end_s]]);
 $('recordSeek').max=catalog.end_s;await seekTime(0);ready=true;
 for(const id of ['recordPlay','recordRestart','recordSeek'])$(id).disabled=false;
}catch(error){$('recordStatus').textContent='回放读取失败：'+error.message}
}
