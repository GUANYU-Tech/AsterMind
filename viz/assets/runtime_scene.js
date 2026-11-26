
import {createFrameCache} from '/assets/runtime_frames.js';
import {createMotionFrames} from '/assets/motion_frames.js';
import '/assets/runtime_time.js';
window.runtimeGlobeMode=true;
const status=document.getElementById('geometryStatus');
const root=location.pathname.replace(/runtime$/,'');
let environment,presented,version=0,topologyEvents=null,topologyPages=null,motion,logicalKey='',logicalSnapshot,logicalFaults;
const topologyCache=createFrameCache(async row=>{const r=await fetch(root+'runtime-data/'+row.artifact);if(!r.ok)throw Error('拓扑事件页读取失败');return r.json()},3);
const cache=createFrameCache(async descriptor=>{
 const response=await fetch(root+descriptor.artifact);
 if(!response.ok)throw Error('环境快照读取失败');
 return response.json();
});
const ready=(async()=>{
 const response=await fetch(root+'environment-index');
 if(!response.ok)throw Error('当前实验没有可用的按帧环境数据');
 environment=await response.json();
 if(environment.motion_frames?.length)motion=createMotionFrames(root,environment.motion_frames);
 const manifestResponse=await fetch(root+'runtime-data/index.json');
 if(manifestResponse.ok){const manifest=await manifestResponse.json(),artifact=manifest.environment?.topology_events_artifact;topologyPages=manifest.environment?.topology_pages;if(!topologyPages?.length&&artifact){const r=await fetch(root+'runtime-data/'+artifact);if(!r.ok)throw Error('实际故障与拓扑日志读取失败');topologyEvents=await r.json()}}
 return environment.artifacts.frames;
})();
window.networkReplayState=()=>presented;
window.setRuntimeGeometry=async(runtime,record)=>{
 const own=++version;
 if(!presented)status.textContent='正在定位事件对应的环境快照…';
 try{
  const frames=await ready,frameIndex=RuntimePlayback.heldFrameIndex(frames,runtime.time_s);
  if(frameIndex<0)throw Error('事件发生在环境记录起点之前');
  const pose=motion?motion.at(runtime.time_s):Promise.resolve(null);
  let snapshot=await cache.get(frames[frameIndex]);
  const order=runtime.event.time_s===runtime.time_s?(runtime.event.event_order??Infinity):Infinity;
  const before=e=>e.time_s<runtime.time_s||e.time_s===runtime.time_s&&e.event_order<=order;
  let topology=topologyEvents,topologyKey='legacy';
  if(topologyPages?.length){let page=0;for(let i=1;i<topologyPages.length;i++){const p=topologyPages[i];if(p.first_time_s>runtime.time_s||p.first_time_s===runtime.time_s&&p.first_event_order>order)break;page=i}const data=await topologyCache.get(topologyPages[page]);let cut=MotionState.heldIndex(data.events,runtime.time_s);while(cut>=0&&!before(data.events[cut]))cut--;topology=[...data.initial.filter(before),...data.events.slice(0,cut+1)];topologyKey=page+'|'+cut}
  const key=frameIndex+'|'+topologyKey+'|'+(topologyPages?.length?'':runtime.event.event_order);
  if(key!==logicalKey){
  const observed=(topology||[]).filter(before);
  const faults=topology?RuntimeVisual.stateAt({},observed,observed.length-1).faults:(environment.network.faults||[]).filter(f=>f.start_s<=runtime.time_s&&runtime.time_s<f.end_s);
  const edges=new Map(snapshot.links.map(e=>[e.link_id,e]));
  for(const e of observed){if(e.kind==='topology_command'){
   if(e.action==='disconnect'&&edges.has(e.target_id))edges.set(e.target_id,{...edges.get(e.target_id),state:'DOWN',manual_disconnected:true,reason:'manual_link_disconnected'});
   if(e.action==='connect'){const edge=e.edge||edges.get(e.target_id);if(edge)edges.set(e.target_id,{...edge,...e.link_state,manual_disconnected:false})}
  }else if(e.kind==='link_state'&&edges.has(e.link_id))edges.set(e.link_id,{...edges.get(e.link_id),state:e.state,reason:e.reason})}
  snapshot=RuntimeVisual.applyFaults({...snapshot,links:[...edges.values()]},faults);
  logicalSnapshot=snapshot;logicalFaults=faults;logicalKey=key;
  }
  snapshot=logicalSnapshot;
  const bracket=await pose;
  if(bracket)snapshot=MotionState.pose(snapshot,bracket.left,bracket.right,runtime.time_s);
  if(own!==version)return;
  const nodes=environment.network.nodes,comm=nodes.filter(n=>n.role==='communication').length,compute=nodes.filter(n=>n.role==='compute').length,ground=nodes.filter(n=>n.role==='ground').length;
  document.getElementById('constellationInfo').textContent=comm+' 颗通信卫星 + '+compute+' 颗计算卫星 = '+(comm+compute)+' 颗物理卫星 '+ground+' 个地面站 仿真时长 '+environment.time.duration_s+' s 当前 UP 链路 '+snapshot.links.filter(l=>l.state==='UP').length;
  presented={orbitSnapshot:logicalSnapshot,data:{...environment,network:{...environment.network,faults:logicalFaults}},snapshot,kind:'all',runtime,record};
  status.textContent='播放 '+runtime.time_s.toFixed(6)+' s 物理采样 '+(snapshot.physical_time_s??snapshot.time_s)+' s '+(motion?'ITRS Hermite 位置插值 业务状态遵循实际事件':'位置保持（旧数据无运动帧）')+' 缓存 '+cache.size()+' 拓扑 / '+(motion?.size()||0)+' 运动帧';
  // Preload the next two geometry frames; every subsecond hop shares this frame.
  for(const frame of frames.slice(frameIndex+1,frameIndex+2))cache.get(frame).catch(()=>{});
 }catch(error){if(own===version)status.textContent='3D 环境不可用：'+error.message}
};
window.inspectNetworkNode=id=>{document.getElementById('nodeInfo').textContent='节点：'+id};
window.inspectNetworkLink=id=>{document.getElementById('nodeInfo').textContent='链路：'+id};
if(window.currentRuntimeState)window.setRuntimeGeometry(window.currentRuntimeState.state,window.currentRuntimeState.record);
await ready;
await import('/assets/globe.js');
