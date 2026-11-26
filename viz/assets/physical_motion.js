import {createMotionFrames} from '/assets/motion_frames.js';
import {createTimeClock} from '/assets/time_clock.js';
const $=id=>document.getElementById(id),initial=window.networkReplayState(),data=initial.data;
if(data.motion_frames?.length){
 const root=location.pathname.replace(/viewer$/,''),motion=createMotionFrames(root,data.motion_frames),frames=data.artifacts?.frames||data.snapshots;
 const label=document.createElement('label');label.textContent='连续播放倍速 ';
 const speed=document.createElement('select');speed.setAttribute('aria-label','连续播放倍速');
 for(const rate of [1,10,60,300]){const option=document.createElement('option');option.value=rate;option.textContent=rate+'×';speed.append(option)}speed.value='60';label.append(speed);$('play').parentElement.append(label);
 const requested=Number(new URLSearchParams(location.search).get('time'));
 let time=Number.isFinite(requested)&&requested>=0&&requested<=data.time.end_s?requested:initial.snapshot.time_s,baseIndex=-1,base=null,version=0,stateKey='',links=null;
 async function render(target){
  const own=++version,i=Math.max(0,MotionState.heldIndex(frames,target));
  if(i!==baseIndex||!base){await window.seekPhysicalSample(i);base=window.physicalSnapshot();baseIndex=i;stateKey=''}
  const {left,right}=await motion.at(target);if(own!==version)return;
  const events=base.state_events||data.events||[],end=MotionState.heldIndex(events,target),key=i+'|'+end;
  if(key!==stateKey){const edges=new Map(base.links.map(e=>[e.link_id,e]));for(const e of events.slice(0,end+1))if(e.time_s>=base.time_s&&edges.has(e.link_id))edges.set(e.link_id,{...edges.get(e.link_id),state:e.state,reason:e.reason});links=[...edges.values()];stateKey=key}
  window.motionPhysicalFrame=MotionState.pose({...base,links},left,right,target);time=target;
  $('clock').textContent=`${target.toFixed(3)} s / ${data.time.end_s} s`;
  $('status').textContent=`ITRS Hermite 位置插值 拓扑按物理状态变化切换 运动缓存 ${motion.size()}/6 帧`;
 }
 const player=createTimeClock({render,onStop:()=>$('play').textContent='连续播放',onError:error=>$('status').textContent=error.message});
 $('play').textContent='连续播放';$('play').onclick=()=>{if(player.running){player.stop();return}player.start(time>=data.time.end_s?0:time,data.time.end_s,Number(speed.value));$('play').textContent='暂停连续播放'};
 speed.onchange=()=>player.stop();
 for(const id of ['prev','next','time'])$(id).addEventListener(id==='time'?'input':'click',()=>{version++;player.stop();baseIndex=-1;setTimeout(()=>{time=window.physicalSnapshot().time_s},0)});
 document.querySelector('footer').textContent=`地固坐标系保持不变。物理采样 ${data.time.step_s} s；相邻位置和速度作 Hermite 插值，太阳方向平滑变化。箭头按钮逐采样核查；连续播放只加载邻近运动帧，不改变物理与业务结果。`;
 try { await render(time); }
 catch(error) { $('status').textContent=error.message; }
}
