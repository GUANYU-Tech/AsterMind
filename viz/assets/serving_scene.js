import '/assets/motion_state.js';
import {createFrameCache} from '/assets/runtime_frames.js';
const $=id=>document.getElementById(id),query=new URLSearchParams(location.search);
for(const id of ['users','parallel','backend'])if([...$(id).options].some(o=>o.value===query.get(id)))$(id).value=query.get(id);
const users=$('users').value,parallel=$('parallel').value,backend=$('backend').value;
window.runtimeGlobeMode=true;window.collectivePhysicalMembers=true;window.collectiveModelScale=.5;window.collectiveVisualScale=.6;window.servingFlowView=backend;
let source,data,presented,time=2,playing=false,version=0,tracking=false,previousCenter;
const format=(v,d=3)=>Number(v).toFixed(d),safe=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function trafficName(h){const id=h.flow_id||'';return id.includes('pp-')?(id.includes('residual')?'PP residual':'PP hidden'):id.includes('ep-output')?'EP reduce':id.includes('all-reduce')?'TP reduce':h.stage;}
function laneName(h){return trafficName(h)+' '+h.source.split('-').at(-1)+'→'+h.target.split('-').at(-1);}
const args=new URLSearchParams({users,parallel,backend});
async function read(url){const r=await fetch(url),v=await r.json();if(!r.ok)throw Error(v.error||'数据不可用');return v;}
const frames=createFrameCache(row=>read('/serving/frame?parallel='+parallel+'&frame='+row.index),4);
window.networkReplayState=()=>presented;
window.inspectNetworkNode=id=>$('details').textContent=JSON.stringify(presented.snapshot.nodes.find(n=>n.node_id===id),null,2);
window.inspectNetworkLink=id=>{if([...$('link').options].some(o=>o.value===id))$('link').value=id;showLink();$('details').textContent=JSON.stringify(presented.snapshot.links.find(n=>n.link_id===id),null,2);};
function record(){return data.records[Number($('request').value)||0];}
function pathLeg(h){return {...h,source_node_id:h.source,target_node_id:h.target,payload_bytes:h.bytes,
 segments:h.segments.map(s=>({start_s:s.interval_start_s,finish_s:s.interval_end_s,bandwidth_bps:s.allocated_bandwidth_bps,bytes:s.bytes_sent})),
 status:time<h.start_s?'future':time<h.finish_s?'transmitting':time<h.arrival_s?'propagating':'delivered'};}
function timeline(){
 const r=record(),id=r.request.request_id,events=data.events.filter(e=>e.request_id===id),find=kind=>events.find(e=>e.kind===kind)?.time_s;
 const arrival=find('request_arrival'),input=find('input_done'),first=find('astermind_first_token'),finish=find('astermind_compute_complete'),reply=find('request_complete');
 const native=data.hops.filter(h=>h.stage==='astermind_collective'&&h.source.startsWith(r.decision.serving_compute_node_id.replace(/-0$/,'-')));
 const pp=native.find(h=>h.flow_id?.includes('pp-')),ep=native.find(h=>h.flow_id?.includes('ep-output'));
 const steps=[['用户进入通信壳层',arrival,input],['并行推理 / 首 token',input,first],['Decode / 剩余输出',first,finish],['回复到达用户',finish,reply]];
 $('phases').innerHTML=steps.map(([name,start,end])=>`<div class="phase ${time>=end?'done':time>=start?'active':''}"><b>${name}</b><br>${format(start)}–${format(end)} s</div>`).join('');
 $('jumps').replaceChildren();
 for(const [name,at] of [['用户接入',arrival],['通信壳层转发',arrival+.14],['TP / EP 通信',input+.05],['PP 传输',pp?pp.start_s:null],['EP 输出归约',ep?ep.start_s:null],['回复用户',reply-.0002]]){
  if(at==null)continue;const b=document.createElement('button');b.textContent=name;b.onclick=()=>seek(at);$('jumps').append(b);
 }
 const node=r.decision.serving_compute_node_id,requestHops=data.hops.filter(h=>h.request_id===id),cellHops=data.hops.filter(h=>h.source.startsWith(node.replace(/-0$/,'-'))&&h.stage==='astermind_collective');
 const ids=[...new Set([...requestHops,...cellHops].map(h=>h.link_id))];$('link').replaceChildren();for(const id of ids){const o=document.createElement('option');o.value=id;o.textContent=id;$('link').append(o);}showLink();
}
function showLink(){
 if(!data)return;const link=$('link').value,hops=data.hops.filter(h=>h.link_id===link),flowMap=new Map(hops.map(h=>[h.transfer_id,h]));
 const palette=['#62d5ff','#ffbc65','#c49cff','#79e2b5','#f18bc7'];
 let svg='';
 if(backend==='ns3'){
  const packets=data.packets.filter(p=>flowMap.has(p.transfer_id)),nearest=Math.max(0,packets.findIndex(p=>p.finish_s>=time));
  const visible=packets.slice(Math.max(0,nearest-4),Math.max(0,nearest-4)+36);
  if(!visible.length){$('linkTimeline').textContent='此链路没有包发送事件';return;}
  const start=visible[0].start_s,end=Math.max(...visible.map(p=>p.finish_s)),span=Math.max(1e-9,end-start),owners=[...new Set(visible.map(p=>p.transfer_id))];
  svg=`<svg viewBox="0 0 900 ${55+owners.length*40}" style="width:100%;min-width:650px"><text x="10" y="18" fill="#aac4df" font-size="12">实际 ns-3 发送区间：${format(start,6)}–${format(end,6)} s</text>`;
  owners.forEach((id,i)=>svg+=`<text x="8" y="${51+i*40}" fill="${palette[i%5]}" font-size="11">${safe(laneName(flowMap.get(id)))}</text>`);
  for(const p of visible){const i=owners.indexOf(p.transfer_id),x=145+(p.start_s-start)/span*730,w=Math.max(1,(p.finish_s-p.start_s)/span*730);svg+=`<rect x="${x}" y="${31+i*40}" width="${w}" height="25" fill="${palette[i%5]}" stroke="#081421" stroke-width="1.4"><title>${safe(p.transfer_id)}；${p.payload_bytes} B 载荷 / ${p.wire_bytes} B 线上；${p.start_s}–${p.finish_s} s</title></rect>`;}
  if(time>=start&&time<=end)svg+=`<line x1="${145+(time-start)/span*730}" x2="${145+(time-start)/span*730}" y1="22" y2="100%" stroke="#fff" stroke-width="2"/>`;svg+='</svg>';
  $('packetNote').textContent=`当前链路 ${packets.length} 个真实包发送完成事件；方块不是随机动画。3D 小方块代表当前正在序列化的包，沿链路的位置表示包在整条消息内的字节位置，不表示真实飞行位置。`;
  const current=packets.find(p=>p.start_s<=time&&time<p.finish_s);
  $('packetDetail').textContent=current?`正在发送 ${trafficName(flowMap.get(current.transfer_id))}：载荷 ${current.payload_bytes} B / 线上 ${current.wire_bytes} B；${format(current.start_s,9)} → ${format(current.finish_s,9)} s；消息内偏移 ${current.offset_bytes} B。`:'当前游标处无数据包正在序列化。白线为统一仿真时刻；不同颜色行区分消息，反向链路可同时发送。';
 }else{
  const transmitting=hops.filter(h=>h.start_s<=time&&time<h.finish_s),nearby=hops.filter(h=>h.start_s<=time+.25&&h.finish_s>=time-.25).sort((a,b)=>Math.abs(a.start_s-time)-Math.abs(b.start_s-time));
  const visible=(transmitting.length?transmitting:nearby.length?nearby:hops).slice(0,8),start=Math.min(...visible.map(h=>h.start_s)),end=Math.max(...visible.map(h=>h.finish_s)),span=Math.max(1e-8,end-start);
  svg=`<svg viewBox="0 0 900 ${50+visible.length*34}" style="width:100%;min-width:650px"><text x="10" y="18" fill="#aac4df" font-size="12">连续流发送区间：${format(start,6)}–${format(end,6)} s</text>`;
  visible.forEach((h,i)=>{svg+=`<text x="8" y="${49+i*34}" fill="${palette[i%5]}" font-size="11">${safe(laneName(h))}</text>`;for(const s of h.segments){const x=145+(s.interval_start_s-start)/span*730,w=Math.max(.5,(s.interval_end_s-s.interval_start_s)/span*730),height=4+21*Math.min(1,s.allocated_bandwidth_bps/s.link_capacity_bps);svg+=`<rect x="${x}" y="${55+i*34-height}" width="${w}" height="${height}" fill="${palette[i%5]}" opacity=".75"><title>${safe(h.transfer_id)}；${format(s.allocated_bandwidth_bps/1e6,3)} Mbps；${s.interval_start_s}–${s.interval_end_s} s</title></rect>`;}});if(time>=start&&time<=end)svg+=`<line x1="${145+(time-start)/span*730}" x2="${145+(time-start)/span*730}" y1="22" y2="100%" stroke="#fff" stroke-width="2"/>`;svg+='</svg>';
  $('packetNote').textContent='FlowSim 显示连续的流占用区间。3D 流带宽度随当前分配带宽变化，发送和完成事件会改变共享份额；此后端没有数据包。';
  $('packetDetail').textContent=transmitting.map(h=>{const s=h.segments.find(s=>s.interval_start_s<=time&&time<s.interval_end_s);return s?`${laneName(h)}：${format(s.allocated_bandwidth_bps/1e6,3)} / ${format(s.link_capacity_bps/1e6,3)} Mbps`:'';}).filter(Boolean).join('；')||'条带高度表示所分配带宽占该方向链路容量的比例。';
 }
 $('linkTimeline').innerHTML=svg;
}
let lastFrameKey='';
async function seek(value){
 if(!data)return;time=Math.max(0,Math.min(60,value));const own=++version;
 let i=0;while(i+1<source.frame_times.length&&source.frame_times[i+1]<=time)i++;
 const [left,right]=await Promise.all([frames.get({artifact:'f'+i,index:i}),frames.get({artifact:'f'+Math.min(i+1,source.frame_times.length-1),index:Math.min(i+1,source.frame_times.length-1)})]);
 if(own!==version)return;
 const u=right.time_s>left.time_s?(time-left.time_s)/(right.time_s-left.time_s):0,other=new Map(right.nodes.map(n=>[n.node_id,n]));
 const snapshot={...left,physical_time_s:left.time_s,time_s:time,nodes:left.nodes.map(n=>({...n,position_itrs_km:n.position_itrs_km.map((x,j)=>x+(other.get(n.node_id).position_itrs_km[j]-x)*u)}))};
 const r=record(),active=data.hops.filter(h=>h.start_s<=time&&time<h.arrival_s).map(pathLeg);
 presented={orbitSnapshot:left,loadOrbitSample:()=>right!==left ? right : i ? frames.get({artifact:'f'+(i-1),index:i-1}) : null,data:{network:source.network,time:source.time,sun_positions:source.sun_positions},snapshot,kind:'all',record:r,
  runtime:{time_s:time,event:{time_s:time,kind:'parallel_inference',event_order:time},globalMode:true,legs:active,activeTransfers:active,
   compute:'waiting',routePlans:[{reason:'initial',path:r.routing.selected_path.filter(h=>h.stage==='input')}],recoveryEvents:[],lossEvents:[]}};
 $('clock').textContent=format(time,6)+' s 快照 '+left.time_s+' s';$('time').value=time;
 $('active').innerHTML='<table><tr><th>实际传输</th><th>'+ (backend==='ns3'?'区间有效速率':'分配带宽')+'</th></tr>'+active.slice(0,12).map(h=>{const s=h.segments.find(s=>s.start_s<=time&&time<s.finish_s);return `<tr><td>${safe(h.source)} → ${safe(h.target)}<br><small>${safe(h.flow_id?.includes('pp-')?'PP activation':h.flow_id?.includes('ep-output')?'EP output all-reduce':h.flow_id?.includes('tp-all-reduce')?'TP all-reduce':h.stage)}</small></td><td>${s?format(s.bandwidth_bps/1e6,2)+' Mbps':'传播中'}</td></tr>`}).join('')+'</table>';
 const id=r.request.request_id,ev=data.events.filter(e=>e.request_id===id);
 const times=[ev.find(e=>e.kind==='request_arrival')?.time_s,ev.find(e=>e.kind==='input_done')?.time_s,ev.find(e=>e.kind==='astermind_first_token')?.time_s,ev.find(e=>e.kind==='astermind_compute_complete')?.time_s,ev.find(e=>e.kind==='request_complete')?.time_s];
 [...$('phases').children].forEach((node,k)=>node.className='phase '+(time>=times[k+1]?'done':time>=times[k]?'active':''));
 if(tracking&&window.orbitGlobe){const g=window.orbitGlobe,c=cellCenter();if(previousCenter){const delta=c.clone().sub(previousCenter);g.camera.position.add(delta);g.controls.target.add(delta);}previousCenter=c;}
 const key=i+'|'+Math.floor(time*(backend==='ns3'?1e7:500));if(key!==lastFrameKey){lastFrameKey=key;showLink();}
}
function cellCenter(){const g=window.orbitGlobe,ids=record().placement.compute_owner_satellite_ids,points=presented.snapshot.nodes.filter(n=>ids.includes(n.node_id)).map(n=>g.vector(n.position_itrs_km));return points.reduce((a,b)=>a.add(b),new g.THREE.Vector3()).multiplyScalar(1/points.length);}
function focusCell(){const g=window.orbitGlobe;if(!g)return;g.overlay?.stopFocus();const center=cellCenter(),out=center.clone().normalize(),side=new g.THREE.Vector3().crossVectors(out,new g.THREE.Vector3(0,1,0)).normalize();g.controls.target.copy(center);g.camera.position.copy(center).add(out.multiplyScalar(.38)).add(side.multiplyScalar(.18));g.controls.minDistance=.025;tracking=true;previousCenter=center;window.collectiveModelScale=.17;window.collectiveVisualScale=.08;}
function focusGlobal(){const g=window.orbitGlobe;if(!g)return;g.overlay?.stopFocus();tracking=false;window.collectiveModelScale=.5;window.collectiveVisualScale=.6;const ids=new Set(record().routing.selected_path.flatMap(h=>[h.source_node_id,h.target_node_id]));const dir=presented.snapshot.nodes.filter(n=>ids.has(n.node_id)).reduce((a,n)=>a.add(g.vector(n.position_itrs_km).normalize()),new g.THREE.Vector3()).normalize();g.controls.target.set(0,0,0);g.camera.position.copy(dir).multiplyScalar(19);g.controls.minDistance=6.6;}
$('cell').onclick=focusCell;$('global').onclick=focusGlobal;$('link').onchange=showLink;
$('request').onchange=()=>{timeline();seek(record().request.arrival_time_s);focusGlobal();};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'连续播放';};$('time').oninput=e=>seek(Number(e.target.value));
$('load').onclick=()=>location.href='/serving?'+new URLSearchParams({users:$('users').value,parallel:$('parallel').value,backend:$('backend').value});
$('run').onclick=async()=>{try{$('run').disabled=true;$('status').textContent='正在执行端到端推理仿真…';const selection={users:$('users').value,parallel:$('parallel').value,backend:$('backend').value},runArgs=new URLSearchParams(selection);const response=await fetch('/serving/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(selection)});const v=await response.json();if(!response.ok)throw Error(v.error);const poll=setInterval(async()=>{try{const s=await read('/serving/status?'+runArgs);if(s.state==='completed'){clearInterval(poll);location.href='/serving?'+runArgs;}else if(s.state==='failed'){clearInterval(poll);$('status').textContent=s.error;$('run').disabled=false;}}catch(e){clearInterval(poll);$('status').textContent=e.message;$('run').disabled=false;}},1500);}catch(e){$('status').textContent=e.message;$('run').disabled=false;}};
function focusLink(){const g=window.orbitGlobe;if(!g)return;const h=data.hops.find(h=>h.link_id===$('link').value),positions=new Map(presented.snapshot.nodes.map(n=>[n.node_id,g.vector(n.position_itrs_km)]));if(!h)return;g.overlay?.stopFocus();tracking=false;const a=positions.get(h.source),b=positions.get(h.target),center=a.clone().add(b).multiplyScalar(.5),out=center.clone().normalize(),side=new g.THREE.Vector3().crossVectors(out,new g.THREE.Vector3(0,1,0)).normalize(),distance=Math.max(.3,a.distanceTo(b)*2);g.controls.target.copy(center);g.camera.position.copy(center).add(out.multiplyScalar(distance)).add(side.multiplyScalar(distance*.3));g.controls.minDistance=.025;window.collectiveVisualScale=.15;window.collectiveModelScale=.17;}
$('packetFocus').onclick=()=>{const link=$('link').value,ids=new Set(data.hops.filter(h=>h.link_id===link).map(h=>h.transfer_id));if(backend==='ns3'){const p=data.packets.find(p=>ids.has(p.transfer_id));if(p){$('speed').value='0.0001';seek((p.start_s+p.finish_s)/2);focusLink();}}else{const h=data.hops.find(h=>h.link_id===link);if(h){$('speed').value='0.1';seek((h.start_s+h.finish_s)/2);focusLink();}}};
$('competition').onclick=async()=>{
 const allowed=new Set([...$('link').options].map(o=>o.value)),groups=new Map();let best;
 for(const h of data.hops){if(!allowed.has(h.link_id))continue;const key=h.link_id+'|'+h.source,rows=groups.get(key)||[];rows.push(h);groups.set(key,rows);}
 for(const rows of groups.values()){rows.sort((a,b)=>a.start_s-b.start_s);for(let i=0;i<rows.length;i++)for(let j=i+1;j<rows.length&&rows[j].start_s<rows[i].finish_s;j++){const a=rows[i],b=rows[j],span=Math.min(a.finish_s,b.finish_s)-b.start_s;if(span>1e-8&&(!best||span>best.span))best={link:a.link_id,at:b.start_s+span*.5,span};}}
 if(!best){$('packetDetail').textContent='当前路径未发现同向发送重叠。';return;}$('link').value=best.link;$('speed').value=backend==='ns3'?'0.0001':'0.01';await seek(best.at);focusLink();
};
try{
 $('status').textContent='读取完整壳层与推理结果…';[source,data]=await Promise.all([read('/serving/physical?parallel='+parallel),read('/serving/data?'+args)]);
 $('summary').textContent='1584 颗通信卫星（72×22） → '+(parallel==='tp_pp'?'8':'4')+' 颗晨昏计算卫星 / 2 个模型副本；已完成 '+data.summary.completed_request_count+'/'+data.summary.total_request_count+' 请求，'+Object.values(data.kernels).reduce((a,b)=>a+b,0)+' 个 Midlayer kernel 时延事件。';
 $('deployment').textContent=parallel==='tp_pp'?'Qwen3-4B：每副本 PP=2、每阶段 TP=2，共 4 颗计算卫星。两个独立副本按未完成请求量分流。':'Qwen3-30B-A3B：每副本 TP=EP=2，MoE 专家分布在两颗计算卫星。两个独立副本按未完成请求量分流。';
 $('renderLegend').textContent=backend==='flowsim'?'连续流带 宽度表示带宽份额':'离散方块 实际正在发送的数据包';
 data.records.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.request.request_id+' → '+r.decision.serving_compute_node_id;$('request').append(o);});
 timeline();await seek(record().request.arrival_time_s);await import('/assets/globe.js');document.getElementById('showLinks').checked=false;focusGlobal();
 const comparison=await read('/serving/compare?'+args);$('compare').innerHTML='<table><tr><th>同输入结果</th><th>FlowSim</th><th>ns-3</th></tr>'+[['平均 E2E / s','mean_e2e_s'],['平均 TTFT / s','mean_ttft_s'],['逐跳有效字节','payload_hop_bytes']].map(([label,k])=>`<tr><td>${label}</td>${['flowsim','ns3'].map(mode=>`<td>${comparison[mode]?format(comparison[mode][k],k.includes('bytes')?0:6):'未运行'}</td>`).join('')}</tr>`).join('')+`<tr><td>实际发送包数</td><td>无包模型</td><td>${comparison.ns3?.native_counters?.packets_started??'未记录'}</td></tr><tr><td>实际线速字节</td><td>有效字节连续流</td><td>${comparison.ns3?.native_counters?.wire_bytes_started??'未记录'}</td></tr></table>`;
 const g=window.orbitGlobe,packetGroup=new g.THREE.Group();g.scene.add(packetGroup);
 const cubes=Array.from({length:300},()=>{const m=new g.THREE.Mesh(new g.THREE.BoxGeometry(.012,.012,.012),new g.THREE.MeshBasicMaterial({color:0xffde70}));m.visible=false;packetGroup.add(m);return m;});
 const hopMap=new Map(data.hops.map(h=>[h.transfer_id,h]));const packets=data.packets||[];
 let last=performance.now();function animate(now){const dt=Math.min(.1,(now-last)/1000);last=now;if(playing){seek(Math.min(60,time+dt*Number($('speed').value)));if(time>=60)playing=false;}
  if(backend==='ns3'&&presented){const positions=new Map(presented.snapshot.nodes.map(n=>[n.node_id,g.vector(n.position_itrs_km)]));cubes.forEach(m=>m.visible=false);let index=0;for(const p of packets){if(p.start_s<=time&&time<p.finish_s&&index<cubes.length){const h=hopMap.get(p.transfer_id),m=cubes[index++];m.position.copy(positions.get(h.source)).lerp(positions.get(h.target),(p.offset_bytes+p.payload_bytes/2)/h.bytes);m.scale.setScalar(tracking?.12:1);m.visible=true;}}}
  requestAnimationFrame(animate);}requestAnimationFrame(animate);$('status').textContent='真实推理运行已载入 '+backend;
}catch(e){$('status').textContent=e.message;}
