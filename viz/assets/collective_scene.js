import '/assets/motion_state.js';
window.runtimeGlobeMode=true;
window.collectivePhysicalMembers=true;
const $=id=>document.getElementById(id);
const query=new URLSearchParams(location.search);
$('scenario').value=query.get('mode')==='multi'?'multi':'ring4';
$('backend').value=query.get('backend')==='ns3'?'ns3':'flowsim';
window.collectiveModelScale=$('scenario').value==='ring4'?.17:1;
window.collectiveVisualScale=$('scenario').value==='ring4'?.1:1;
let primitiveMap=new Map();let data,presented,time=12,playing=false,tracking=false,previousCenter,loaded=false;
const tensorNotes={all_reduce:'梯度组 A：每 rank 8 MB，3 轮 ReduceScatter + 3 轮 AllGather',reduce_scatter:'梯度组 B：每 rank 8 MB，得到 2 MB 归约分片',all_gather:'更新后的参数分片：每 rank 2 MB，收齐为 8 MB',reduce:'独立归约向量：每 rank 4 MB，树形汇总到 sat-0'};
const names={all_reduce:'AllReduce',reduce_scatter:'ReduceScatter',all_gather:'AllGather',reduce:'Reduce'};
window.networkReplayState=()=>presented;
window.inspectNetworkNode=id=>$('details').textContent=JSON.stringify(presented.snapshot.nodes.find(n=>n.node_id===id),null,2);
window.inspectNetworkLink=id=>$('details').textContent=JSON.stringify(presented.snapshot.links.find(n=>n.link_id===id),null,2);
const format=(n,d=3)=>Number(n).toFixed(d),safe=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function leg(h){const p=primitiveMap.get(h.flow_id);return {...h,shard_index:p?.shard_index,source_node_id:h.source,target_node_id:h.target,payload_bytes:h.bytes,
 segments:h.segments.map(s=>({start_s:s.interval_start_s,finish_s:s.interval_end_s,
  bandwidth_bps:s.allocated_bandwidth_bps,bytes:s.bytes_sent})),
 status:time<h.start_s?'future':time<h.finish_s?'transmitting':time<h.arrival_s?'propagating':'delivered'};}
function update(){
 if(!data)return;
 let index=0;for(let i=1;i<data.frames.length&&data.frames[i].time_s<=time;i++)index=i;
 const first=data.frames[index],last=data.frames[Math.min(index+1,data.frames.length-1)],u=last.time_s>first.time_s?(time-first.time_s)/(last.time_s-first.time_s):0;
 const other=new Map(last.nodes.map(n=>[n.node_id,n]));
 const snapshot={...first,physical_time_s:first.time_s,time_s:time,nodes:first.nodes.map(n=>({...n,
  position_itrs_km:n.position_itrs_km.map((v,i)=>v+(other.get(n.node_id).position_itrs_km[i]-v)*u)}))};
 const path=data.hops.map(leg),active=path.filter(h=>h.start_s<=time&&time<h.arrival_s);
 const record={request:{request_id:'training-step-0'},routing:{selected_path:path}};
 presented={orbitSnapshot:first,loadOrbitSample:()=>data.frames[index+1] || data.frames[index-1],data:{network:data.network,time:data.time,sun_positions:data.sun_positions},snapshot,kind:'all',record,
  runtime:{time_s:time,event:{time_s:time,kind:'collective_replay',event_order:time},globalMode:true,
    legs:active,activeTransfers:active,compute:'waiting',routePlans:[],recoveryEvents:[],lossEvents:[]}};
 $('clock').textContent=format(time)+' s 物理快照 '+first.time_s+' s';$('time').value=time;
 const starts=data.collectives.filter(e=>e.kind==='collective_start'),done=new Map(data.collectives.filter(e=>e.kind==='collective_complete').map(e=>[e.id,e.time_s]));
 $('phases').innerHTML=starts.map(e=>{const end=done.get(e.id),state=time<e.time_s?'waiting':time<end?'active':'done';return `<div class="phase ${state}"><b>${names[e.operation]}</b> ${format(e.time_s)}–${format(end)} s<br><small>${tensorNotes[e.operation]}</small><br><small>${state==='active'?'执行中':state==='done'?'完成':'等待前序任务'}</small></div>`}).join('');
 const prim=new Map(data.collectives.filter(e=>e.kind==='primitive_start').map(e=>[e.id,e]));
 $('transfers').innerHTML='<table><tr><th>发送 → 接收</th><th>阶段 / 轮次 / 分片</th><th>数据 / 进度 / 速率</th></tr>'+active.map(h=>{const p=prim.get(h.flow_id),fraction=MotionState.progress(h,time),segment=h.segments.find(s=>s.start_s<=time&&time<s.finish_s);return `<tr><td>${safe(h.source)} → ${safe(h.target)}</td><td>${p?safe(p.phase)+' / '+p.round+(p.shard_index!=null?' / #'+p.shard_index:''):safe(h.request_id)}</td><td>${format(h.bytes/1e6,2)} MB / ${format((fraction??0)*100,1)}%${segment?' / '+format(segment.bandwidth_bps/1e6,2)+' Mbps':''}</td></tr>`}).join('')+'</table>';
 $('arrivals').innerHTML=data.events.map(e=>`<p>${safe(e.request_id)} ${e.time_s} s ${time>=e.time_s?'已到达':'尚未到达'}</p>`).join('');
 if(tracking&&window.orbitGlobe){const g=window.orbitGlobe,c=memberCenter();if(previousCenter){const delta=c.clone().sub(previousCenter);g.camera.position.add(delta);g.controls.target.add(delta);}previousCenter=c;}
}
function memberCenter(){const g=window.orbitGlobe,points=presented.snapshot.nodes.filter(n=>n.role!=='ground').map(n=>g.vector(n.position_itrs_km));return points.reduce((a,b)=>a.add(b),new g.THREE.Vector3()).multiplyScalar(1/points.length);}
function focus(){const g=window.orbitGlobe;if(!g)return;g.overlay?.stopFocus();const center=memberCenter(),points=presented.snapshot.nodes.filter(n=>n.role!=='ground').map(n=>g.vector(n.position_itrs_km));
 const span=Math.max(...points.map(p=>p.distanceTo(center))),out=center.clone().normalize(),up=new g.THREE.Vector3(0,1,0),side=new g.THREE.Vector3().crossVectors(out,up).normalize();
 g.controls.target.copy(center);g.camera.position.copy(center).add(out.multiplyScalar(Math.max(.3,span*3.5))).add(side.multiplyScalar(Math.max(.15,span*1.3)));g.controls.minDistance=.025;g.controls.maxDistance=60;g.controls.update();tracking=true;previousCenter=center;
}
$('focus').onclick=focus;
$('overview').onclick=()=>{const g=window.orbitGlobe;if(!g)return;tracking=false;g.overlay?.stopFocus();g.controls.target.set(0,0,0);g.camera.position.copy(memberCenter().normalize().multiplyScalar(20));g.controls.minDistance=6.6;};
$('time').oninput=e=>{time=Number(e.target.value);update()};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'连续播放'};
$('load').onclick=()=>location.href='/collectives?mode='+$('scenario').value+'&backend='+$('backend').value;
$('run').onclick=async()=>{ $('status').textContent='正在运行真实集合通信任务…';$('run').disabled=true;
 try{const r=await fetch('/collectives/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:$('scenario').value,backend:$('backend').value})});if(!r.ok)throw Error((await r.json()).error);const poll=setInterval(async()=>{const q=await fetch('/collectives/status?mode='+$('scenario').value+'&backend='+$('backend').value);const s=await q.json();if(s.state==='completed'){clearInterval(poll);location.reload()}else if(s.state==='failed'){clearInterval(poll);$('status').textContent=s.error;$('run').disabled=false}},1500)}catch(e){$('status').textContent=e.message;$('run').disabled=false}};
try{
 const response=await fetch('/collectives/data/'+$('scenario').value+'-'+$('backend').value+'.json');data=await response.json();if(!response.ok)throw Error(data.error);primitiveMap=new Map(data.collectives.filter(e=>e.kind==='primitive_start').map(e=>[e.id,e]));
 $('description').textContent=($('scenario').value==='ring4'?'一个 cell 包含 sat-0、sat-1、sat-2、sat-3，物理链路为 0–1–2–3–0 的 ring。':'四个单星 cell 沿晨昏轨道独立运动，四个 rank 跨 cell 共同执行同一次训练通信任务。')+' LTAN 06:00，倾角 '+format(data.orbit.inclination_deg,2)+'°；10 s 物理采样，30 s 链路容量减半。';
 $('checks').textContent='原语 '+data.checks.primitive_count+' / 完成 '+data.checks.completed_count+'；依赖前序完成后才启动；业务完成 '+data.summary.completed_request_count+'/'+data.summary.total_request_count;
 const starts=data.collectives.filter(e=>e.kind==='collective_start');for(const e of starts){const b=document.createElement('button');b.textContent=names[e.operation];b.onclick=()=>{time=e.time_s+.1;update()};$('jumps').append(b)}
 for(const t of [12,18,30]){const b=document.createElement('button');b.textContent=t+' s '+(t===30?'容量变化':'新请求');b.onclick=()=>{time=t;update()};$('jumps').append(b)}
 update();await import('/assets/globe.js');document.getElementById('showLinks').checked=true;
 const g=window.orbitGlobe,labels=[];for(const id of ['sat-0','sat-1','sat-2','sat-3']){const label=document.createElement('div');label.style.cssText='position:absolute;pointer-events:none;background:#071522dd;border:1px solid #8dbbed;border-radius:5px;padding:5px 8px;color:#e2ecfa;font-size:12px';label.textContent=id+' '+($('scenario').value==='ring4'?'rank '+id.slice(-1):'cell-'+id.slice(-1));g.container.append(label);labels.push({id,label})}
 focus();loaded=true;$('status').textContent='已载入真实 '+data.backend+' 执行结果';let last=performance.now();
 function animate(now){const dt=Math.min(.1,(now-last)/1000);last=now;if(playing){time=Math.min(240,time+dt*Number($('speed').value));update();if(time===240)playing=false;}
  for(const {id,label} of labels){const node=presented.snapshot.nodes.find(n=>n.node_id===id),v=g.vector(node.position_itrs_km).project(g.camera);label.style.left=((v.x+1)*g.container.clientWidth/2+10)+'px';label.style.top=((1-v.y)*g.container.clientHeight/2+12)+'px';label.hidden=Math.abs(v.x)>1||Math.abs(v.y)>1||v.z>1;}requestAnimationFrame(animate)}requestAnimationFrame(animate);
}catch(e){$('status').textContent=e.message;}
