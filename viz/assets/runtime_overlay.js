
import '/assets/motion_state.js';
export function attachOverlay({THREE,scene,camera,controls,container,toolbar,vector}){
 const group=new THREE.Group();scene.add(group);
 scene.add(new THREE.AmbientLight(0x9ebde8,.65));
 const palette={queued:0x9eb4cb,transmitting:0xffc665,propagating:0xbb9dff,delivered:0x83dab6,cancelled:0xff8585,dropped:0xff5069,future:0x435873};
 const labels=[],pulses=[],up=new THREE.Vector3(0,1,0);
 const hud=document.createElement('div');hud.style.cssText='position:absolute;left:12px;top:12px;max-width:88%;background:#071522dc;border:1px solid #375570;border-radius:8px;padding:9px 12px;font:12px ui-monospace;color:#d4e5ff;pointer-events:none;white-space:pre-line';container.append(hud);
 const autoLabel=document.createElement('label'),auto=document.createElement('input');auto.type='checkbox';auto.checked=Boolean(window.runtimeLiveMode);autoLabel.append(auto,document.createTextNode('聚焦当前跳（远景）'));toolbar.append(autoLabel);
 const overview=document.createElement('button');overview.textContent='整体星座与并发流量';toolbar.append(overview);
 overview.onclick=()=>{auto.checked=false;focusKey='';signature='';controls.target.set(0,0,0);camera.position.copy(overviewDirection).multiplyScalar(22);controls.minDistance=6.6};
 const historyLabel=document.createElement('label'),history=document.createElement('input');history.type='checkbox';history.checked=!window.runtimeLiveMode;historyLabel.append(history,document.createTextNode('显示已执行路径'));toolbar.append(historyLabel);
 const planLabel=document.createElement('label'),planned=document.createElement('input');planned.type='checkbox';planned.checked=!window.runtimeLiveMode;planLabel.append(planned,document.createTextNode('显示原定路线与改道方案'));toolbar.append(planLabel);
 let signature='',focusKey='',overviewDirection=new THREE.Vector3(12,8,14).normalize();
 window.focusRuntimeTransmission=()=>{auto.checked=true;focusKey='';signature=''};
 history.onchange=()=>signature='';
 planned.onchange=()=>signature='';
 auto.onchange=()=>{focusKey='';signature='';if(!auto.checked){controls.target.set(0,0,0);camera.position.copy(overviewDirection).multiplyScalar(22);controls.minDistance=6.6}};
 function clear(){pulses.splice(0);while(group.children.length){const object=group.children[0];group.remove(object);object.geometry?.dispose();if(Array.isArray(object.material))object.material.forEach(m=>m.dispose());else object.material?.dispose()}labels.splice(0).forEach(l=>l.element.remove())}
 function line(a,b,color,opacity,radius,arrowed=true){
  const delta=b.clone().sub(a);
  if(delta.length()===0)return;
  radius=Math.min(radius*(window.collectiveVisualScale||1),delta.length()*.035);
  const mesh=new THREE.Mesh(new THREE.CylinderGeometry(radius,radius,delta.length(),8),new THREE.MeshBasicMaterial({color,transparent:true,opacity}));
  mesh.position.copy(a).add(b).multiplyScalar(.5);mesh.quaternion.setFromUnitVectors(up,delta.clone().normalize());group.add(mesh);
  if(arrowed){const arrow=new THREE.Mesh(new THREE.ConeGeometry(Math.min(radius*3.5,delta.length()*.07),Math.min(radius*10,delta.length()*.2),10),new THREE.MeshBasicMaterial({color,transparent:true,opacity}));
  arrow.position.copy(a).lerp(b,.62);arrow.quaternion.copy(mesh.quaternion);group.add(arrow)}
 }
 function mark(position,color,size){const mesh=new THREE.Mesh(new THREE.SphereGeometry(size*(window.collectiveVisualScale||1),16,10),new THREE.MeshBasicMaterial({color}));mesh.position.copy(position);group.add(mesh)}
 function dashed(a,b,color,radius){if(window.servingFlowView)radius*=window.collectiveVisualScale||1;const delta=b.clone().sub(a);for(let i=0;i<12;i+=2){const mesh=new THREE.Mesh(new THREE.CylinderGeometry(radius,radius,delta.length()/12,6),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.8}));mesh.position.copy(a).lerp(b,(i+.5)/12);mesh.quaternion.setFromUnitVectors(up,delta.clone().normalize());group.add(mesh)}}
 function label(position,text,color){
  const element=document.createElement('div');element.textContent=text;element.style.cssText='position:absolute;pointer-events:none;background:#071522e8;border:1px solid '+color+';color:'+color+';padding:5px 8px;border-radius:6px;font:12px system-ui;white-space:nowrap;max-width:340px;overflow:hidden;text-overflow:ellipsis';
  container.append(element);labels.push({position,element});
 }
 function update(state){
  if(!state.runtime)return;
  const runtime=state.runtime,event=runtime.event,key=state.record.request.request_id+'|'+event.event_order+'|'+event.kind+'|'+runtime.time_s+'|'+state.snapshot.time_s;
  const computeNode=event.compute_node_id||state.snapshot.nodes.find(n=>event.cell_id&&n.cell_id===event.cell_id)?.node_id;
  if(key!==signature){
   signature=key;clear();
   const positions=new Map(state.snapshot.nodes.map(n=>[n.node_id,window.computeCellCenter?.(n.node_id)||vector(n.position_itrs_km)]));
   const active=runtime.legs.find(l=>RuntimePlayback.key(l)===RuntimePlayback.key(event))||runtime.legs.filter(l=>['queued','transmitting','propagating'].includes(l.status)).slice(-1)[0];
   const concurrent=runtime.globalMode?(runtime.activeTransfers||[]):runtime.legs.filter(l=>['queued','transmitting','propagating'].includes(l.status));
   const plans=runtime.routePlans||[],original=plans.find(p=>p.reason==='initial'),lastRecovery=(runtime.recoveryEvents||[]).at(-1);
   const replacement=lastRecovery?plans.find(p=>p.reason==='recovery'&&p.time_s>=lastRecovery.time_s):null;
   if(original?.path?.length){const a=positions.get(original.path[0].source_node_id),b=positions.get(original.path.at(-1).target_node_id);if(a&&b){if(!focusKey){overviewDirection.copy(a).normalize().add(b.clone().normalize()).normalize();camera.position.copy(overviewDirection).multiplyScalar(22);controls.target.set(0,0,0);focusKey='planned-overview'}if(planned.checked){label(a,'用户 / 地面接入','#9dbbff');label(b,'目标计算 cell','#c9a4ff')}}}
   if(planned.checked){for(const leg of original?.path||[]){const a=positions.get(leg.source_node_id),b=positions.get(leg.target_node_id);if(a&&b)dashed(a,b,0x528dff,.022)}for(const leg of replacement?.path||[]){const a=positions.get(leg.source_node_id),b=positions.get(leg.target_node_id);if(a&&b)line(a,b,0x35e4ef,.85,.026)}}
   const selectedId=state.record.request.request_id;
   const perRequest=new Map();for(const leg of concurrent)perRequest.set(leg.request_id||selectedId,(perRequest.get(leg.request_id||selectedId)||0)+1);
   hud.textContent='事件 '+runtime.time_s.toFixed(9)+' s · 物理快照 '+(state.snapshot.physical_time_s??state.snapshot.time_s)+' s\n'+(runtime.globalMode?'全系统':'本请求')+'并发 '+concurrent.length+' 跳 · '+Array.from(perRequest,([id,count])=>id+': '+count+' 跳').slice(0,8).join(' / ')+(perRequest.size>8?' / …':'');
   const historyLegs=history.checked?runtime.legs.filter(l=>['delivered','cancelled','dropped'].includes(l.status)):[];
   for(const leg of historyLegs) {
    const a=positions.get(leg.source_node_id),b=positions.get(leg.target_node_id);if(a&&b)line(a,b,palette[leg.status],.85,.026);
   }
   for(const leg of concurrent){
    if(leg.status==='future')continue;
    const a=positions.get(leg.source_node_id),b=positions.get(leg.target_node_id);if(!a||!b)continue;
    const selected=leg.request_id===selectedId||!runtime.globalMode;let flowColor=window.collectivePhysicalMembers&&Number.isInteger(leg.shard_index)?[0x62d5ff,0xffbc65,0xd49cff,0x79e2b5][leg.shard_index%4]:palette[leg.status];if(window.servingFlowView)flowColor=leg.stage==='input'?0x62d5ff:['output','first_token'].includes(leg.stage)?0x79e2b5:(leg.flow_id||'').includes('pp-')?0xc49cff:(leg.flow_id||'').includes('ep-output')?0xf18bc7:0xffbc65;const rate=leg.segments?.find(s=>s.start_s<=runtime.time_s&&runtime.time_s<s.finish_s)?.bandwidth_bps||0;const width=window.servingFlowView?(window.servingFlowView==='ns3'?.14:Math.max(.18,Math.min(1.8,Math.sqrt(rate/32e6)))):1;line(a,b,flowColor,selected?1:.75,(auto.checked?.012:.03)*width);
    if(['transmitting','propagating'].includes(leg.status)&&!window.servingFlowView){const mesh=new THREE.Mesh(new THREE.SphereGeometry(.055*(window.collectiveVisualScale||1),12,8),new THREE.MeshBasicMaterial({color:leg.status==='propagating'?0xbb9dff:flowColor}));group.add(mesh);const actual=state.record?.routing?.selected_path?.find(l=>RuntimePlayback.key(l)===RuntimePlayback.key(leg));pulses.push({mesh,a,b,leg:{...actual,...leg,segments:actual?.segments??leg.segments,start_s:actual?.start_s??leg.start_s,finish_s:actual?.finish_s??leg.finish_s,arrival_s:actual?.arrival_s??leg.arrival_s}})}
   }
   const relevant=new Set([...(original?.path||[]),...(replacement?.path||[])].map(l=>l.link_id));
   for(const fault of state.data.network.faults||[]){if(!(fault.start_s<=runtime.time_s&&runtime.time_s<fault.end_s))continue;for(const edge of state.snapshot.links){if(!relevant.has(edge.link_id)||(fault.target_type==='link'?fault.target_id!==edge.link_id:![edge.endpoint_a,edge.endpoint_b].includes(fault.target_id)))continue;const a=positions.get(edge.endpoint_a),b=positions.get(edge.endpoint_b);if(a&&b){line(a,b,0xff354f,1,.065,false)}}}
   const lastLoss=(runtime.lossEvents||[]).at(-1);if(lastLoss&&runtime.time_s>=lastLoss.time_s&&(!lastRecovery||lastRecovery.time_s<lastLoss.time_s)){const node=lastLoss.resume_node_id||lastLoss.source_node_id,p=positions.get(node);if(p){mark(p,0xffdf83,.1);label(p,'数据保留于 '+node+' · 等待 Runtime 执行恢复选路','#ffdf83')}}
   if(lastRecovery&&runtime.compute==='waiting'){const done=runtime.legs.filter(l=>l.stage==='input'&&l.status==='delivered').at(-1),node=done?.target_node_id||lastRecovery.source_node_id,p=positions.get(node);if(p){mark(p,0x35e4ef,.09);label(p,'当前缓存节点 '+node,'#62f4ff')}}
   if(active&&active.status==='delivered'){
    const a=positions.get(active.source_node_id),b=positions.get(active.target_node_id);if(a&&b)line(a,b,palette.delivered,1,auto.checked?.012:.03);
   }
   if(active){
    const a=positions.get(active.source_node_id),b=positions.get(active.target_node_id);
    if(a&&b){
     const color='#'+palette[active.status].toString(16).padStart(6,'0');
     mark(a,palette[active.status],auto.checked?.035:.065);mark(b,palette[active.status],auto.checked?.035:.065);
     if(auto.checked){label(a,'发送：'+active.source_node_id,color);label(b,'接收：'+active.target_node_id,color)}
     const focus=active.link_id+'|'+active.source_node_id+'|'+state.record.request.request_id;
     if(!auto.checked&&!focusKey){
      overviewDirection.copy(a).add(b).normalize();camera.position.copy(overviewDirection).multiplyScalar(22);controls.target.set(0,0,0);controls.minDistance=6.6;focusKey='overview';
     }
     if(auto.checked&&focus!==focusKey){
      focusKey=focus;const center=a.clone().add(b).multiplyScalar(.5),out=center.clone().normalize(),span=a.distanceTo(b),offset=Math.max(9.0,span*2.4),side=new THREE.Vector3().crossVectors(out,up).normalize();
      if(side.length()<.5)side.set(1,0,0);
      controls.target.copy(center);camera.position.copy(center).add(out.multiplyScalar(offset*.85)).add(side.multiplyScalar(offset*.65));controls.minDistance=2.0;
     }
    }
   }else if(computeNode&&positions.has(computeNode)){
    const point=positions.get(computeNode);mark(point,0xd99cff,.065);label(point,(event.kind==='surrogate_complete'?'ProtoBridge 通信完成：':'计算：')+computeNode,'#d99cff');
    if(auto.checked&&focusKey!==computeNode){focusKey=computeNode;controls.target.copy(point);camera.position.copy(point).add(point.clone().normalize().multiplyScalar(9.0));controls.minDistance=.15}
   }
   document.getElementById('hopInfo').textContent=(runtime.globalMode?'全系统并发 '+concurrent.length+' 跳 · 当前请求 '+selectedId+' · ':'当前请求 '+selectedId+' · ')+(active?active.source_node_id+' → '+active.target_node_id+' · '+({queued:'等待链路服务',transmitting:'序列化发送中',propagating:'光速传播中',delivered:'本跳已到达',cancelled:'传输中止',dropped:'传输丢弃'}[active.status]||active.status):'当前事件没有正在执行的物理跳');
  }
  const bounds=container.getBoundingClientRect();
  for(const p of pulses){const fraction=MotionState.progress(p.leg,runtime.time_s);p.mesh.visible=fraction!==null;if(fraction!==null)p.mesh.position.copy(p.a).lerp(p.b,fraction)}
  const placed=[];
  for(const item of labels){
   const projected=item.position.clone().project(camera),toPoint=item.position.clone().sub(camera.position);
   // Hide far-side labels just as the opaque Earth hides physical paths.
   const ray=new THREE.Ray(camera.position,toPoint.clone().normalize()),hit=ray.intersectSphere(new THREE.Sphere(new THREE.Vector3(),6.35),new THREE.Vector3());
   const hidden=projected.z>1||(hit&&hit.distanceTo(camera.position)<toPoint.length()-.03)||Math.abs(projected.x)>1||Math.abs(projected.y)>1;
   item.element.style.display=hidden?'none':'block';if(hidden)continue;
   const width=item.element.offsetWidth,height=item.element.offsetHeight;
   const left=Math.max(8,Math.min(bounds.width-width-8,(projected.x+1)*bounds.width/2+10));
   const base=(1-projected.y)*bounds.height/2+10;
   let top=Math.max(92,Math.min(bounds.height-height-50,base));
   for(const shift of [0,1,-1,2,-2,3,-3,4,-4,5,-5]){
    const candidate=Math.max(92,Math.min(bounds.height-height-50,base+shift*(height+6)));
    if(!placed.some(r=>left<r.right+5&&left+width>r.left-5&&candidate<r.bottom+4&&candidate+height>r.top-4)){top=candidate;break}
   }
   placed.push({left,right:left+width,top,bottom:top+height});
   item.element.style.left=left+'px';item.element.style.top=top+'px';
  }
 }
 return {update,stopFocus(){auto.checked=false;focusKey='fault-inspection';signature=''}};
}
