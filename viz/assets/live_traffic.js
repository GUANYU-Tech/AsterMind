import {aggregateTraffic,channelKey,matchesTraffic,createTrafficClock,MAX_TRAFFIC_CHANNELS,TRAFFIC_PALETTE,endToEndFlowId,pathAt,pathComplete,trafficMarkerFraction} from './live_traffic_model.js';
import {attachLivePanelFeedback} from './live_panel_feedback.js';

// The live view owns its camera only during explicit focus/overview actions.
// Cubes illustrate flow direction, not packets emitted by the fluid solver.
export function attachOverlay({THREE,scene,camera,controls,container,vector}){
 const group=new THREE.Group();group.name='live-traffic';scene.add(group);
 const palette=TRAFFIC_PALETTE,panelFeedback=attachLivePanelFeedback(container);
 const capacity=MAX_TRAFFIC_CHANNELS+1;
 function pool(name,geometry){
  const mesh=new THREE.InstancedMesh(geometry,new THREE.MeshBasicMaterial({color:0xffffff}),capacity);
  mesh.name=name;mesh.count=0;mesh.frustumCulled=false;mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);group.add(mesh);return mesh;
 }
 const lines=pool('live-traffic-channels',new THREE.CylinderGeometry(1,1,1,6));
 const routeLines=pool('live-request-route',new THREE.CylinderGeometry(1,1,1,6));
 const cubeGeometry=new THREE.BoxGeometry(1,1,1),shades=[];
 // Face shading makes the small square read as a cube even on the Earth's night side.
 for(let face=0;face<6;face++)for(let vertex=0;vertex<4;vertex++){const shade=[1,.64,.93,.58,.83,.7][face];shades.push(shade,shade,shade)}
 cubeGeometry.setAttribute('color',new THREE.Float32BufferAttribute(shades,3));
 const cubes=pool('live-traffic-cubes',cubeGeometry);cubes.material.vertexColors=true;
 const dummy=new THREE.Object3D(),up=new THREE.Vector3(0,1,0),color=new THREE.Color(),clock=createTrafficClock();
 const reducedMotion=window.matchMedia?.('(prefers-reduced-motion: reduce)');
 const bar=document.createElement('div');bar.className='live-traffic-controls';container.before(bar);
 function field(name,element){const label=document.createElement('label');label.append(document.createTextNode(name),element);bar.append(label);return element}
 const scope=field('显示范围',document.createElement('select'));scope.setAttribute('aria-label','通信显示范围');
 for(const [value,text] of [['all','全部通信'],['node','所选节点'],['link','所选链路']]){const option=document.createElement('option');option.value=value;option.textContent=text;scope.append(option)}
 const query=field('筛选传输',document.createElement('input'));query.type='search';query.placeholder='请求或传输 ID';query.setAttribute('aria-label','筛选请求或传输 ID');
 const trace=field('跟踪传输',document.createElement('select'));trace.setAttribute('aria-label','跟踪端到端传输');
 const focus=document.createElement('button');focus.textContent='定位当前事件';focus.title='只定位一次，后续事件保持当前视角';bar.append(focus);
 const overview=document.createElement('button');overview.textContent='整体星座';bar.append(overview);
 const replayPath=document.createElement('button');replayPath.textContent='回放这条传输';bar.append(replayPath);
 const caption=document.createElement('div');caption.className='live-route-caption';bar.append(caption);
 const hud=document.createElement('div');hud.className='live-traffic-hud';container.append(hud);
 let selection={},filterRevision=0,signature='',positions=new Map(),draws=[],routeDraws=[],focusPoints=[],traceOptions='',pinnedPath=null,currentPath=null;
 const filter=()=>({scope:scope.value,selection,query:query.value,flowId:trace.value});
 scope.onchange=()=>{filterRevision++};
 query.oninput=()=>{trace.value='';pinnedPath=null;filterRevision++;window.queryRuntimePaths?.(query.value)};
 trace.onchange=()=>{pinnedPath=null;filterRevision++};
 replayPath.onclick=()=>{if(currentPath){trace.value=currentPath.id;pinnedPath=currentPath;filterRevision++;window.inspectRuntimePath?.(currentPath)}};
 function position(id){
  if(!id)return null;
  if(positions.has(id))return positions.get(id);
  // GPU endpoints belong to physical satellites; keep the global globe at Cell scale.
  const satellite=id.includes('/gpu')?id.slice(0,id.indexOf('/gpu')):null;
  return satellite?positions.get(satellite):null;
 }
 function manualFocus(points){
  if(!points.length)return;
  const center=points.reduce((sum,p)=>sum.add(p),new THREE.Vector3()).multiplyScalar(1/points.length);
  const outward=center.clone().normalize(),span=points.reduce((max,p)=>Math.max(max,p.distanceTo(center)*2),0);
  if(outward.length()<.01)outward.copy(points[0]).normalize();
  controls.target.copy(center);camera.position.copy(center).addScaledVector(outward,Math.max(3,span*1.8));controls.minDistance=.15;
  const rotate=document.getElementById('autoRotate');if(rotate)rotate.checked=false;
  controls.autoRotate=false;controls.update();
 }
 focus.onclick=()=>manualFocus(focusPoints);
 window.focusRuntimeTransmission=focus.onclick;
 overview.onclick=()=>{controls.target.set(0,0,0);camera.position.set(13,8,15);controls.minDistance=6.6;controls.update()};
 function rebuild(state){
  const runtime=state.runtime,event=runtime.event||{},options=filter();
  const order=runtime.observed_order??event.event_order??Infinity,complete=p=>pathComplete(p,order,runtime.time_s,runtime.time_group);
  const paths=(state.pathContext?.paths||[]).filter(p=>p.time_semantics==='whole_path_shared_resources'?p.segments?.length:p.hops?.length&&['request_input','request_output','kv_migration','pd_kv'].includes(p.stage));
  const byFlow=new Map(paths.map(p=>[p.id,p]));
  if(pinnedPath&&!byFlow.has(pinnedPath.id)){paths.push(pinnedPath);byFlow.set(pinnedPath.id,pinnedPath)}
  const search=query.value.trim().toLowerCase(),pending=paths.filter(p=>p.start_order<=order&&!complete(p)),recent=paths.filter(complete).slice(-10);
  const menuPaths=search?paths.filter(p=>[p.id,p.request_id].some(v=>String(v||'').toLowerCase().includes(search))):[...pending,...recent];
  if(pinnedPath){const index=menuPaths.findIndex(p=>p.id===pinnedPath.id);if(index>=0)menuPaths.splice(index,1);menuPaths.unshift(pinnedPath)}
  const choiceSignature=menuPaths.map(p=>p.id+'|'+p.request_id).join(';');
  if(choiceSignature!==traceOptions||!trace.children.length){
   traceOptions=choiceSignature;const previous=trace.value,items=[];
   const automatic=document.createElement('option');automatic.value='';automatic.textContent='跟随当前事件';items.push(automatic);
   for(const p of menuPaths.slice(0,100)){const option=document.createElement('option');option.value=p.id;const who=p.request_id?.match(/^user-(\d+)-request-/)?.[1];option.textContent=`${who?'用户 '+who+' / ':''}${p.time_semantics==='whole_path_shared_resources'?'V1 路径':p.stage==='request_input'?'请求输入':p.stage==='request_output'?'结果返回':'跨 Cell 通信'} / #${p.id.replace('serving-','')}`;items.push(option)}
   trace.replaceChildren(...items);trace.value=byFlow.has(previous)?previous:'';
  }
  const enrich=leg=>({...leg,business_request_id:leg.business_request_id||byFlow.get(endToEndFlowId(leg))?.request_id});
  const activeTransfers=(runtime.activeTransfers||[]).map(enrich);
  positions=new Map(state.snapshot.nodes.map(n=>[n.node_id,window.computeCellCenter?.(n.node_id)||vector(n.position_itrs_km)]));
  const eventLeg=(runtime.legs||[]).find(l=>l.transfer_id===event.transfer_id)||event;
  const eventHasPath=!!(eventLeg.source_node_id&&eventLeg.target_node_id);
  const active=enrich(eventHasPath||event.compute_node_id||event.cell_id?eventLeg:activeTransfers.find(leg=>matchesTraffic(leg,options))||eventLeg);
  const result=aggregateTraffic(activeTransfers,options,MAX_TRAFFIC_CHANNELS,channelKey(active));
  const a=position(active.source_node_id),b=position(active.target_node_id);
  const computeNode=event.compute_node_id||state.snapshot.nodes.find(n=>event.cell_id&&n.cell_id===event.cell_id)?.node_id;
  const computePoint=position(computeNode);
  focusPoints=a&&b?[a,b]:computePoint?[computePoint]:[];focus.disabled=!focusPoints.length;
  const businessId=event.business_request_id||event.correlation_id||event.request_id;
  const related=paths.find(p=>p.request_id===businessId&&p.stage==='request_input');
  const tracked=byFlow.get(trace.value)||(search?menuPaths[0]:byFlow.get(endToEndFlowId(active))||related||pending.find(p=>p.stage==='request_input')||recent.at(-1));
  currentPath=tracked;replayPath.disabled=!tracked;
  if(trace.value&&tracked)pinnedPath=tracked;
  const observed=pathAt(tracked,order).filter(h=>h.source_node_id!==h.target_node_id);
  routeDraws=[];
  for(const hop of observed){
   const start=position(hop.source_node_id),end=position(hop.target_node_id);if(!start||!end)continue;
   const delta=end.clone().sub(start),length=delta.length();if(length<.000001)continue;
   routeDraws.push({a:start,b:end,length,status:hop.status,orientation:new THREE.Quaternion().setFromUnitVectors(up,delta.normalize())});
  }
  if(routeDraws.length){focusPoints=routeDraws.flatMap(d=>[d.a,d.b]);focus.disabled=false;focus.title='定位这条传输已走过的路径；后续事件保持当前视角'}
  const delivered=observed.filter(h=>h.status==='delivered').length,done=complete(tracked);
  const sequence=tracked?.time_semantics==='whole_path_shared_resources'?'V1 整条路径共享资源':tracked?.stage==='request_input'?'地面站 → 通信星座 → 计算 Cell':tracked?.stage==='request_output'?'计算 Cell → 通信星座 → 地面站':'跨 Cell 通信';
  const user=tracked?.request_id?.match(/^user-(\d+)-request-/)?.[1];
  caption.textContent=tracked?`${user?'用户 '+user:'传输 #'+tracked.id.replace('serving-','')} / ${sequence} / ${done?(tracked.terminal==='failed'?'传输失败':'传输已完成'):tracked.time_semantics==='whole_path_shared_resources'?'路径传输中':`已走 ${delivered} 跳`}`:search?'未找到匹配的已执行传输':'选择传输后显示已走路径与当前一跳';
  caption.title=(tracked?.time_semantics==='whole_path_shared_resources'?'整条路径同时占用共享资源；方块仅示意方向。':'绿色保留已走路径；方块标出当前一跳。')+(tracked?.request_id||'');
  if(routeDraws.length>capacity)caption.textContent+=` / 路径显示前 ${capacity} 跳`;
  routeDraws=routeDraws.slice(0,capacity);
  draws=[];let unmapped=0,internal=0;
  function add(channel,status){
   const start=position(channel.source_node_id),end=position(channel.target_node_id);
   if(!start||!end){if(['queued','transmitting','propagating'].includes(status))unmapped+=channel.count;return}
   const delta=end.clone().sub(start),length=delta.length(),local=length<.000001;
   if(local&&['queued','transmitting','propagating'].includes(status))internal+=channel.count;
   const orientation=new THREE.Quaternion().setFromUnitVectors(up,local?up:delta.clone().normalize());
   let seed=0;for(const char of channel.key)seed=(seed*31+char.charCodeAt(0))>>>0;
   draws.push({...channel,a:start,b:end,length,local,orientation,status,color:palette[status],phase:(seed%997)/997});
  }
  for(const channel of result.channels)add(channel,channel.transmitting?'transmitting':channel.propagating?'propagating':'queued');
  // A completed short hop leaves a green trace for the current viewing frame.
  // Its final state never contributes to the active transmission count.
  const recentChannels=new Map();
  for(const event of state.recentTransfers||[]){
   if(!matchesTraffic(enrich(event),options))continue;
   const key=channelKey(event);if(result.channels.some(channel=>channel.key===key))continue;
   const prior=recentChannels.get(key);recentChannels.set(key,{...event,key,count:(prior?.count||0)+1});
  }
  for(const channel of recentChannels.values())add(channel,channel.kind==='transfer_cancelled'?'cancelled':'delivered');
  const finalStatus=({hop_arrival:'delivered',transfer_cancelled:'cancelled'})[event.kind];
  if(finalStatus&&a&&b&&matchesTraffic(active,options)&&!draws.some(d=>d.key===channelKey(active)))add({...active,key:channelKey(active),count:1},finalStatus);
  if(computePoint&&(!trace.value||tracked?.request_id===(event.business_request_id||event.request_id))&&options.scope!=='link'&&(!query.value||matchesTraffic(event,options))&&(!selection.nodeId||options.scope!=='node'||selection.nodeId===computeNode)){
   draws.push({a:computePoint,b:computePoint,length:0,local:true,count:1,status:'compute',color:palette.compute,phase:0,orientation:new THREE.Quaternion()});
  }
  // Pool capacity also covers the current event marker when no flow is active.
  draws=draws.slice(0,capacity);
  const wholePath=activeTransfers.some(e=>e.time_semantics==='whole_path_shared_resources')||event.time_semantics==='whole_path_shared_resources';
  const parts=[`${result.matched.toLocaleString()} 条${wholePath?'活跃路径段':'并发传输'}`,`${result.linkCount.toLocaleString()} 条活跃链路`];
  if(result.matched!==result.total)parts.push(`全场 ${result.total.toLocaleString()} 条`);
  if(result.hiddenChannels)parts.push(`连线显示上限 ${MAX_TRAFFIC_CHANNELS}，可筛选查看`);
  if(internal)parts.push(`Cell 内部 ${internal} 条`);
  if(unmapped)parts.push(`${unmapped} 条无位置数据`);
  hud.replaceChildren(...parts.map(text=>{const item=document.createElement('span');item.textContent=text;return item}));
  hud.title=`发送 ${result.counts.transmitting}，传播 ${result.counts.propagating}，排队 ${result.counts.queued}。`;
  const selectedHop=observed.find(h=>['queued','transmitting','propagating'].includes(h.status))||observed.at(-1);
  const info=document.getElementById('hopInfo');if(info)info.textContent=trace.value&&selectedHop?`所选传输：${selectedHop.source_node_id} → ${selectedHop.target_node_id}`:a&&b?`${eventHasPath?'当前事件':'活跃传输'}：${active.source_node_id} → ${active.target_node_id}`:computeNode?`当前计算：${computeNode}`:'当前事件没有物理链路传输';
 }
 function update(state,now=performance.now()){
  if(!state.runtime)return;
  panelFeedback.update(state.playback?.waiting,now);
  const event=state.runtime.event||{},key=[state.revision,state.snapshot.time_s,event.event_order,event.kind,filterRevision].join('|');
  if(key!==signature){signature=key;rebuild(state)}
  const phase=clock.sample(now,{...state.playback,playing:!!state.playback?.playing&&!reducedMotion?.matches});
  const height=Math.max(1,container.clientHeight),pixelScale=2*Math.tan(camera.fov*Math.PI/360)/height;
  let lineCount=0,cubeCount=0,routeCount=0;
  for(const draw of routeDraws){
   const mid=draw.a.clone().lerp(draw.b,.5),unit=camera.position.distanceTo(mid)*pixelScale;
   dummy.position.copy(mid);dummy.quaternion.copy(draw.orientation);dummy.scale.set(unit*1.6,draw.length,unit*1.6);dummy.updateMatrix();
   routeLines.setMatrixAt(routeCount,dummy.matrix);routeLines.setColorAt(routeCount++,color.setHex(draw.status==='delivered'?0xc3ed8c:palette[draw.status]||palette.delivered));
  }
  for(const draw of draws){
   const mid=draw.a.clone().lerp(draw.b,.5),unit=camera.position.distanceTo(mid)*pixelScale;
   const width=unit*Math.min(1.35,.55+Math.log2(draw.count+1)*.15);
   if(!draw.local){
    dummy.position.copy(mid);dummy.quaternion.copy(draw.orientation);dummy.scale.set(width,draw.length,width);dummy.updateMatrix();
    lines.setMatrixAt(lineCount,dummy.matrix);lines.setColorAt(lineCount++,color.setHex(draw.color).multiplyScalar(.65));
   }
   const fraction=trafficMarkerFraction(draw.status,phase,draw.phase,!!state.playback?.playing&&!reducedMotion?.matches);
   dummy.position.copy(draw.a).lerp(draw.b,fraction);
   // A local marker sits beside a collapsed Cell icon, without inventing a physical hop.
   if(draw.local)dummy.position.addScaledVector(mid.clone().normalize(),unit*16);
   dummy.quaternion.copy(draw.orientation);
   dummy.scale.setScalar(draw.local?unit*8:Math.min(unit*8,draw.length*.15));dummy.updateMatrix();
   cubes.setMatrixAt(cubeCount,dummy.matrix);cubes.setColorAt(cubeCount++,color.setHex(draw.color));
  }
  lines.count=lineCount;cubes.count=cubeCount;routeLines.count=routeCount;
  for(const mesh of [lines,cubes,routeLines]){mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true}
 }
 return {update,setLinksVisible(visible){lines.visible=visible;routeLines.visible=visible},select(value){selection=value||{};filterRevision++},stopFocus(){},
  dispose(){panelFeedback.dispose();bar.remove();hud.remove();scene.remove(group);for(const mesh of [lines,cubes,routeLines]){mesh.geometry.dispose();mesh.material.dispose();mesh.dispose()}}};
}
