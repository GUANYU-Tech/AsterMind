import '/assets/fault_state.js';
// Faults remain visible independently of background links and business requests.
export function attachFaults({THREE,scene,vector,camera,controls,toolbar,container,onFocus}){
 const {categories,collect,counts}=FaultVisual,colors={...categories,restored:{label:'故障恢复',color:'#4af0ab'}};
 function glyph(){
  const canvas=document.createElement('canvas');canvas.width=canvas.height=64;
  const ctx=canvas.getContext('2d');ctx.strokeStyle='#fff';ctx.lineWidth=5;
  ctx.beginPath();ctx.arc(32,32,25,0,Math.PI*2);ctx.stroke();
  ctx.beginPath();ctx.moveTo(22,22);ctx.lineTo(42,42);ctx.moveTo(42,22);ctx.lineTo(22,42);ctx.stroke();
  return new THREE.CanvasTexture(canvas);
 }
 const map=glyph(),layers=Object.fromEntries(Object.entries(colors).map(([name,{color}])=>{
  const points=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color,size:24,sizeAttenuation:false,map,transparent:true,alphaTest:.1,depthWrite:false}));
  const tips=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color,size:13,sizeAttenuation:false,map,transparent:true,alphaTest:.1,depthWrite:false}));
  const lines=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:name==='restored'?color:0xff354f,transparent:true,opacity:1,depthWrite:false}));
  points.renderOrder=tips.renderOrder=6;lines.renderOrder=5;
  scene.add(points,tips,lines);return [name,{points,tips,lines}];
 }));
 const host=document.createElement('section');host.setAttribute('aria-label','当前时刻故障');
 host.style.cssText='padding:12px 14px;margin:12px 0;border:1px solid #375570;border-radius:8px;background:#101e31;color:#d4e5ff;font:12px/1.7 system-ui';toolbar.after(host);
 const summary=document.createElement('div');summary.setAttribute('role','status');summary.id='faultVisualSummary';host.append(summary);
 const legend=document.createElement('div');legend.style.cssText='display:flex;gap:14px;flex-wrap:wrap;margin:7px 0';host.append(legend);
 const badges=Object.fromEntries(Object.entries(categories).map(([key,category])=>{const span=document.createElement('span');span.style.color=category.color;legend.append(span);return [key,span]}));
 const hud=document.createElement('div');hud.setAttribute('aria-label','三维故障图例');hud.style.cssText='position:absolute;left:12px;bottom:12px;max-width:calc(100% - 24px);background:#071522e8;border:1px solid #375570;border-radius:8px;padding:8px 12px;pointer-events:none;color:#d4e5ff;font:12px/1.8 system-ui';container?.append(hud);
 const actions=document.createElement('div');actions.style.cssText='display:flex;gap:8px;flex-wrap:wrap;align-items:center';host.append(actions);
 const group=document.createElement('select');group.setAttribute('aria-label','故障来源');
 for(const [value,text]of [['all','全部来源'],...Object.entries(categories).map(([k,v])=>[k,v.label])]){const option=document.createElement('option');option.value=value;option.textContent=text;group.append(option)}
 const select=document.createElement('select');select.setAttribute('aria-label','当前故障对象');select.style.cssText='flex:1;min-width:180px;max-width:100%';
 const focus=document.createElement('button');focus.textContent='聚焦故障';actions.append(group,select,focus);
 const hint=document.createElement('div');hint.style.cssText='font-size:11px;color:#9fb5ce;margin-top:7px';hint.textContent='故障链路统一标红；节点与端点圈叉颜色区分故障来源。地球背面会被遮挡，可选择对象后聚焦。这里只显示当前有效故障，累计统计包含已恢复故障。';host.append(hint);
 let current,rows=[],positions=new Map(),visible=[],pendingSource=null,uiKey='';
 window.focusRuntimeFault=source=>{pendingSource=source};
 function focusSelected(){
  const row=visible.find(r=>r.type+'|'+r.id===select.value);if(!row)return;
  const ids=row.type==='node'?[row.id]:[row.object.endpoint_a,row.object.endpoint_b],points=ids.map(id=>positions.get(id)).filter(Boolean);
  if(!points.length)return;onFocus?.();
  const center=points.reduce((sum,p)=>sum.add(p),new THREE.Vector3()).multiplyScalar(1/points.length),direction=center.clone().normalize();
  if(center.length()<1)direction.copy(points[0]).normalize();
  controls.target.copy(center);controls.minDistance=.25;camera.position.copy(center).add(direction.multiplyScalar(Math.max(3,points.length>1?points[0].distanceTo(points[1])*1.5:3)));controls.update();
  if(row.type==='node')window.inspectNetworkNode?.(row.id);else window.inspectNetworkLink?.(row.id);
 }
 focus.onclick=focusSelected;group.onchange=()=>current&&update(current);
 function update(state){
  current=state;rows=collect(state.snapshot);
  if(pendingSource)group.value=pendingSource;
  positions=new Map(state.snapshot.nodes.map(n=>{const p=(window.computeCellCenter?.(n.node_id)||vector(n.position_itrs_km)).clone();p.add(p.clone().normalize().multiplyScalar(.035));return [n.node_id,p]}));
  visible=rows.filter(r=>group.value==='all'||r.sources.includes(group.value));
  const values=Object.fromEntries(Object.keys(colors).map(k=>[k,{points:[],tips:[],lines:[],pointIds:[],tipIds:[],lineIds:[]}]));
  for(const row of visible){
   for(const name of row.sources){const data=values[name];if(row.type==='node'){const p=positions.get(row.id);if(p){data.points.push(...p.toArray());data.pointIds.push(row)}}
    else{const a=positions.get(row.object.endpoint_a),b=positions.get(row.object.endpoint_b);if(a&&b){data.lines.push(...a.toArray(),...b.toArray());data.lineIds.push(row);data.tips.push(...a.toArray(),...b.toArray());data.tipIds.push(row,row)}}
   }
  }
  const event=state.runtime?.event;
  if(event?.kind==='fault_restored'||event?.kind==='topology_command'&&['node_restore','link_restore','connect'].includes(event.action)){
   const n=state.snapshot.nodes.find(n=>n.node_id===event.target_id),e=state.snapshot.links.find(e=>e.link_id===event.target_id),data=values.restored;
   if(n?.state==='UP'&&positions.has(n.node_id)){data.points.push(...positions.get(n.node_id).toArray());data.pointIds.push({type:'node',id:n.node_id})}
   else if(e?.state==='UP'&&!e.sources?.length&&positions.has(e.endpoint_a)&&positions.has(e.endpoint_b)){data.lines.push(...positions.get(e.endpoint_a).toArray(),...positions.get(e.endpoint_b).toArray());data.lineIds.push({type:'link',id:e.link_id})}
  }
  for(const [name,layer]of Object.entries(layers))for(const [kind,ids]of [['points','pointIds'],['tips','tipIds'],['lines','lineIds']]){
   const geometry=layer[kind].geometry;geometry.setAttribute('position',new THREE.Float32BufferAttribute(values[name][kind],3));geometry.computeBoundingSphere();layer[kind].userData.ids=values[name][ids];
  }
  summary.textContent=`当前 ${state.runtime?.time_s??state.snapshot.time_s} s 故障节点 ${rows.filter(r=>r.type==='node').length} 故障链路 ${rows.filter(r=>r.type==='link').length}（含太阳避让）`;
  const key=group.value+'|'+visible.map(r=>r.type+'|'+r.id+'|'+r.sources.join(',')).join(';');
  if(key===uiKey){if(pendingSource&&visible.length){pendingSource=null;focusSelected()}return}uiKey=key;
  hud.replaceChildren();
  for(const [key,{label,color}]of Object.entries(categories)){const own=counts(rows,key);badges[key].textContent=`⊗ ${label}：${own.nodes} 节点 / ${own.links} 链路`;if(own.nodes||own.links){const line=document.createElement('div');line.style.color=color;line.textContent=badges[key].textContent;hud.append(line)}}
  if(!hud.children.length)hud.textContent='当前无有效故障 已恢复故障仅计入累计统计';
  const prior=select.value,chosen=state.runtime?.event?.target_id||state.runtime?.event?.link_id;select.replaceChildren();
  for(const row of visible){const option=document.createElement('option');option.value=row.type+'|'+row.id;option.textContent=row.sources.map(s=>categories[s].label).join(' + ')+' '+(row.type==='node'?'节点 ':'链路 ')+row.id;select.append(option)}
  const choice=visible.find(r=>r.type+'|'+r.id===prior)||visible.find(r=>r.id===chosen)||visible[0];
  select.value=choice?choice.type+'|'+choice.id:'';
  focus.disabled=select.disabled=!visible.length;
  if(!visible.length){const option=document.createElement('option');option.textContent='当前时刻没有此类有效故障';select.append(option)}
  if(pendingSource&&visible.length){pendingSource=null;focusSelected()}
 }
 function pick(raycaster,earthDistance){
  const hits=raycaster.intersectObjects(Object.values(layers).flatMap(l=>[l.points,l.tips,l.lines])).filter(h=>h.distance<earthDistance+.03);
  if(!hits.length)return false;
  const hit=hits[0],row=hit.object.userData.ids[hit.object.isPoints?hit.index:Math.floor(hit.index/2)];
  if(row){if(row.type==='node')window.inspectNetworkNode?.(row.id);else window.inspectNetworkLink?.(row.id);return true}return false;
 }
 return {update,pick};
}
