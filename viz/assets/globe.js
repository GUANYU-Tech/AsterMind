import * as THREE from '/assets/three.module.js';
import {attachOrbitPlanes} from './globe_orbits.js';
import {OrbitControls} from '/assets/OrbitControls.js';
import {pickGlobeNode} from '/assets/globe_picking.js';
import {createTopologyLinkLayers,updateTopologyLinkLayers,setTopologyLinksVisible} from '/assets/globe_links.js';
const runtimeMode=Boolean(window.runtimeGlobeMode);
const old=document.getElementById('globe');
old.style.display='none';
const container=document.createElement('div');container.style.cssText='width:100%;height:560px;position:relative;background:#02060d;border-radius:12px;overflow:hidden';
container.className='globe-viewport';
old.after(container);
const toolbar=document.createElement('div');toolbar.style.cssText='display:flex;gap:16px;flex-wrap:wrap;margin:10px 0';
for(const[id,text,checked]of [['showOrbits','轨道面',false],['showLinks','显示链路',!runtimeMode||window.runtimeLiveMode],['autoRotate','自动环绕',false]]){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.id=id;input.checked=checked;label.append(input,document.createTextNode(text));toolbar.append(label)}container.after(toolbar);
if(window.runtimeLiveMode){
 const options=document.createElement('details'),summary=document.createElement('summary');
 options.className='live-secondary';summary.textContent='三维显示选项';
 options.append(summary);toolbar.after(options);options.append(toolbar);
}
const note=document.createElement('div');note.textContent='滚轮缩放 拖动环绕 点击节点或链路查看物理数据 卫星与太阳翼模型尺寸已放大';note.style.cssText='font-size:12px;color:#9fb2cc;margin:6px 0';toolbar.after(note);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x02050b);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;container.append(renderer.domElement);
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.02,500);camera.position.set(13,8,15);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.07;controls.minDistance=6.6;controls.maxDistance=60;controls.autoRotateSpeed=.35;
const sunlight=new THREE.DirectionalLight(0xfff5e5,3.2);sunlight.position.set(20,10,10);scene.add(sunlight,new THREE.AmbientLight(0x7caddb,.18));
const texture=new THREE.TextureLoader().load('/assets/earth.jpg');texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=renderer.capabilities.getMaxAnisotropy();
const earth=new THREE.Mesh(new THREE.SphereGeometry(6.378137,96,64),new THREE.MeshStandardMaterial({map:texture,roughness:.86,metalness:.02}));earth.scale.y=6356.752314/6378.137;scene.add(earth);
const atmosphere=new THREE.Mesh(new THREE.SphereGeometry(6.49,72,48),new THREE.ShaderMaterial({uniforms:{color:{value:new THREE.Color(0x3289ec)}},vertexShader:'varying vec3 n; varying vec3 v; void main(){vec4 p=modelViewMatrix*vec4(position,1.0);n=normalize(normalMatrix*normal);v=normalize(-p.xyz);gl_Position=projectionMatrix*p;}',fragmentShader:'uniform vec3 color; varying vec3 n; varying vec3 v; void main(){float edge=pow(1.0-max(dot(normalize(n),normalize(v)),0.0),4.0);gl_FragColor=vec4(color,edge*0.5);}',transparent:true,side:THREE.FrontSide,depthWrite:false,blending:THREE.AdditiveBlending}));scene.add(atmosphere);
let seed=1729;function random(){seed=(seed*1664525+1013904223)>>>0;return seed/4294967296}
const starPositions=new Float32Array(6000);for(let i=0;i<2000;i++){const z=random()*2-1,a=random()*Math.PI*2,r=110;starPositions[i*3]=Math.sqrt(1-z*z)*Math.cos(a)*r;starPositions[i*3+1]=z*r;starPositions[i*3+2]=Math.sqrt(1-z*z)*Math.sin(a)*r}
const starGeo=new THREE.BufferGeometry();starGeo.setAttribute('position',new THREE.BufferAttribute(starPositions,3));scene.add(new THREE.Points(starGeo,new THREE.PointsMaterial({color:0xb9d7ff,size:.13,sizeAttenuation:true,transparent:true,opacity:.7,depthWrite:false})));
while(!window.networkReplayState()?.snapshot?.nodes)await new Promise(resolve=>setTimeout(resolve,40));
const initial=window.networkReplayState(),catalog=initial.data.network.nodes,space=catalog.filter(n=>n.role!=='ground'&&(!n.cell_id||window.collectivePhysicalMembers)),grounds=catalog.filter(n=>n.role==='ground');
const cellVisual=(await import('/assets/cell_visual.js')).attachComputeCells({THREE,scene,container,vector:v=>new THREE.Vector3(v[0]/1000,v[2]/1000,-v[1]/1000),catalog});
const body=new THREE.InstancedMesh(new THREE.BoxGeometry(.035,.023,.025),new THREE.MeshStandardMaterial({color:0xe8bb63,metalness:.72,roughness:.3}),space.length);
const panelCanvas=document.createElement('canvas');panelCanvas.width=256;panelCanvas.height=64;const panelCtx=panelCanvas.getContext('2d');panelCtx.fillStyle='#173973';panelCtx.fillRect(0,0,256,64);panelCtx.strokeStyle='#82a4ce';panelCtx.lineWidth=1;for(let x=2;x<256;x+=16){panelCtx.beginPath();panelCtx.moveTo(x,0);panelCtx.lineTo(x,64);panelCtx.stroke()}for(let y=2;y<64;y+=16){panelCtx.beginPath();panelCtx.moveTo(0,y);panelCtx.lineTo(256,y);panelCtx.stroke()}const panelTexture=new THREE.CanvasTexture(panelCanvas);panelTexture.colorSpace=THREE.SRGBColorSpace;
const wings=new THREE.InstancedMesh(new THREE.BoxGeometry(.12,.003,.034),new THREE.MeshStandardMaterial({color:0xffffff,map:panelTexture,metalness:.48,roughness:.35,emissive:0x082247,emissiveIntensity:.25}),space.length);
scene.add(body,wings);
const groundMesh=new THREE.InstancedMesh(new THREE.ConeGeometry(.035,.09,5),new THREE.MeshStandardMaterial({color:0xffb95a,emissive:0xb36518,emissiveIntensity:.4}),grounds.length);scene.add(groundMesh);
const orbitToggle=document.getElementById('showOrbits');
const geometryPreview = initial.data.provenance?.scope === 'orbital_geometry_preview';
if (geometryPreview) {
  orbitToggle.checked = true;
  document.getElementById('showLinks').checked = false;
}
orbitToggle.parentElement.title='按当前高度显示瞬时轨道面，地球会遮挡背面的轨道线。';
const orbitLegend=document.createElement('span'), orbitStatus=document.createElement('span');
orbitLegend.hidden=true;
for (const [text,color] of [['通信轨道面','#4ed8ff'],['计算轨道面','#ffb745']]) {
  const swatch=document.createElement('span');swatch.textContent='━ ';swatch.style.color=color;
  const label=document.createElement('span');label.textContent=text;
  orbitLegend.append(swatch,label,document.createTextNode('　'));
}
orbitStatus.setAttribute('role','status');orbitLegend.append(orbitStatus);toolbar.append(orbitLegend);
const updateOrbitPlanes=attachOrbitPlanes(THREE,scene,container,orbitStatus,initial);
const linkColors={communication_isl:0x45cfff,sgl:0xffa42f,cross_layer:0xf1a1ff,cell_isl:0xbb8aff,inter_cell:0xf47fbd};
const linkLayers=createTopologyLinkLayers({THREE,scene,colors:linkColors,live:window.runtimeLiveMode});
const markerGroups=[];
for(const[role,color,size]of [['communication',0x54d8ff,6],['compute',0xd99cff,10],['ground',0xffbe47,19]]){const nodes=catalog.filter(n=>n.role===role&&(!n.cell_id||window.collectivePhysicalMembers)),geometry=new THREE.BufferGeometry(),canvas=document.createElement('canvas');canvas.width=canvas.height=64;const c=canvas.getContext('2d');c.fillStyle='#fff';c.beginPath();if(role==='ground'){c.moveTo(32,3);c.lineTo(61,58);c.lineTo(3,58);c.closePath()}else if(role==='compute'){c.rect(8,8,48,48)}else c.arc(32,32,24,0,Math.PI*2);c.fill();const map=new THREE.CanvasTexture(canvas),mesh=new THREE.Points(geometry,new THREE.PointsMaterial({color,size,sizeAttenuation:false,map,transparent:true,alphaTest:.2,depthWrite:false}));mesh.userData.nodes=nodes;scene.add(mesh);markerGroups.push({nodes,geometry,mesh,role})}
const linkLegend=document.createElement('div');linkLegend.style.cssText='font-size:12px;line-height:1.8;color:#acc5df';linkLegend.innerHTML='<span style="color:#45cfff">━ 通信 ISL</span>　<span style="color:#ffa42f">━ 星地链路</span>　<span style="color:#f1a1ff">━ 通信—计算</span>　<span style="color:#bb8aff">━ cell 内部</span><br>● 通信卫星　▤ 计算 cell（点击展开）　▲ 地面站 标记按屏幕大小显示，虚线为建链中';note.after(linkLegend);
const pathGeometry=new THREE.BufferGeometry(),pathLines=new THREE.LineSegments(pathGeometry,new THREE.LineBasicMaterial({color:0xffd66f,transparent:true,opacity:.95}));scene.add(pathLines);
const vector=v=>new THREE.Vector3(v[0]/1000,v[2]/1000,-v[1]/1000),dummy=new THREE.Object3D(),up=new THREE.Vector3(0,1,0);
const overviewPosition=camera.position.clone();
if (geometryPreview && initial.data.sun_positions?.length) {
  const sun=vector(initial.data.sun_positions[0].position_itrs_km).normalize();
  const north=up.clone().addScaledVector(sun,-up.dot(sun)).normalize();
  overviewPosition.crossVectors(up,sun).normalize().addScaledVector(north,.23)
    .addScaledVector(sun,.10).normalize().multiplyScalar(22);
  camera.position.copy(overviewPosition);
  controls.update();
}
function segments(target,pairs,positions){const points=[];for(const[a,b]of pairs){if(positions.has(a)&&positions.has(b)){points.push(...positions.get(a).toArray(),...positions.get(b).toArray())}}target.setAttribute('position',new THREE.Float32BufferAttribute(points,3));target.computeBoundingSphere()}
let signature='',focusId=null,latestPositions=new Map();
const focusButton=document.createElement('button');focusButton.textContent='卫星特写';toolbar.append(focusButton);
const overviewButton=document.createElement('button');overviewButton.textContent='整体星座';toolbar.append(overviewButton);
if(runtimeMode){focusButton.hidden=true;overviewButton.hidden=true}
focusButton.onclick=()=>{const state=window.networkReplayState(),sun=state.data.sun_positions?.find(s=>s.time_s===Math.floor(state.snapshot.time_s/60)*60)?.position_itrs_km||[1e8,0,0],direction=vector(sun).normalize();const chosen=space.filter(n=>latestPositions.has(n.node_id)).sort((a,b)=>latestPositions.get(b.node_id).clone().normalize().dot(direction)-latestPositions.get(a.node_id).clone().normalize().dot(direction))[0];if(!chosen)return;focusId=chosen.node_id;const position=latestPositions.get(focusId);controls.target.copy(position);camera.position.copy(position).add(position.clone().normalize().multiplyScalar(.45)).add(new THREE.Vector3(.1,.08,.1));controls.minDistance=.12;window.inspectNetworkNode?.(focusId)};
overviewButton.onclick=()=>{focusId=null;controls.target.set(0,0,0);camera.position.copy(overviewPosition);controls.minDistance=6.6};
if(!runtimeMode){const routeFocus=document.createElement('button');routeFocus.textContent='聚焦所选路径';routeFocus.onclick=()=>{const state=window.networkReplayState(),ids=new Set((state.route?.path||[]).flatMap(e=>[e.source_node_id,e.target_node_id])),positions=[...ids].map(id=>latestPositions.get(id)).filter(Boolean);if(!positions.length){note.textContent='请先查询并选择一条可达路径';return}const direction=positions.reduce((sum,p)=>sum.add(p.clone().normalize()),new THREE.Vector3());if(direction.length()<.01)direction.copy(positions[0]);focusId=null;controls.target.set(0,0,0);camera.position.copy(direction.normalize().multiplyScalar(18));controls.minDistance=6.6;controls.update()};toolbar.append(routeFocus)}

function update(state){const snapshot=state.snapshot,positions=new Map(snapshot.nodes.map(n=>[n.node_id,vector(n.position_itrs_km)]));
cellVisual.update(state);latestPositions=positions;if(focusId&&positions.has(focusId)){const delta=positions.get(focusId).clone().sub(controls.target);camera.position.add(delta);controls.target.copy(positions.get(focusId))}

space.forEach((n,i)=>{dummy.position.copy(positions.get(n.node_id));dummy.quaternion.setFromUnitVectors(up,dummy.position.clone().normalize());const scale=(n.role==='compute'?2.1:1)*(window.collectiveModelScale||1);dummy.scale.setScalar(scale);dummy.updateMatrix();body.setMatrixAt(i,dummy.matrix);wings.setMatrixAt(i,dummy.matrix)});body.instanceMatrix.needsUpdate=true;wings.instanceMatrix.needsUpdate=true;
grounds.forEach((n,i)=>{dummy.position.copy(positions.get(n.node_id));dummy.quaternion.setFromUnitVectors(up,dummy.position.clone().normalize());dummy.scale.setScalar(1);dummy.updateMatrix();groundMesh.setMatrixAt(i,dummy.matrix)});groundMesh.instanceMatrix.needsUpdate=true;
updateTopologyLinkLayers(THREE,linkLayers,snapshot,positions,state.kind);
for(const group of markerGroups){const points=[];for(const n of group.nodes){const p=positions.get(n.node_id).clone();if(group.role==='ground')p.add(p.clone().normalize().multiplyScalar(.008));points.push(...p.toArray())}group.geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));group.geometry.computeBoundingSphere()}
const paths=window.runtimeLiveMode?[]:state.route?.path||[];
segments(pathGeometry,paths.map(e=>[e.source_node_id,e.target_node_id]),positions);
pathLines.userData.links=paths;
}
function resize(){
 const width=container.clientWidth,height=container.clientHeight;
 if(!width||!height)return;
 renderer.setSize(width,height);
 camera.aspect=width/height;camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(container);resize();
const raycaster=new THREE.Raycaster(),mouse=new THREE.Vector2();let pointer;
let selectedObject={},selectionSignature='',overlay=null,pickedOrbitNode;
const selectionMarker=new THREE.Mesh(new THREE.RingGeometry(.78,1,48),new THREE.MeshBasicMaterial({color:0xc5e78b,side:THREE.DoubleSide,depthWrite:false}));selectionMarker.visible=false;scene.add(selectionMarker);
const connectedSelection=new THREE.LineBasicMaterial({color:0xc5e78b,depthWrite:false}),disconnectedSelection=new THREE.LineDashedMaterial({color:0xe9ba70,depthWrite:false,dashSize:.06,gapSize:.04});
const selectionLink=new THREE.Line(new THREE.BufferGeometry(),connectedSelection);selectionLink.visible=false;scene.add(selectionLink);
window.selectGlobeObject=value=>{selectedObject=value||{};pickedOrbitNode=selectedObject.nodeId;overlay?.select?.(selectedObject)};
function updateSelection(){
 const point=selectedObject.nodeId&&(window.computeCellCenter?.(selectedObject.nodeId)||latestPositions.get(selectedObject.nodeId));
 selectionMarker.visible=!!point;if(point){selectionMarker.position.copy(point);selectionMarker.quaternion.copy(camera.quaternion);selectionMarker.scale.setScalar(camera.position.distanceTo(point)*Math.tan(camera.fov*Math.PI/360)*28/container.clientHeight)}
 const key=signature+'|'+(selectedObject.linkId||'');
 if(key!==selectionSignature){selectionSignature=key;const edge=selectedObject.linkId&&window.networkReplayState().snapshot.links.find(e=>e.link_id===selectedObject.linkId);
  selectionLink.visible=!!edge;if(edge){selectionLink.material=edge.state==='UP'?connectedSelection:disconnectedSelection;segments(selectionLink.geometry,[[edge.endpoint_a,edge.endpoint_b]],latestPositions);selectionLink.computeLineDistances()}
 }
}
renderer.domElement.addEventListener('pointerdown',e=>pointer=[e.clientX,e.clientY]);
renderer.domElement.addEventListener('click',e=>{
 if(!pointer||Math.hypot(e.clientX-pointer[0],e.clientY-pointer[1])>5)return;
 const r=container.getBoundingClientRect(),screen={x:e.clientX-r.left,y:e.clientY-r.top};
 mouse.set(screen.x/r.width*2-1,-screen.y/r.height*2+1);raycaster.setFromCamera(mouse,camera);
 raycaster.params.Points.threshold=camera.position.distanceTo(controls.target)*.006;
 raycaster.params.Line.threshold=8*camera.position.distanceTo(controls.target)*Math.tan(camera.fov*Math.PI/360)/r.height;
 const earthDistance=raycaster.intersectObject(earth)[0]?.distance??Infinity;
 const cell=cellVisual.pick(raycaster,earthDistance);
 if(cell){pickedOrbitNode=catalog.find(n=>n.cell_id===cell && n.compute_target)?.node_id || catalog.find(n=>n.cell_id===cell)?.node_id;return}
 const node=pickGlobeNode({THREE,camera,positions:latestPositions,nodes:catalog.filter(n=>!n.cell_id||window.collectivePhysicalMembers),pointer:screen,width:r.width,height:r.height,radius:7});
 if(node){pickedOrbitNode=node;window.inspectNetworkNode?.(node);return}
 if(faults?.pick(raycaster,earthDistance))return;
 const link=raycaster.intersectObjects([pathLines,...linkLayers.filter(l=>l.mesh.visible).map(l=>l.mesh)]).find(h=>h.distance<earthDistance);
 if(link){const edge=link.object.userData.links[Math.floor(link.index/2)];if(edge){pickedOrbitNode=null;window.inspectNetworkLink?.(edge.link_id)}}
});
overlay=runtimeMode?(await import(window.runtimeLiveMode?'/assets/live_traffic.js':'/assets/runtime_overlay.js')).attachOverlay({THREE,scene,camera,controls,container,toolbar,vector}):null;
const faults=runtimeMode?(await import('/assets/runtime_faults.js')).attachFaults({THREE,scene,vector,camera,controls,toolbar,container,onFocus:()=>overlay?.stopFocus()}):null;
const solar=(await import('/assets/sun_visual.js')).attachSun({THREE,scene,sunlight,camera,controls,container,toolbar,vector,onFocus:()=>{focusId=null;overlay?.stopFocus()}});
if(!runtimeMode)await import('/assets/physical_motion.js');
window.orbitGlobe={THREE,scene,camera,controls,container,vector,overlay};
renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label',runtimeMode?'真实 Runtime 三维地球、故障与传输路径':'三维地球与物理拓扑');
function animate(){requestAnimationFrame(animate);if(!container.clientWidth||!container.clientHeight)return;const state=window.networkReplayState(),key=state.snapshot.time_s+'|'+(state.runtime?.event?.event_order??'')+'|'+state.kind+'|'+(state.revision??'')+'|'+(document.getElementById('probe')?.value||'')+'|'+(state.route?.probe_revision??'');if(key!==signature){signature=key;update(state);if(faults)faults.update(state);solar.update(state)}orbitLegend.hidden=!orbitToggle.checked;updateOrbitPlanes(state,orbitToggle.checked,pickedOrbitNode);controls.autoRotate=document.getElementById('autoRotate').checked;controls.update();if(overlay)overlay.update(state);updateSelection();const linksVisible=document.getElementById('showLinks').checked;setTopologyLinksVisible(linkLayers,linksVisible,pathLines);selectionLink.visible=linksVisible&&!!selectedObject.linkId&&!!selectionLink.geometry.getAttribute('position')?.count;overlay?.setLinksVisible?.(linksVisible);const visibleCount=String(linksVisible?linkLayers.reduce((n,l)=>n+l.mesh.userData.links.length,0):0);if(toolbar.dataset.visibleLinks!==visibleCount)toolbar.dataset.visibleLinks=visibleCount;solar.animate?.(performance.now()/1000);renderer.render(scene,camera)}animate();
