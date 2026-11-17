// Candidate edges are available for inspection, but only allocated connections
// belong in the background topology. DOWN candidates must never become a fan.
export function createTopologyLinkLayers({THREE,scene,colors,live=false}){
 const layers=[];
 for(const [kind,color]of Object.entries(colors))for(const state of ['UP','SETTING_UP']){
  const geometry=new THREE.BufferGeometry();
  const material=state==='UP'
   ?new THREE.LineBasicMaterial({color,transparent:true,opacity:live ? .55 : .48,depthWrite:false})
   :new THREE.LineDashedMaterial({color,transparent:true,opacity:.4,depthWrite:false,dashSize:.03,gapSize:.02});
  const mesh=new THREE.LineSegments(geometry,material);scene.add(mesh);layers.push({kind,state,geometry,mesh});
 }
 return layers;
}

export function setTopologyLinksVisible(layers,visible,...extra){
 for(const mesh of [...layers.map(layer=>layer.mesh),...extra])mesh.visible=visible;
}

export function updateTopologyLinkLayers(THREE,layers,snapshot,positions,kind='all'){
 for(const layer of layers){
  const links=snapshot.links.filter(e=>e.state===layer.state&&e.kind===layer.kind&&(kind==='all'||kind===e.kind)&&positions.has(e.endpoint_a)&&positions.has(e.endpoint_b));
  const points=links.flatMap(e=>[...positions.get(e.endpoint_a).toArray(),...positions.get(e.endpoint_b).toArray()]);
  layer.mesh.userData.links=links;
  layer.geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
  layer.geometry.computeBoundingSphere();
  if(layer.state==='SETTING_UP')layer.mesh.computeLineDistances();
 }
}
