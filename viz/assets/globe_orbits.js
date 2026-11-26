import {createOrbitSampleLoader,orbitPlanes} from './orbit_planes.js';

// Extrude each ring in screen space. WebGL's native line width is often
// fixed at one device pixel, particularly faint on high-DPI screens.
function ringGeometry(T) {
  const points=[], sides=[], indices=[], segments=192;
  for (let i=0;i<=segments;i++) {
    const a=i*2*Math.PI/segments, x=Math.cos(a), y=Math.sin(a);
    points.push(x,y,0,x,y,0); sides.push(-1,1);
    const j=2*i;
    if (i<segments) indices.push(j,j+1,j+2,j+2,j+1,j+3);
  }
  const g=new T.BufferGeometry();
  g.setAttribute('position',new T.Float32BufferAttribute(points,3));
  g.setAttribute('sideSign',new T.Float32BufferAttribute(sides,1)); g.setIndex(indices);
  return g;
}

export function attachOrbitPlanes(T,scene,container,status,initial) {
  const geometry=ringGeometry(T), viewport=new T.Vector2();
  const transform=new T.Object3D(), axis=new T.Vector3(0,0,1), normal=new T.Vector3();
  const layers=new Map();
  for (const [name,color,halfWidth] of [['communication',0x4ed8ff,1.6],['compute',0xffb745,2.1],['selected',0xfff5ac,3]]) {
    const material=new T.ShaderMaterial({
      uniforms:{viewport:{value:viewport},halfWidth:{value:halfWidth},ink:{value:new T.Color(color)}},
      side:T.DoubleSide,transparent:true,depthWrite:false,toneMapped:false,
      vertexShader:`
        attribute float sideSign;
        uniform vec2 viewport; uniform float halfWidth; varying float across;
        void main(){
          mat4 m=projectionMatrix*modelViewMatrix*instanceMatrix;
          vec4 p=m*vec4(position,1.), t=m*vec4(-position.y,position.x,0.,0.);
          vec2 d=(t.xy*p.w-p.xy*t.w)*viewport;
          vec2 n=vec2(-d.y,d.x)/max(length(d),1.e-7);
          p.xy+=n*sideSign*halfWidth*2./viewport*p.w;
          gl_Position=p; across=sideSign;
        }`,
      fragmentShader:`
        uniform vec3 ink; uniform float halfWidth; varying float across;
        void main(){
          float edge=(1.-abs(across))*halfWidth;
          gl_FragColor=vec4(mix(vec3(.012,.019,.028),ink,smoothstep(.35,1.0,edge)),smoothstep(0.,.45,edge));
          #include <colorspace_fragment>
        }`
    });
    layers.set(name,{material,mesh:null});
  }
  function draw(guides,selected) {
    for (const [name,layer] of layers) {
      const rows=guides.filter(g=>name==='selected' ? selected && g.members.includes(selected) : g.role===name);
      if (rows.length && (!layer.mesh || layer.mesh.instanceMatrix.count<rows.length)) {
        if (layer.mesh) { scene.remove(layer.mesh); layer.mesh.dispose(); }
        layer.mesh=new T.InstancedMesh(geometry,layer.material,rows.length);
        layer.mesh.name='orbit-plane-'+name; layer.mesh.frustumCulled=false;
        layer.mesh.renderOrder=name==='selected'?3:name==='compute'?2:1;
        scene.add(layer.mesh);
      }
      if (!layer.mesh) continue;
      layer.mesh.count=rows.length;
      rows.forEach(({normal:n,radius},i)=>{
        transform.quaternion.setFromUnitVectors(axis,normal.set(n[0],n[2],-n[1]));
        transform.scale.setScalar(radius/1000); transform.updateMatrix();
        layer.mesh.setMatrixAt(i,transform.matrix);
      });
      layer.mesh.instanceMatrix.needsUpdate=true;
    }
  }
  const loadDefault=createOrbitSampleLoader(location.pathname.replace(/[^/]*$/,''),initial.data);
  let epoch,revision=0,selection,guides=[],enabled=false;
  function show(result,pending=false) {
    guides=result.guides; draw(guides,selection);
    status.textContent=result.missing ? pending ? '正在读取轨道采样…' :
      '部分轨道面缺少可用的运动采样' : '';
  }
  return function update(state,on,nodeId) {
    if (on!==enabled) {
      enabled=on; epoch=undefined; revision++;
      for (const {mesh} of layers.values()) if (mesh) mesh.visible=on;
      if (!on) status.textContent='';
    }
    if (!enabled) return;
    viewport.set(container.clientWidth,container.clientHeight);
    const frame=state.orbitSnapshot || (!window.runtimeGlobeMode && window.physicalSnapshot?.()) || state.snapshot;
    const time=frame.physical_time_s ?? frame.time_s;
    if (time===epoch) {
      if (nodeId!==selection) { selection=nodeId; draw(guides,selection); }
      return;
    }
    epoch=time; selection=nodeId;
    const own=++revision, result=orbitPlanes(frame.nodes,time);
    if (!result.missing) { show(result); return; }
    if (!guides.length) show(result,true);
    Promise.resolve().then(()=>(state.loadOrbitSample || loadDefault)(time)).then(sample=>{
      if (own===revision) show(sample ? orbitPlanes(frame.nodes,time,sample) : result);
    }).catch(()=>{if (own===revision) show(result)});
  };
}
