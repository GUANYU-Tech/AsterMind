import '/assets/solar_state.js';

export function attachSun({THREE,scene,sunlight,camera,controls,container,toolbar,vector,onFocus}){
 const group=new THREE.Group();group.name='solar-direction-schematic';scene.add(group);
 const clock={value:0};
 const noise=`
 float hash(vec3 p){p=fract(p*.3183099+vec3(.1,.2,.3));p*=17.;return fract(p.x*p.y*p.z*(p.x+p.y+p.z));}
 float noise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);}
 float fbm(vec3 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*noise(p);p=p*2.03+5.7;a*=.5;}return v;}`;
 const surface=new THREE.ShaderMaterial({uniforms:{time:clock},toneMapped:false,
  vertexShader:'varying vec3 p;varying vec3 n;varying vec3 v;void main(){p=position;vec4 mv=modelViewMatrix*vec4(position,1.);n=normalize(normalMatrix*normal);v=normalize(-mv.xyz);gl_Position=projectionMatrix*mv;}',
  fragmentShader:`uniform float time;varying vec3 p;varying vec3 n;varying vec3 v;${noise}
  void main(){vec3 q=normalize(p);float field=fbm(q*7.+vec3(0.,time*.025,0.));float cells=noise(q*82.+field*3.);float filaments=fbm(q*25.+field*4.+time*.012);float limb=pow(max(dot(normalize(n),normalize(v)),0.),.38);float heat=clamp(field*.65+filaments*.3+cells*.15,0.,1.);vec3 orange=vec3(1.,.23,.025),gold=vec3(1.,.69,.21),cream=vec3(1.,.94,.69);vec3 color=mix(orange,gold,limb);color=mix(color,cream,smoothstep(.28,.78,heat)*limb);color*=.86+cells*.18;gl_FragColor=vec4(color,1.);
  #include <colorspace_fragment>
  }`});
 const sun=new THREE.Mesh(new THREE.SphereGeometry(1.65,80,64),surface);sun.name='sun-photosphere';group.add(sun);
 const corona=new THREE.Mesh(new THREE.PlaneGeometry(9,9),new THREE.ShaderMaterial({uniforms:{time:clock},transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,toneMapped:false,
  vertexShader:'varying vec2 uvPosition;void main(){uvPosition=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
  fragmentShader:`uniform float time;varying vec2 uvPosition;void main(){vec2 p=uvPosition-.5;float r=length(p),a=atan(p.y,p.x);float edge=.1833;float glow=exp(-max(0.,r-edge)*20.);float plume=.8+.12*sin(a*9.+time*.12)+.07*sin(a*17.-time*.08);float alpha=glow*.34*plume*smoothstep(.17,.2,r)*(1.-smoothstep(.38,.5,r));gl_FragColor=vec4(1.,.45,.09,alpha);}`
 }));corona.name='sun-corona';group.add(corona);
 const offsets=[[0,0],[-3.5,0],[3.5,0],[0,-3.5],[0,3.5],[-7.5,0],[7.5,0]],rays=offsets.map(()=>{
  const ray=new THREE.ArrowHelper(new THREE.Vector3(-1,0,0),new THREE.Vector3(),1,0xe9ba70,.28,.1);ray.line.material.transparent=true;ray.line.material.opacity=.22;ray.cone.material.transparent=true;ray.cone.material.opacity=.4;group.add(ray);return ray;
 });let direction=null,available=false,focusPose=null;
 const label=document.createElement('label'),show=document.createElement('input');show.type='checkbox';show.id='showSun';show.checked=true;label.append(show,document.createTextNode('太阳'));toolbar.append(label);
 const focus=document.createElement('button');focus.textContent='查看太阳';toolbar.append(focus);
 const rayLabel=document.createElement('label'),showRays=document.createElement('input');showRays.type='checkbox';showRays.id='showSunRays';showRays.checked=false;rayLabel.append(showRays,document.createTextNode('光线方向'));toolbar.append(rayLabel);
 const info=document.createElement('div');info.setAttribute('aria-label','太阳方向');info.style.cssText='position:absolute;right:12px;top:12px;padding:6px 10px;background:#1a1b14d9;border:1px solid #59513a;border-radius:4px;font-size:11px;color:#e9c88d;pointer-events:none';container.append(info);
 const warning=document.createElement('p');warning.setAttribute('role','status');warning.hidden=true;container.before(warning);
 function visibility(){group.visible=show.checked&&available;info.hidden=!show.checked||!available;focus.disabled=!show.checked||!available;for(const ray of rays)ray.visible=showRays.checked}
 show.onchange=showRays.onchange=visibility;
 focus.onclick=()=>{
  if(!direction)return;onFocus?.();
  if(focusPose){camera.position.copy(focusPose.position);controls.target.copy(focusPose.target);controls.minDistance=focusPose.min;controls.maxDistance=focusPose.max;focusPose=null;focus.textContent='查看太阳'}
  else{focusPose={position:camera.position.clone(),target:controls.target.clone(),min:controls.minDistance,max:controls.maxDistance};const reference=Math.abs(direction.y)>.9?new THREE.Vector3(1,0,0):new THREE.Vector3(0,1,0),side=new THREE.Vector3().crossVectors(direction,reference).normalize();controls.target.copy(sun.position);camera.position.copy(sun.position).addScaledVector(direction,-11).addScaledVector(side,2.5);controls.minDistance=3;controls.maxDistance=80;focus.textContent='返回地球'}
  controls.update();container.scrollIntoView?.({block:'center',behavior:reducedMotion?'auto':'smooth'});
 };
 function update(state){
  warning.textContent=state.data.physics_warning||'';warning.hidden=!warning.textContent;
  const row=SolarVisual.sunInterpolated(state.data.sun_positions||[],state.snapshot.time_s);
  available=!!row;if(!row){visibility();return}
  const previous=sun.position.clone();direction=vector(row.position_itrs_km).normalize();sun.position.copy(direction).multiplyScalar(26);corona.position.copy(sun.position);sunlight.position.copy(direction).multiplyScalar(60);
  if(focusPose){const delta=sun.position.clone().sub(previous);camera.position.add(delta);controls.target.add(delta)}
  const reference=Math.abs(direction.y)>.9?new THREE.Vector3(1,0,0):new THREE.Vector3(0,1,0),u=new THREE.Vector3().crossVectors(direction,reference).normalize(),v=new THREE.Vector3().crossVectors(direction,u).normalize(),incoming=direction.clone().negate();
  offsets.forEach(([x,y],index)=>{const origin=direction.clone().multiplyScalar(22).addScaledVector(u,x).addScaledVector(v,y),hit=SolarVisual.earthEntry(origin.toArray(),incoming.toArray()),length=hit??32;const ray=rays[index];ray.position.copy(origin);ray.setDirection(incoming);ray.setLength(length,.28,.1);ray.userData.earthEntry=hit;});
  info.textContent='太阳';info.title='位置随太阳星历更新；显示距离和尺寸经过缩放。';visibility();
 }
 const reducedMotion=typeof window!=='undefined'&&window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
 function animate(seconds){clock.value=reducedMotion?0:seconds;corona.quaternion.copy(camera.quaternion)}
 return {update,animate};
}
