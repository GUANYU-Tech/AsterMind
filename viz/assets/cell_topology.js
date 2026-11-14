// Both levels use the saved native resources; no illustrative substitute graph.
const ns='http://www.w3.org/2000/svg';
function element(tag,text,attrs={}){const e=document.createElement(tag);if(text!=null)e.textContent=text;for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e}
function svg(tag,attrs={}){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e}
const short=id=>id.split('/').at(-1).replace(/^.*-m/,'卫星 ');
function select(label,values,change){const wrap=element('label',label),input=element('select',null,{'aria-label':label});for(const[value,text]of values)input.append(element('option',text,{value}));input.onchange=()=>change(input.value);wrap.append(input);return[wrap,input]}
function button(text,action){const b=element('button',text,{type:'button'});b.onclick=action;return b}
export function createCellTopology(host,{edit,command,onSatellite}={}){
 if(!document.querySelector('link[data-cell-topology]')){const css=element('link',null,{rel:'stylesheet',href:'/assets/cell_topology.css','data-cell-topology':''});document.head.append(css)}
 host.classList.add('cell-topology');let config=null,live=null,cellId='',owner='',chosen=null,working=false;
 const status=element('p','',{role:'status'}),head=element('div',null,{class:'topology-heading'}),views=element('div',null,{class:'topology-views'}),details=element('div',null,{class:'topology-details'});
 host.append(head,views,details,status);
 const isLive=()=>!!live,editable=()=>!working&&(!isLive()||['paused','advancing'].includes(live.state));
 async function perform(action){if(working)return;working=true;status.textContent='正在应用拓扑…';render();try{const next=await action();if(next?.physical)config=next;status.textContent=isLive()?'拓扑已更新':'拓扑已更新，保存场景后生效。'}catch(e){status.textContent=e.message}finally{working=false;render()}}
 function model(){const cell=config?.physical.cells.find(c=>c.id===cellId),members=new Set(cell?.satellites||[]),devices=new Map((config?.physical.devices||[]).map(d=>[d.id,d]));return{cell,members,devices}}
 function cellEdges(){const{members,devices}=model();return config.physical.links.filter(l=>{const a=devices.get(l.source)?.satellite,b=devices.get(l.target)?.satellite;return a!==b&&members.has(a)&&members.has(b)}).map(l=>{const identity=config.physical_link_bindings[l.id],row=live?.links?.find(e=>e.link_id===identity);return{...l,source:devices.get(l.source).satellite,target:devices.get(l.target).satellite,physicalId:identity,state:row?.state||(config.disabled_physical_links?.includes(identity)?'DOWN':'UP')}})}
 function gpuEdges(){const{devices}=model();return config.physical.links.filter(l=>devices.get(l.source)?.satellite===owner&&devices.get(l.target)?.satellite===owner).map(l=>({...l,state:live?.device_links?.find(e=>e.id===l.id)?.state||(l.enabled===false?'DOWN':'UP')}))}
 function draw(target,nodes,links,kind){
  const canvas=svg('svg',{viewBox:'0 0 400 270',role:'group','aria-label':kind==='cell'?'Cell 内卫星拓扑':'卫星内 GPU 拓扑'}),positions=new Map();
  nodes.forEach((n,i)=>{const angle=-Math.PI/2+Math.PI*2*i/Math.max(nodes.length,1),x=nodes.length===1?200:200+132*Math.cos(angle),y=nodes.length===1?130:130+85*Math.sin(angle);positions.set(n.id,{x,y})});
  for(const l of links){const a=positions.get(l.source),b=positions.get(l.target);if(!a||!b)continue;const group=svg('g',{tabindex:0,role:'button','aria-label':'选择链路 '+short(l.source)+' 到 '+short(l.target),class:'topology-edge '+(l.state==='DOWN'?'is-down':'')+(chosen?.id===l.id?' is-selected':'')});group.append(svg('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'hit'}),svg('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y}));group.onclick=()=>{chosen={...l,kind};render()};group.onkeydown=e=>{if(e.key==='Enter')group.onclick()};canvas.append(group)}
  for(const n of nodes){const p=positions.get(n.id),g=svg('g',{transform:`translate(${p.x},${p.y})`,tabindex:0,role:'button','aria-label':(kind==='cell'?'选择卫星 ':'选择设备 ')+n.id,class:'topology-node '+(n.id===owner?'is-selected':'')});
   if(kind==='cell'){g.append(svg('rect',{x:-18,y:-9,width:10,height:18,rx:1}),svg('rect',{x:8,y:-9,width:10,height:18,rx:1}),svg('rect',{x:-5,y:-12,width:10,height:24,rx:2}))}
   else g.append(svg('rect',{x:-23,y:-16,width:46,height:32,rx:n.kind==='gpu'?5:1}));
   const label=svg('text',{x:0,y:kind==='cell'?32:36,'text-anchor':'middle'});label.textContent=short(n.id);g.append(label);
   if(kind==='gpu'){const inside=svg('text',{x:0,y:4,'text-anchor':'middle'});inside.textContent={gpu:'GPU',switch:'交换',terminal:'终端'}[n.kind];g.append(inside)}
   g.onclick=()=>{if(kind==='cell'){owner=n.id;chosen=null;onSatellite?.(owner);render()}else{chosen={...n,kind:'device'};renderDetails()}};g.onkeydown=e=>{if(e.key==='Enter')g.onclick()};canvas.append(g)
  }target.append(canvas);
 }
 function preset(panel,kind){if(isLive())return;const choices=kind==='cell'?[['ring','环形'],['chain','链式'],['fully_connected','全连接']]:[['switch','共享交换互联'],['ring','GPU 环形直连'],['chain','GPU 链式直连'],['fully_connected','GPU 全直连']];const row=element('div',null,{class:'topology-actions'});for(const[value,text]of choices){const b=button(text,()=>perform(()=>edit({action:kind==='cell'?'cell_topology':'gpu_topology',cell:cellId,satellite:owner,kind:value})));b.disabled=!editable();row.append(b)}panel.append(row)}
 function renderDetails(){details.replaceChildren();if(!config)return;
  if(!chosen){details.append(element('p',isLive()?'选择卫星查看 GPU，选择链路进行断开、重连或故障操作。':'点击卫星或链路编辑；灰色虚线表示已禁用。'));return}
  if(chosen.kind==='device'){const ranks=config.cells.flatMap(c=>c.placements.filter(p=>p.device===chosen.id).map(p=>'rank '+p.rank));details.append(element('strong',chosen.id),element('p',ranks.join(' · ')||'物理连接设备'));return}
  const l=chosen;details.append(element('strong',short(l.source)+' ↔ '+short(l.target)),element('p',l.state==='DOWN'?'已断开 / 故障':l.state==='SETTING_UP'?'正在建链':'已连接'));
  const actions=element('div',null,{class:'topology-actions'});
  if(isLive()){
   for(const[action,label]of [['disconnect','断开'],['connect','重连'],['link_fail','链路故障'],['link_restore','解除故障']]){const b=button(label,()=>perform(()=>command(l.kind==='cell'?{action,target_id:l.physicalId,endpoint_a:l.source,endpoint_b:l.target}:{action:'device_'+action,target_id:l.id})));b.disabled=!editable();actions.append(b)}
  }else{
   const b=button(l.state==='DOWN'?'启用链路':'禁用链路',()=>perform(()=>edit(l.kind==='cell'?{action:'cell_link',cell:cellId,source:l.source,target:l.target,enabled:l.state==='DOWN'}:{action:'gpu_link',cell:cellId,link:l.id,enabled:l.state==='DOWN'})));b.disabled=!editable();actions.append(b);
   if(l.kind==='gpu'){const rate=element('input',null,{type:'number',min:'0.000001',step:'any','aria-label':'GPU 链路带宽 Gbps'}),delay=element('input',null,{type:'number',min:'0',step:'1','aria-label':'GPU 链路时延 ns'});rate.value=l.bandwidth_gbps;delay.value=l.latency_ns||0;for(const[text,input]of [['带宽 / Gbps',rate],['时延 / ns',delay]]){const wrap=element('label',text);wrap.append(input);actions.append(wrap)}const apply=button('应用参数',()=>{if(!rate.reportValidity()||!delay.reportValidity())return;perform(()=>edit({action:'gpu_link',cell:cellId,link:l.id,bandwidth_gbps:Number(rate.value),latency_ns:Number(delay.value)}))});apply.disabled=!editable();actions.append(apply)}
  }details.append(actions)
 }
 function render(){
  head.replaceChildren();views.replaceChildren();if(!config){host.hidden=true;return}host.hidden=false;
  const cells=config.cells;if(!cells.some(c=>c.id===cellId))cellId=cells[0].id;
  const{cell,members,devices}=model();if(!members.has(owner))owner=cell.satellites[0];
  const [field,input]=select('计算 Cell',cells.map(c=>[c.id,c.id]),id=>{cellId=id;owner=config.physical.cells.find(c=>c.id===id).satellites[0];chosen=null;onSatellite?.(owner);render()});input.value=cellId;head.append(field);
  const left=element('section'),right=element('section');left.append(element('h3','Cell 内卫星连接'));right.append(element('h3','卫星内 GPU 互联'));
  const edges=cellEdges();draw(left,[...members].map(id=>({id})),edges,'cell');preset(left,'cell');
  const [satellite,selectSat]=select('当前卫星',[...members].map(id=>[id,short(id)]),id=>{owner=id;chosen=null;onSatellite?.(id);render()});selectSat.value=owner;right.append(satellite);
  const locals=[...devices.values()].filter(d=>d.satellite===owner);draw(right,locals,gpuEdges(),'gpu');preset(right,'gpu');
  right.append(element('small','GPU 直连与交换接入分别显示，切换直连拓扑保留终端接入。'));views.append(left,right);
  if(chosen&&chosen.kind!=='device'){const fresh=(chosen.kind==='cell'?edges:gpuEdges()).find(e=>e.id===chosen.id);chosen=fresh?{...fresh,kind:chosen.kind}:null}
  renderDetails();
 }
 return{update(value,state=null,selectedOwner=null){config=value;live=state;if(selectedOwner){const c=config?.physical.cells.find(c=>c.satellites.includes(selectedOwner)&&config.cells.some(s=>s.id===c.id));if(c){cellId=c.id;owner=selectedOwner}}render()},clear(){config=null;render()}};
}
