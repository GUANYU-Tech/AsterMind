const $=id=>document.getElementById(id),runId=location.pathname.split('/')[2];
const data=window.networkReplayState().data;
const labels={min_hops:'最少跳数',min_propagation:'最短累计传播距离'};
const reasons={route_endpoint_or_time_unavailable:'已到场景终点，无后续传输时间',no_current_route:'当前拓扑无可用路径'};
let enabled=false,busy=false,response=null,selectedPolicy='min_hops',lastTime=null,revision=0,failedKey=null;
const routeCache=new Map();
function request(){const snapshot=window.physicalSnapshot();return {run_id:runId,time_s:snapshot.time_s,source:$('routeSource').value.trim(),target:$('routeTarget').value.trim()}}
function key(payload){return JSON.stringify(payload)}
function valid(){const p=request();return data.network.nodes.some(n=>n.node_id===p.source)&&data.network.nodes.some(n=>n.node_id===p.target)}
function current(){return enabled&&response?.queryKey===key(request())}
window.physicalRouteActive=()=>enabled;
window.interactivePhysicalRoute=()=>current()?{...response.routes.find(r=>r.policy===selectedPolicy),probe_revision:revision}:null;
function render(){const table=$('routeComparison');table.replaceChildren();if(!current())return;
 for(const r of response.routes){const tr=document.createElement('tr'),radio=document.createElement('input');radio.type='radio';radio.name='physicalPolicy';radio.checked=r.policy===selectedPolicy;radio.setAttribute('aria-label','高亮'+labels[r.policy]);radio.onchange=()=>{selectedPolicy=r.policy;revision++;render();window.redrawPhysical()};
 const td=document.createElement('td');td.append(radio);tr.append(td);const pathNodes=[r.source,...r.path.map(l=>l.target_node_id)],nodes=r.completed?(pathNodes.length>5?[...pathNodes.slice(0,2),'…',pathNodes.at(-1)]:pathNodes).join(' → '):(reasons[r.reason]||r.reason);
 for(const text of [labels[r.policy],r.completed?r.hops:'—',r.completed?r.distance_km.toFixed(3):'—',nodes]){const cell=document.createElement('td');cell.textContent=text;tr.append(cell)}
 tr.title=r.completed?pathNodes.join(' → '):r.reason;tr.style.background=r.policy===selectedPolicy?'#263d55':'';tr.style.cursor='pointer';tr.onclick=()=>{selectedPolicy=r.policy;revision++;render();window.showPhysicalDetails(r);window.redrawPhysical()};table.append(tr)}
}
function useCached(payload){const queryKey=key(payload),cached=response?.queryKey===queryKey?response:routeCache.get(queryKey);if(!cached)return false;const changed=response!==cached;response=cached;if(changed)revision++;$('routeQueryStatus').textContent=`${payload.time_s} s 当前拓扑 黄色为所选路径；不预测消息传输时间`;render();if(changed)window.redrawPhysical();return true}
async function query(payload){if(busy)return;const queryKey=key(payload);busy=true;render();$('routeQueryStatus').textContent=`正在${payload.time_s===request().time_s?'比较':'预取'} ${payload.time_s} s 的路径…`;try{const result=await fetch('/physical-route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),value=await result.json();if(!result.ok)throw Error(value.error||'路径查询失败');routeCache.set(queryKey,{...value,queryKey});while(routeCache.size>8)routeCache.delete(routeCache.keys().next().value);failedKey=null;
 }catch(error){failedKey=queryKey;$('routeQueryStatus').textContent=error.message}
 finally{busy=false;if(enabled&&valid())compare()}
}
function compare(){if(!enabled||!valid())return;const payload=request();if(useCached(payload)||failedKey===key(payload)||busy)return;query(payload)}
window.physicalRouteCanAdvance=time=>!enabled||!valid()||routeCache.has(key({...request(),time_s:time}))||failedKey===key({...request(),time_s:time});
window.preparePhysicalRoute=time=>{if(!enabled||!valid())return;compare();if(current()&&!busy){const payload={...request(),time_s:time};if(!routeCache.has(key(payload))&&failedKey!==key(payload))query(payload)}};
window.onPhysicalFrame=frame=>{if(lastTime!==frame.time_s){lastTime=frame.time_s;render();if(enabled)compare()}};
$('compareRoutes').onclick=()=>{if(!valid()){$('routeQueryStatus').textContent='请选择此模型内的起点和终点';return}enabled=true;failedKey=null;compare()};
$('probe').addEventListener('input',()=>{enabled=false;render();window.redrawPhysical()});
for(const id of ['routeSource','routeTarget'])$(id).addEventListener('change',()=>{failedKey=null;render();window.redrawPhysical();if(enabled)compare()});
$('swapRoute').onclick=()=>{const a=$('routeSource').value;$('routeSource').value=$('routeTarget').value;$('routeTarget').value=a;render();window.redrawPhysical();if(enabled)compare()};
for(const n of data.network.nodes){const option=document.createElement('option');option.value=n.node_id;option.label=n.role+(n.cell_id?' '+n.cell_id:'');$('physicalNodes').append(option)}
const nodes=data.network.nodes;
$('routeSource').value=nodes.find(n=>n.role==='ground')?.node_id||nodes[0]?.node_id||'';
$('routeTarget').value=nodes.find(n=>n.role==='compute'&&n.compute_target)?.node_id||nodes.find(n=>n.role==='compute')?.node_id||nodes.at(-1)?.node_id||'';
const actions=document.createElement('div');actions.className='controls';$('details').before(actions);
const endpointActions=document.createElement('div');endpointActions.className='controls';$('routeQueryStatus').before(endpointActions);
for(const [text,id]of [['设为起点','routeSource'],['设为终点','routeTarget']]){const button=document.createElement('button');button.textContent=text;button.onclick=()=>{try{const n=JSON.parse($('details').textContent);if(!n.node_id)throw Error();$(id).value=n.node_id;$(id).dispatchEvent(new Event('change'))}catch{$('routeQueryStatus').textContent='请先点击一个节点'}};endpointActions.append(button)}
const selection=document.createElement('p');selection.textContent='点击图中节点 / 链路查看属性；选中节点后可设为起终点。';endpointActions.after(selection);
const rate=bps=>{for(const [unit,scale]of [['Gbps',1e9],['Mbps',1e6],['kbps',1e3],['bps',1]])if(bps>=scale)return (bps/scale).toFixed(3)+' '+unit;return bps===0?'0 bps':'—'};
function directionName(obj,d,i){if(obj.kind!=='sgl')return i===0?'A→B ':'B→A ';const ground=data.network.nodes.find(n=>n.role==='ground'&&(n.node_id===obj.endpoint_a||n.node_id===obj.endpoint_b));if(!ground)return i===0?'A→B ':'B→A ';const up=d.source_node_id===ground.node_id;return ground.ground_kind==='user'?(up?'用户上传 ':'用户下载 '):(up?'网关→星 ':'星→网关 ')}
window.onPhysicalSelection=obj=>{if(!obj)return;if(obj.node_id){selection.textContent=obj.node_id+' '+({compute:'计算卫星',communication:'通信卫星',ground:'地面站'}[obj.role]||obj.role)+' ITRS ('+(obj.position_itrs_km||[]).map(v=>v.toFixed(3)).join(', ')+') km'}else if(obj.link_id){selection.textContent=obj.link_id+' '+obj.state+' 属性采样 '+(obj.attributes_time_s??window.physicalSnapshot().time_s)+' s 距离 '+(obj.distance_km?.toFixed(3)??'—')+' km 传播 '+(obj.propagation_delay_s==null?'—':(obj.propagation_delay_s*1000).toFixed(3))+' ms 每方向带宽 '+(obj.directions||[]).map((d,i)=>directionName(obj,d,i)+rate(d.bandwidth_bps)).join(' / ')}};
const inspect=document.createElement('label');inspect.textContent='查找节点 ';const input=document.createElement('input');input.setAttribute('aria-label','查找节点');input.setAttribute('list','physicalNodes');input.oninput=input.onchange=()=>{if(data.network.nodes.some(n=>n.node_id===input.value))window.inspectNetworkNode(input.value)};inspect.append(input);actions.append(inspect);
const linkSearch=document.createElement('input');linkSearch.setAttribute('aria-label','查找链路');linkSearch.placeholder='输入节点或链路 ID';actions.append(linkSearch);
const linkSelect=document.createElement('select');linkSelect.setAttribute('aria-label','匹配链路');linkSelect.style.maxWidth='100%';actions.append(linkSelect);
function updateLinks(frame){const old=linkSelect.value,query=linkSearch.value.trim(),matches=frame.links.filter(e=>query?e.link_id.includes(query):e.state!=='DOWN').sort((a,b)=>({UP:0,SETTING_UP:1,DOWN:2}[a.state]-{UP:0,SETTING_UP:1,DOWN:2}[b.state]));linkSelect.replaceChildren();const initial=document.createElement('option');initial.textContent=`${matches.length} 条匹配 最多显示 100 条 可输入 ID 筛选`;initial.value='';linkSelect.append(initial);for(const e of matches.slice(0,100)){const o=document.createElement('option');o.value=e.link_id;o.textContent=e.state+' '+e.link_id;linkSelect.append(o)}linkSelect.value=old}
linkSearch.oninput=()=>updateLinks(window.networkReplayState().snapshot);
let linkTime=null;const previousFrame=window.onPhysicalFrame;
window.onPhysicalFrame=frame=>{previousFrame(frame);if(frame.time_s===linkTime)return;linkTime=frame.time_s;updateLinks(frame)};
linkSelect.onchange=()=>{if(linkSelect.value)window.inspectNetworkLink(linkSelect.value)};
if(data.provenance?.transport_execution==='not_run_geometry_and_independent_route_probes_only'){
 $('runtimeViewLink').hidden=true;document.querySelector('h1').textContent='AsterMind 星鉴 物理场景演示';document.title='AsterMind 星鉴 物理场景演示';
 document.querySelector('footer').textContent=`几何采样间隔 ${data.time.step_s} s。按存储时间片播放，片内保持位置和几何属性；容量调度保留精确边界。此页只展示物理环境和路径查询，未执行 Serving 或流量后端。`;
 const edit=document.createElement('a');edit.textContent='返回物理配置';edit.style.color='#8bd8ff';edit.href='/physical';document.querySelector('nav').append(edit);
 fetch(`/runs/${runId}/status`).then(r=>r.json()).then(job=>{if(job.physical_model_id)edit.href='/physical?id='+encodeURIComponent(job.physical_model_id)}).catch(()=>{});
}
if(!data.routes.length)$('probe').parentElement.style.display='none';
document.querySelector('footer').textContent+=' 启用路由对比后，播放会等下一片路径就绪再推进，速率随查询耗时改变。';
window.redrawPhysical();
