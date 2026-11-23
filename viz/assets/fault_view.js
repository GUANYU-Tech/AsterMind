(function(root){
'use strict';
let data,host,time=0;
const svg=(name,attrs,text)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e};
function render(now){time=now;if(!data||!host)return;if(data.metadata.mode==='large_constellation_recovery'){root.LargeRecoveryDemo.render(now);return}if(data.metadata.mode==='current_node_recovery'){renderBuffered();return}host.replaceChildren();const r=data.runs.reference,f=data.runs.failure,a=r.input_path[0],b=f.input_path[0],backup=f.input_path[1],end=Math.max(15,backup.finish_s)+1,x=t=>160+(t-7)/(end-7)*720;
const title=document.createElement('p');title.textContent='同一物理带宽与消息量的两次实际运行：7 s 拓扑正常；7.25 s 开始上传；无故障首跳在 '+a.finish_s.toFixed(3)+' s 发送完成；10 s 断链，11 s 重选备用接入。';host.append(title);
const graph=svg('svg',{viewBox:'0 0 940 220',style:'width:100%;min-width:650px',role:'img','aria-label':'长消息传输与10秒断链恢复对照'});
for(const[t,label]of [[7,'7 拓扑正常'],[7.25,'7.25 开始'],[10,'10 断链'],[11,'11 重传'],[15,'15 无故障完成']]){graph.append(svg('line',{x1:x(t),y1:30,x2:x(t),y2:175,stroke:'#45617c','stroke-dasharray':'3 4'}));graph.append(svg('text',{x:x(t),y:t===7.25?215:20,fill:'#adc4dc','font-size':12},label))}
for(const[y,label,start,finish,color]of [[65,'无故障实测',a.start_s,a.finish_s,'#7bdcb2'],[120,'原链路中断',b.start_s,b.cancelled_at_s,'#ff8b8b'],[165,'备用接入重传',backup.start_s,backup.finish_s,'#80caff']]){graph.append(svg('text',{x:4,y:y+5,fill:color,'font-size':14},label));graph.append(svg('rect',{x:x(start),y:y-9,width:Math.max(1,x(finish)-x(start)),height:18,rx:3,fill:color}))}
if(time>=7&&time<=end)graph.append(svg('line',{x1:x(time),y1:25,x2:x(time),y2:190,stroke:'#ffdf81','stroke-width':2}));host.append(graph);
const summary=document.createElement('p');summary.textContent='消息 '+(data.metadata.input_bytes/2**20).toFixed(2)+' MiB 10 s 丢弃已发送 '+(f.losses[0].bytes_discarded/2**20).toFixed(2)+' MiB 11 s 整条重传 备用首跳 '+backup.link_id+' 发送完成 '+backup.finish_s.toFixed(6)+' s。此处为整条消息存储转发。';host.append(summary);
if(f.route_audit){const route=document.createElement('p'),a=f.route_audit;route.textContent='故障后实际输入路线：'+a.successful_input_hops+' 跳 '+a.distance_km.toFixed(1)+' km 纬度 '+a.min_geocentric_latitude_deg.toFixed(1)+'°～'+a.max_geocentric_latitude_deg.toFixed(1)+'° 到达原计算 cell '+a.input_arrival_at_cell_s.toFixed(6)+' s。备用接入已比较到同一 cell 的完整路径。';route.style.color='#80caff';host.append(route)}
}
function renderBuffered(){host.replaceChildren();const r=data.runs.reference,f=data.runs.failure,broken=f.input_path.find(l=>l.cancelled_at_s!=null),recovery=f.recoveries[0],done=f.input_path.at(-1).arrival_s,normal=r.input_path.at(-1).arrival_s,end=Math.max(done,normal)+1,x=t=>180+(t-7)/(end-7)*710;
const title=document.createElement('p');title.textContent='7.25 s 从用户发送；无故障时 '+normal.toFixed(3)+' s 输入到达计算 cell。10 s 中途星间链路中断，已完成的上游跳保留；11 s 从 '+recovery.source_node_id+' 重选后续路线。';host.append(title);
const graph=svg('svg',{viewBox:'0 0 940 270',style:'width:100%;min-width:650px',role:'img','aria-label':'保留已完成路径并从当前节点改道'});
for(const[t,label]of [[7,'7 正常拓扑'],[7.25,'7.25 发送'],[10,'10 断链'],[11,'11 当前节点改道'],[15,'15 正常到达']]){graph.append(svg('line',{x1:x(t),y1:30,x2:x(t),y2:235,stroke:'#45617c','stroke-dasharray':'3 4'}));graph.append(svg('text',{x:x(t),y:t===7.25?260:20,fill:'#adc4dc','font-size':12},label))}
for(const[y,label,start,finish,color]of [[65,'无故障完整输入',7.25,normal,'#528dff'],[115,'上游已完成并保留',7.25,broken.start_s,'#7bdcb2'],[165,'当前跳在途丢弃',broken.start_s,10,'#ff354f'],[215,'当前节点绕行',11,done,'#35e4ef']]){graph.append(svg('text',{x:2,y:y+5,fill:color,'font-size':13},label));graph.append(svg('rect',{x:x(start),y:y-9,width:Math.max(1,x(finish)-x(start)),height:18,rx:3,fill:color}))}
if(time>=7&&time<=end)graph.append(svg('line',{x1:x(time),y1:25,x2:x(time),y2:240,stroke:'#ffdf81','stroke-width':2}));host.append(graph);
const prefix=f.input_path.filter(l=>l.arrival_s!=null&&l.arrival_s<=10&&!l.cancelled_at_s),summary=document.createElement('p');summary.textContent='已完成 '+prefix.length+' 跳保留；缓存节点 '+recovery.source_node_id+' 保有 '+(broken.payload_bytes/2**20).toFixed(2)+' MiB。仅当前故障跳丢弃已发送 '+(f.losses[0].bytes_discarded/2**20).toFixed(2)+' MiB；原用户接入只发送一次。改道后 '+done.toFixed(6)+' s 输入到达原 cell。';host.append(summary);
const note=document.createElement('p');note.textContent='本场景采用标明工程假设的 E-band 网关预算与光学链路预算，便于展示中间跳故障。Serving 解析通信耗时保持原样。';host.append(note);
}
async function initialize(){host=document.getElementById('faultTimeline');if(!host)return;try{const r=await fetch(location.pathname.replace(/runtime$/,'fault-reference'));const d=await r.json();if(d.metadata?.mode==='large_constellation_recovery'){data=d;root.LargeRecoveryDemo.initialize(d);render(time);return}if(!d.runs?.failure)return;data=d;document.getElementById('scenarioJumps').hidden=false;document.getElementById('faultPanel').hidden=false;render(time)}catch{}}
root.FaultView={initialize,render};
})(window);
