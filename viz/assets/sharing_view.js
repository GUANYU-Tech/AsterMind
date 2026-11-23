(function(root){
'use strict';
const colors=['#5bd4f1','#f6c45f','#c49bff','#7fdfa9','#fb91b5','#96b9ff'];
let evidence;
function svg(name,attrs,text){const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const[k,v]of Object.entries(attrs))node.setAttribute(k,v);if(text!==undefined)node.textContent=text;return node}
function chart(mode,start,end){
 const rows=evidence[mode].flows,height=rows.length*36+45,plot=svg('svg',{viewBox:'0 0 800 '+height,role:'img',style:'width:100%;height:auto;min-width:650px','aria-label':mode+' 实际发送与等待时间线'});
 const x=t=>110+(t-start)/(end-start)*660;
 rows.forEach((flow,i)=>{
  const y=35+i*36;plot.append(svg('text',{x:8,y:y+15,fill:colors[i%6],'font-size':13},flow.request_id));
  plot.append(svg('line',{x1:x(flow.ready_s),x2:x(flow.start_s),y1:y+9,y2:y+9,stroke:'#8396ae','stroke-width':3,'stroke-dasharray':'4 3'}));
  for(const segment of flow.segments){const rect=svg('rect',{x:x(segment.start_s),y,width:Math.max(.4,x(segment.finish_s)-x(segment.start_s)),height:19,rx:2,fill:colors[i%6],opacity:Math.max(.25,Math.min(1,segment.bandwidth_bps/(segment.link_capacity_bps||evidence.peak.link_capacity_bps)))});rect.append(svg('title',{},flow.request_id+' '+segment.start_s.toFixed(3)+'–'+segment.finish_s.toFixed(3)+' s '+(segment.bandwidth_bps/1000).toFixed(3)+' kbit/s'));plot.append(rect)}
  plot.append(svg('text',{x:x(flow.finish_s)+3,y:y+15,fill:'#d1e1f2','font-size':10},flow.finish_s.toFixed(2)+' s'));
 });
 for(let t=Math.ceil(start);t<=end;t++){plot.append(svg('text',{x:x(t),y:15,fill:'#a9bfd8','font-size':10},t+' s'))}
 return plot;
}
async function initialize(){
 const panel=document.getElementById('sharingComparison');
 try{
  evidence=await(await fetch(location.pathname.replace(/runtime$/,'comparison'))).json();
  if(evidence.available===false){panel.textContent='当前为单模式结果。请在配置页选择“共享对照”并点击“运行 Queue / SimGrid 对照”。';return}
  panel.replaceChildren();
  const intro=document.createElement('p');intro.textContent='同一输入、同一星座、同一有向链路：'+evidence.source_node_id+' → '+evidence.target_node_id+'。以下来自两次实际执行；虚线表示等待，色块表示发送。色块深浅表示占用链路容量的比例。';panel.append(intro);
  for(const mode of ['queue','simgrid']){const heading=document.createElement('h3');heading.textContent=mode==='queue'?'Queue：同向独占，后续流排队':'SimGrid：同向并发，速率随竞争变化';panel.append(heading);const flows=[...evidence.queue.flows,...evidence.simgrid.flows];panel.append(chart(mode,Math.min(...flows.map(f=>f.ready_s))-.25,Math.max(...flows.map(f=>f.finish_s))+1))}
  const peak=evidence.peak,heading=document.createElement('h3');heading.textContent='真实共享区间 '+peak.start_s.toFixed(3)+'–'+peak.finish_s.toFixed(3)+' s '+peak.flows.length+' 条同向流 链路 '+(peak.link_capacity_bps/1000).toFixed(3)+' kbit/s';panel.append(heading);
  const stack=document.createElement('div');stack.style.cssText='display:flex;height:42px;border:1px solid #7085a3;border-radius:7px;overflow:hidden';
  peak.flows.forEach((flow,i)=>{const part=document.createElement('div');part.style.cssText='display:flex;align-items:center;justify-content:center;color:#08101d;font-size:12px;background:'+colors[i%6]+';width:'+(flow.bandwidth_bps/peak.link_capacity_bps*100)+'%';part.textContent=flow.request_id+' '+(flow.bandwidth_bps/1000).toFixed(3);part.title=flow.request_id+' '+flow.bytes.toFixed(3)+' B sent';stack.append(part)});panel.append(stack);
  const note=document.createElement('p');note.textContent='上条为已结束区间的实际平均分配（kbit/s），总容量不因并发而增加。总工作量相同，重点比较各流的排队、完成时间和公平性。';panel.append(note);
  const button=document.createElement('button');button.textContent='定位六流共享事件';button.onclick=()=>{if(location.pathname.includes('/'+evidence.simgrid.run_id+'/'))root.seekSharedInterval(peak.finish_s);else location.href='/runs/'+evidence.simgrid.run_id+'/runtime'};panel.append(button);
  for(const mode of ['queue','simgrid']){const link=document.createElement('a');link.style.marginLeft='18px';link.href='/runs/'+evidence[mode].run_id+'/runtime';link.textContent='打开 '+mode+' 实际回放';panel.append(link)}
 }catch(error){panel.textContent='无法加载对照：'+error.message}
}
root.SharingView={initialize};
})(window);
