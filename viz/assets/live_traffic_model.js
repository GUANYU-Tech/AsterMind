// Display aggregation only: never merge solver events or change simulation time.
export const MAX_TRAFFIC_CHANNELS=512;
export const TRAFFIC_PALETTE=Object.freeze({queued:0x9eb4cb,transmitting:0xffce79,propagating:0xbb9dff,delivered:0x83dab6,cancelled:0xff8585,compute:0xd99cff});
export const TRAFFIC_LEGEND=Object.freeze([
 ['queued','排队中'],['transmitting','发送中'],['propagating','传播中'],['delivered','已到达'],['cancelled','已中断'],
]);
export const channelKey=leg=>JSON.stringify([leg.link_id||'',leg.source_node_id,leg.target_node_id]);
const activeStatuses=new Set(['queued','transmitting','propagating']);
export function matchesTraffic(leg,{scope='all',selection={},query='',flowId=''}={}){
 if(flowId&&endToEndFlowId(leg)!==flowId)return false;
 if(scope==='node'&&(!selection.nodeId||![leg.source_node_id,leg.target_node_id].some(id=>id===selection.nodeId||id?.startsWith(selection.nodeId+'/'))))return false;
 if(scope==='link'&&(!selection.linkId||leg.link_id!==selection.linkId))return false;
 const text=query.trim().toLowerCase();
 return !text||[leg.request_id,leg.business_request_id,leg.transfer_id,leg.cell_transfer_id].some(id=>String(id||'').toLowerCase().includes(text));
}
export function aggregateTraffic(legs,filter={},limit=MAX_TRAFFIC_CHANNELS,preferredKey){
 const groups=new Map(),counts={queued:0,transmitting:0,propagating:0};let total=0,matched=0;
 for(const leg of legs){
  if(!activeStatuses.has(leg.status))continue;
  total++;if(!matchesTraffic(leg,filter))continue;
  matched++;counts[leg.status]++;
  const key=channelKey(leg);
  if(!groups.has(key))groups.set(key,{key,link_id:leg.link_id,source_node_id:leg.source_node_id,target_node_id:leg.target_node_id,count:0,queued:0,transmitting:0,propagating:0});
  const group=groups.get(key);group.count++;group[leg.status]++;
 }
 // Stable ordering keeps crowded scenes from reshuffling when flow counts change.
 const channels=[...groups.values()].sort((a,b)=>a.key<b.key?-1:a.key>b.key?1:0);
 const visible=channels.slice(0,limit),preferred=preferredKey&&groups.get(preferredKey);
 if(preferred&&limit>0&&!visible.includes(preferred))visible[visible.length-1]=preferred;
 const linkCount=new Set(channels.map(c=>c.link_id||JSON.stringify([c.source_node_id,c.target_node_id].sort()))).size;
 return {total,matched,counts,linkCount,channelCount:channels.length,hiddenChannels:Math.max(0,channels.length-limit),channels:visible};
}
export function createTrafficClock(){
 let last=null,wasPlaying=false,phase=.2;
 return {sample(now,{playing=false,rate=1}={}){
  if(last!==null&&playing&&wasPlaying)phase=(phase+Math.max(0,Math.min(100,now-last))/1000*Math.max(1,Number(rate)||1))%1;
  last=now;wasPlaying=playing;return phase;
 }};
}
export function trafficMarkerFraction(status,phase,offset=0,playing=false){
 return playing&&['transmitting','propagating'].includes(status) ? .25+((phase+offset)%1)*.5 : .5;
}
export function endToEndFlowId(event={}){
 if(event.time_semantics==='whole_path_shared_resources')return event.v1_flow_id||event.id;
 return event.cell_transfer_id||(event.request_id?.startsWith('__cell_transfer__/')?event.request_id.slice('__cell_transfer__/'.length):null)||(event.kind==='cell_transfer_terminal'?event.id:null);
}
export function pathAt(path,order=Infinity){
 if(!path||path.start_order>order)return [];
 if(path.time_semantics==='whole_path_shared_resources')return (path.segments||[]).map(s=>({...s,status:path.terminal_order<=order?(path.terminal==='completed'?'delivered':'cancelled'):'transmitting'}));
 return path.hops.filter(h=>h.start_order<=order).map(h=>({...h,status:h.cancelled_order<=order?'cancelled':h.arrived_order<=order?'delivered':h.propagating_order<=order?'propagating':h.sent_order<=order?'transmitting':'queued'}));
}
export function pathComplete(path,order,time,grouped=false){return !!path&&(path.terminal_order<=order||grouped&&Number.isFinite(path.terminal_s)&&path.terminal_s<=time)}
export function eventsForPath(path,maxOrder=Infinity){
 if(path.time_semantics==='whole_path_shared_resources')return [
  {...path.raw_admission,kind:'v1_admit',event_label:'V1 整条路径传输开始',category:'business',event_order:path.start_order},
  ...(Number.isInteger(path.terminal_order)?[{kind:path.terminal==='completed'?'v1_p2p_complete':'v1_failed',id:path.id,request_id:path.request_id,event_label:'V1 整条路径传输终态',category:'business',event_order:path.terminal_order,time_s:path.terminal_s,time_ticks:path.terminal_ticks}]:[])
 ].filter(e=>e.event_order<=maxOrder);
 const events=[],base={category:'business',request_id:'__cell_transfer__/'+path.id,business_request_id:path.request_id,cell_transfer_id:path.id,stage:path.stage};
 for(const hop of path.hops||[])for(const [field,kind]of [['start','transfer_queued'],['sent','transmit_start'],['propagating','transmit_complete'],['arrived','hop_arrival'],['cancelled','transfer_cancelled']]){
  const order=hop[field+'_order'],time=hop[field+'_s'];
  if(Number.isInteger(order)&&Number.isFinite(time)&&order<=maxOrder)events.push({...base,...hop,kind,event_order:order,time_s:time});
 }
 if(Number.isInteger(path.terminal_order)&&path.terminal_order<=maxOrder)events.push({...base,kind:'cell_transfer_terminal',event_order:path.terminal_order,time_s:path.terminal_s,terminal_kind:path.terminal,event_label:path.terminal==='failed'?'传输失败':'传输已到达终点'});
 return events.sort((a,b)=>a.event_order-b.event_order);
}
