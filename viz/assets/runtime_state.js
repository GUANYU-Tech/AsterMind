(function(root){
'use strict';
function key(leg){return leg.transfer_id||[leg.link_id,leg.source_node_id,leg.start_s,leg.stage].join('|')}
function stateAt(record,events,index,transportModel='queue'){
const seen=events.slice(0,index+1),event=events[index]||{time_s:record.request.arrival_time_s,kind:'waiting'};
const observed=new Map(),states=new Map(),kinds=new Set();
for(const e of seen){
 kinds.add(e.kind);
 if(['transfer_queued','transmit_start','transmit_complete','hop_arrival','transfer_dropped','transfer_retried','transfer_cancelled'].includes(e.kind)){
  observed.set(key(e),{...(observed.get(key(e))||{}),...e});
  states.set(key(e),({transfer_queued:'queued',transmit_start:'transmitting',transmit_complete:'propagating',hop_arrival:'delivered',transfer_dropped:'dropped',transfer_retried:'queued',transfer_cancelled:'cancelled'})[e.kind]);
 }
 if(e.kind==='request_failed')for(const [id,status]of states)if(!['delivered','dropped'].includes(status))states.set(id,'cancelled');
}
const paths=new Map(record.routing.selected_path.map(leg=>[key(leg),leg]));
for(const [id,leg]of observed)if(!paths.has(id))paths.set(id,leg);
const legs=Array.from(paths.values()).map(leg=>{
 const id=key(leg),status=states.get(id)||(kinds.has('request_failed')?'cancelled':!leg.transfer_id&&event.time_s>=leg.start_s-(leg.queue_wait_s||0)?'queued':'future');
 if(transportModel==='queue')return {...leg,status};
 const actual=observed.get(id)||{};
 return {...leg,...actual,status,start_s:actual.start_s==null?null:actual.start_s,finish_s:actual.finish_s==null?null:actual.finish_s,arrival_s:actual.arrival_s==null?null:actual.arrival_s,transmission_s:actual.finish_s==null?undefined:actual.transmission_s,queue_wait_s:actual.queue_wait_s};
});
let compute='waiting';
if(kinds.has('compute_start'))compute='prefill';
if(kinds.has('prefill_complete'))compute='decode';
if(kinds.has('compute_complete'))compute='finished';
if(kinds.has('request_failed')&&compute!=='finished')compute='cancelled';
return {time_s:event.time_s,event,legs,compute,
 routePlans:seen.filter(e=>e.kind==='route_plan'||e.kind==='scene_preview'),
 recoveryEvents:seen.filter(e=>e.kind==='route_recovery'),
 lossEvents:seen.filter(e=>e.kind==='transfer_cancelled'),
 completed:kinds.has('request_complete'),failed:kinds.has('request_failed')};
}
function heldFrameIndex(frames,time){
 let low=0,high=frames.length;
 while(low<high){const mid=(low+high)>>1;if(frames[mid].time_s<=time)low=mid+1;else high=mid}
 return low-1;
}
function eventIndexAt(events,current){
 const order=current.event_order==null?Infinity:current.event_order;
 for(let i=events.length-1;i>=0;i--){const e=events[i],before=e.event_order==null?0:e.event_order;if(e.time_s<current.time_s||e.time_s===current.time_s&&before<=order)return i}
 return -1;
}

function activeTransfersAt(initial,events,index){
 const flows=new Map(initial.map(leg=>[leg.request_id+'|'+key(leg),leg]));
 for(const event of events.slice(0,index+1)){
  const id=event.request_id+'|'+key(event),status=({transfer_queued:'queued',transmit_start:'transmitting',transmit_complete:'propagating',transfer_retried:'queued'})[event.kind];
  if(status)flows.set(id,{...event,status});
  else if(['hop_arrival','transfer_dropped','transfer_cancelled'].includes(event.kind))flows.delete(id);
  else if(event.kind==='bandwidth_update'&&flows.has(id))flows.set(id,{...flows.get(id),allocated_bandwidth_bps:event.allocated_bandwidth_bps,sent_bytes:(flows.get(id).sent_bytes||0)+(event.bytes_sent||0)});
  else if(event.kind==='request_failed')for(const [id,leg]of flows)if(leg.request_id===event.request_id)flows.delete(id);
 }
 return Array.from(flows.values());
}
const api={stateAt,key,heldFrameIndex,eventIndexAt,activeTransfersAt};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.RuntimePlayback=api;
})(typeof window!=='undefined'?window:globalThis);
