(function(root){
'use strict';
const faultKey=e=>JSON.stringify([e.target_type,e.target_id,e.source,e.start_s,e.end_s]);
const reducers=new WeakMap();
function stateAt(initial={},events=[],index=events.length-1,requested){
 index=Math.min(events.length-1,Math.max(-1,Math.trunc(index)));
 let state=reducers.get(events);
 // Pages are immutable. Forward playback consumes only newly crossed events;
 // an explicit seek backward rebuilds from the page's saved causal boundary.
 if(!state||state.initial!==initial||index<state.index||state.last!==events[state.index]){
  const flows=new Map((initial.active_transfers||[]).map(e=>[e.transfer_id,{...e}]));
  state={initial,index:-1,last:undefined,flows,legs:new Map(flows),
   plans:new Map((initial.route_plans||[]).map(e=>[[e.request_id,e.stage,e.reason].join('|'),e])),
   faults:new Map((initial.faults||[]).map(e=>[faultKey(e),e])),
   recoveries:[...(initial.recovery_events||[])],losses:[...(initial.loss_events||[])],kinds:new Map()};
  reducers.set(events,state);
 }
 const {flows,legs,plans,faults,recoveries,losses}=state;
 let event=state.last||{kind:'control_boundary',time_s:0},compute='waiting';
 for(let i=state.index+1;i<=index;i++){
  const e=events[i];
  if(!state.kinds.has(e.request_id))state.kinds.set(e.request_id,new Set());
  state.kinds.get(e.request_id).add(e.kind);
  if(e.kind==='v1_admit')for(const part of e.path_legs||[]){const leg={...e,...part,status:'transmitting',v1_flow_id:e.id};flows.set(part.transfer_id,leg);legs.set(part.transfer_id,leg)}
  if(['v1_p2p_complete','v1_retry','v1_failed'].includes(e.kind))for(const[id,leg]of flows)if(leg.v1_flow_id===e.id){flows.delete(id);legs.set(id,{...leg,status:e.kind==='v1_p2p_complete'?'delivered':'cancelled'})}
  event=e;const status=({transfer_queued:'queued',transmit_start:'transmitting',transmit_complete:'propagating',hop_arrival:'delivered',transfer_cancelled:'cancelled'})[e.kind];
  if(status){const leg={...legs.get(e.transfer_id),...e,status};legs.set(e.transfer_id,leg);if(['delivered','cancelled'].includes(status))flows.delete(e.transfer_id);else flows.set(e.transfer_id,leg)}
  if(e.kind==='bandwidth_update'&&flows.has(e.transfer_id)){const leg={...flows.get(e.transfer_id),...e,sent_bytes:(flows.get(e.transfer_id).sent_bytes||0)+(e.bytes_sent||0)};flows.set(e.transfer_id,leg);legs.set(e.transfer_id,leg)}
  if(e.kind==='route_plan')plans.set([e.request_id,e.stage,e.reason].join('|'),e);
  if(e.kind==='route_recovery')recoveries.push(e);
  if(e.kind==='transfer_cancelled')losses.push(e);
  if(e.kind==='fault_injected')faults.set(faultKey(e),e);
  if(e.kind==='fault_restored')faults.delete(faultKey(e));
  if(['request_failed','request_complete'].includes(e.kind))for(const [id,leg]of flows)if(leg.request_id===e.request_id){flows.delete(id);legs.set(id,{...leg,status:'cancelled'})}
 }
 state.index=index;state.last=index>=0?events[index]:undefined;
 const request=requested||event.request_id||[...flows.values()][0]?.request_id||[...plans.values()].at(-1)?.request_id||'';
 const kinds=state.kinds.get(request)||new Set();
 if(kinds.has('compute_start'))compute='prefill';if(kinds.has('prefill_complete'))compute='decode';if(kinds.has('compute_complete'))compute='finished';
 return {time_s:event.time_s,event,globalMode:true,activeTransfers:[...flows.values()],legs:[...legs.values()].filter(e=>e.request_id===request),
  routePlans:[...plans.values()].filter(e=>e.request_id===request),recoveryEvents:recoveries.filter(e=>e.request_id===request),
  lossEvents:losses.filter(e=>e.request_id===request),faults:[...faults.values()],request_id:request,compute,
  completed:kinds.has('request_complete'),failed:kinds.has('request_failed')};
}
function applyFaults(snapshot,faults){
 const sources=(kind,id,ends=[])=>[...new Set(faults.filter(f=>f.target_type===kind&&f.target_id===id||f.target_type==='node'&&ends.includes(f.target_id)).map(f=>f.source||'scheduled'))];
 return {...snapshot,nodes:snapshot.nodes.map(n=>{const causes=sources('node',n.node_id);return {...n,sources:causes,...(causes.length?{state:'DOWN'}:{})}}),
  links:snapshot.links.map(e=>{const causes=sources('link',e.link_id,[e.endpoint_a,e.endpoint_b]);return {...e,sources:causes,...(causes.length?{state:'DOWN',reason:'injected_fault'}:{})}})};
}
const api={stateAt,applyFaults};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.RuntimeVisual=api;
})(typeof window!=='undefined'?window:globalThis);
