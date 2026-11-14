const transmissionKinds=new Set(['transfer_queued','transmit_start','transmit_complete','hop_arrival','transfer_cancelled','transfer_dropped','transfer_retried','external_flow_done','route_recovery','fault_injected','fault_restored','topology_command','v1_admit','v1_service','v1_p2p_complete','v1_retry','v1_failed']);
export function compareTime(event,target){
 if(event.time_ticks!=null&&target?.time_ticks!=null){const a=BigInt(event.time_ticks),b=BigInt(target.time_ticks);return a<b?-1:a>b?1:0}
 const a=event.time_s,b=typeof target==='object'?target.time_s:target;return a<b?-1:a>b?1:0;
}
export function formatTime(event){
 if(event?.time_ticks==null)return Number(event?.time_s??event).toPrecision(17);
 const tick=BigInt(event.time_ticks),whole=tick/1000000000000000000n,part=(tick%1000000000000000000n).toString().padStart(18,'0').replace(/0+$/,'');
 return whole.toString()+(part?'.'+part:'');
}
export function isTransmissionEvent(event){
 if(!transmissionKinds.has(event.kind))return false;
 return !event.source_node_id||!event.target_node_id||event.source_node_id!==event.target_node_id;
}
export function watchTimeline(intervals,minDuration=.6){
 let end=0;
 const segments=intervals.map(([from,to])=>{const duration=Math.max(minDuration,to-from),start=end;end+=duration;return {from,to,start,end}});
 const toSimulation=watch=>{
  if(!segments.length)return 0;
  const s=segments.find(s=>watch<s.end)||segments.at(-1),fraction=Math.max(0,Math.min(1,(watch-s.start)/(s.end-s.start)));
  return fraction===1?s.to:fraction===0?s.from:s.from+(s.to-s.from)*fraction;
 };
 const toWatch=time=>{
  const s=segments.find(s=>time<=s.to);if(!s)return end;
  return s.start+Math.max(0,Math.min(1,(time-s.from)/(s.to-s.from)))*(s.end-s.start);
 };
 return {segments,end,toSimulation,toWatch};
}
export function upperTime(events,time){let low=0,high=events.length;while(low<high){const mid=(low+high)>>1;if(compareTime(events[mid],time)<=0)low=mid+1;else high=mid}return low-1}
export function countAt(times,time){let low=0,high=times.length;while(low<high){const mid=(low+high)>>1;if(typeof times[mid]==='string'&&time?.time_ticks!=null?BigInt(times[mid])<=BigInt(time.time_ticks):times[mid]<=(time?.time_s??time))low=mid+1;else high=mid}return low}
export async function communicationPositionAt(descriptors,loadPage,time){
 let pageIndex=descriptors.findLastIndex(p=>compareTime({time_s:p.start_s,time_ticks:p.start_ticks},time)<=0);
 for(;pageIndex>=0;pageIndex--){
  const page=await loadPage(pageIndex),visible=page.visible.findLastIndex(i=>compareTime(page.events[i],time)<=0);
  if(visible>=0)return {page,pageIndex,visible,cut:upperTime(page.events,time)};
 }
 const pageIndexFirst=descriptors.findIndex(p=>p.count),page=await loadPage(pageIndexFirst);
 return {page,pageIndex:pageIndexFirst,visible:-1,cut:upperTime(page.events,time)};
}
