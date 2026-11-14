(function(root){
'use strict';
function heldIndex(rows,time){let a=0,b=rows.length;while(a<b){const m=(a+b)>>1;if(rows[m].time_s<=time)a=m+1;else b=m}return a-1}
function hermite(a,va,b,vb,t0,t1,time){
 if(t1<=t0)return [...a];const dt=t1-t0,u=Math.max(0,Math.min(1,(time-t0)/dt)),u2=u*u,u3=u2*u;
 return a.map((x,i)=>(2*u3-3*u2+1)*x+(u3-2*u2+u)*dt*va[i]+(-2*u3+3*u2)*b[i]+(u3-u2)*dt*vb[i]);
}
function pose(snapshot,left,right,time){
 const first=new Map(left.nodes.map(n=>[n.node_id,n])),last=new Map(right.nodes.map(n=>[n.node_id,n]));
 return {...snapshot,physical_time_s:snapshot.physical_time_s??snapshot.time_s,time_s:time,
  nodes:snapshot.nodes.map(node=>{const a=first.get(node.node_id),b=last.get(node.node_id);if(!a||!b)return node;
   return {...node,position_itrs_km:hermite(a.position_itrs_km,a.velocity_itrs_km_s,b.position_itrs_km,b.velocity_itrs_km_s,left.time_s,right.time_s,time)};})};
}
function progress(leg,time){
 if(leg.status==='transmitting'&&leg.payload_bytes>0){
  if(leg.segments?.length){const sent=leg.segments.reduce((sum,s)=>sum+Math.max(0,Math.min(time,s.finish_s)-s.start_s)*s.bandwidth_bps/8,0);return Math.max(0,Math.min(1,sent/leg.payload_bytes))}
  if(Number.isFinite(leg.sent_bytes))return Math.max(0,Math.min(1,leg.sent_bytes/leg.payload_bytes));
 }
 const low=leg.status==='propagating'?leg.finish_s:leg.start_s,high=leg.status==='propagating'?leg.arrival_s:leg.finish_s;
 if(!Number.isFinite(low)||!Number.isFinite(high)||high<=low)return null;
 return Math.max(0,Math.min(1,(time-low)/(high-low)));
}
const api={heldIndex,hermite,pose,progress};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.MotionState=api;
})(typeof window!=='undefined'?window:globalThis);
