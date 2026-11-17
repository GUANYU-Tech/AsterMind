(function(root){
'use strict';
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>{const n=Math.hypot(...a)||1;return a.map(v=>v/n)};
function describe(catalog,snapshot,cellId){
 const latest=new Map(snapshot.nodes.map(n=>[n.node_id,n]));
 const members=catalog.filter(n=>n.cell_id===cellId).map(n=>({...n,...latest.get(n.node_id)}));
 const ids=new Set(members.map(n=>n.node_id));
 const center=[0,1,2].map(i=>members.reduce((sum,n)=>sum+n.position_itrs_km[i],0)/members.length);
 const offsets=members.map(n=>sub(n.position_itrs_km,center));
 const axis=unit(offsets.reduce((a,b)=>Math.hypot(...a)>Math.hypot(...b)?a:b,[1e-12,0,0]));
 let normal=offsets.map(v=>cross(axis,v)).reduce((a,b)=>Math.hypot(...a)>Math.hypot(...b)?a:b,[0,0,0]);
 if(Math.hypot(...normal)<1e-10)normal=cross(axis,unit(center));
 const axis2=unit(cross(unit(normal),axis));
 const projected=offsets.map(v=>[dot(v,axis),dot(v,axis2)]);
 const links=snapshot.links.filter(e=>e.kind==='cell_isl'&&ids.has(e.endpoint_a)&&ids.has(e.endpoint_b));
 const external=snapshot.links.filter(e=>e.state==='UP'&&(ids.has(e.endpoint_a)!==ids.has(e.endpoint_b)));
 let minSeparation=Infinity;
 for(let i=0;i<members.length;i++)for(let j=i+1;j<members.length;j++)minSeparation=Math.min(minSeparation,Math.hypot(...sub(members[i].position_itrs_km,members[j].position_itrs_km))*1000);
 return {cellId,members,center,projected,links,external,minSeparation,time_s:snapshot.time_s};
}
const api={describe};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.CellModel=api;
})(typeof window!=='undefined'?window:globalThis);
