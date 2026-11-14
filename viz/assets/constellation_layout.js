(function(root){
'use strict';
const placed=n=>Number.isInteger(n.plane)&&Number.isInteger(n.slot)&&n.plane>=0&&n.slot>=0;
function layout(nodes,shells,shell,start=0){
  const members=nodes.filter(n=>n.role==='communication'&&(n.shell_id||'default')===shell);
  const assigned=members.filter(placed),reference=shells.find(s=>s.shell_id===shell);
  const shape=reference?.grid_shape||[Math.max(0,...assigned.map(n=>n.plane))+1,Math.max(0,...assigned.map(n=>n.slot))+1];
  start=Math.max(0,Math.min(shape[0]-1,Math.trunc(Number(start)||0)));
  return {members,assigned,visible:assigned.filter(n=>n.plane>=start&&n.plane<start+6),shape,start,
    unplaced:members.length-assigned.length,empty:shape[0]*shape[1]-assigned.length,
    slotScale:Math.max(1,shape[1]-1),planeScale:Math.max(1,Math.min(5,shape[0]-1-start))};
}
function orbitGroups(nodes){
  const groups=new Map();
  for(const node of nodes.filter(n=>n.role==='communication'&&placed(n))){
    const key=(node.shell_id||'default')+':'+node.plane;
    if(!groups.has(key))groups.set(key,[]);groups.get(key).push(node);
  }
  return [...groups.values()].filter(g=>g.length>=3).map(g=>g.sort((a,b)=>a.slot-b.slot));
}
const api={layout,orbitGroups};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.ConstellationLayout=api;
})(typeof window!=='undefined'?window:globalThis);
