(function(root){
'use strict';
const categories={manual:{label:'人为故障',color:'#ff4f70'},random:{label:'随机故障',color:'#ffad36'},scheduled:{label:'定制故障',color:'#ffd16f'},sun:{label:'日凌 / 太阳避让',color:'#b789ff'}};
function collect(snapshot){
 const rows=[];
 for(const node of snapshot.nodes){
  if(node.state!=='DOWN')continue;
  const sources=[...new Set((node.sources?.length?node.sources:['scheduled']).map(s=>categories[s]?s:'scheduled'))];
  rows.push({type:'node',id:node.node_id,sources,object:node});
 }
 for(const edge of snapshot.links){
  if(edge.configuration_disabled)continue;
  if(edge.state!=='DOWN')continue;
  const sources=new Set((edge.sources||[]).map(s=>categories[s]?s:'scheduled'));
  if(edge.manual_disconnected||['manual_disconnected','manual_link_disconnected'].includes(edge.reason))sources.add('manual');
  if(edge.reason==='sun_exclusion')sources.add('sun');
  if(edge.reason==='injected_fault'&&!sources.size)sources.add('scheduled');
  if(sources.size)rows.push({type:'link',id:edge.link_id,sources:[...sources],object:edge});
 }
 return rows;
}
function counts(rows,source){const own=rows.filter(r=>r.sources.includes(source));return {nodes:own.filter(r=>r.type==='node').length,links:own.filter(r=>r.type==='link').length}}
const api={categories,collect,counts};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.FaultVisual=api;
})(typeof window!=='undefined'?window:globalThis);
