import {watchTimeline,communicationPositionAt,compareTime,isTransmissionEvent} from './communication_clock.js';

// The same viewing clock as recorded playback; it only consumes executed data.
export class LiveCommunicationWatch {
 constructor(load){this.load=load;this.pages=new Map();this.index=null;this.time=null;this.exact=null;this.selected=null}
 update(index){if(index.error)this.streamError=index.error;if((this.index?.through_order??0)>(index.through_order??0))return;this.index=index;this.streamError=index.error||null;this.clock=watchTimeline((index.busy_intervals||[]).filter(([start,end])=>end>start),.6);this.refreshSelected()}
 async page(i){const name=this.index.pages[i].file;if(!this.pages.has(name)){this.pages.set(name,await this.load(name));while(this.pages.size>3)this.pages.delete(this.pages.keys().next().value)}const p=this.pages.get(name);p.visible??=p.events.flatMap((event,i)=>isTransmissionEvent(event)?[i]:[]);return p}
 get remaining(){const clock=this.selected||this.clock;if(!clock)return 0;return Math.max(0,clock.end-clock.toWatch(this.time??clock.toSimulation(0)))}
 get pending(){return this.remaining>1e-8}
 select(path,{restart=true}={}){this.selectedPath=path;this.refreshSelected();if(restart){this.time=this.selected?.toSimulation(0)??this.time;this.exact=null}}
 refreshSelected(path=this.selectedPath){
  this.selectedPath=path;
  const intervals=path?.time_semantics==='whole_path_shared_resources'
   ? [[path.start_s,path.terminal_s??Math.max(path.start_s,this.index?.end_s??path.start_s)]]
   : (path?.hops||[]).filter(h=>h.source_node_id!==h.target_node_id&&h.sent_s!=null).map(h=>[h.sent_s,h.arrived_s??h.cancelled_s??Math.max(h.sent_s,this.index?.end_s??h.sent_s)]);
  this.selected=path?watchTimeline(intervals,.35):null;
 }
 async at(target){
  if(!this.index?.pages.length||this.index.pages.every(page=>page.count===0))return null;
  const prior=this.time??0,number=target?.time_s??target;
  const position=await communicationPositionAt(this.index.pages,i=>this.page(i),target),{page,cut,pageIndex}=position;
  const arriving=[];
  for(let i=Math.max(0,this.index.pages.findLastIndex(p=>p.start_s<=prior));i<=pageIndex;i++){
   const data=await this.page(i);for(const row of data.events)if(['hop_arrival','transfer_cancelled'].includes(row.kind)&&row.time_s>prior&&row.time_s<=number)arriving.push(row);
  }
  this.time=number;this.exact=typeof target==='object'?target:null;
  return {initial:page.initial,events:page.events,cut,event:page.events[page.visible[position.visible]]||{kind:'waiting',event_label:'等待通信',time_s:number,event_order:-1},time:number,arriving};
 }
 async advance(elapsed,rate=1){
  const clock=this.selected||this.clock;if(!clock?.segments.length)return null;
  const watch=clock.toWatch(this.time??clock.toSimulation(0));
  if(watch>=clock.end-1e-8)return null;
  let next=Math.min(clock.end,watch+Math.min(.25,elapsed)*rate);
  if(clock.end-next<1e-8)next=clock.end;
  return this.at(clock.toSimulation(next));
 }
 async next(direction=1){
  if(!this.index?.pages.length||this.index.pages.every(page=>page.count===0))return null;
  const target=this.exact??this.time??-Infinity;
  const located=await communicationPositionAt(this.index.pages,i=>this.page(i),target);
  for(let i=located.pageIndex;i>=0&&i<this.index.pages.length;i+=direction){
   const page=await this.page(i),visible=page.visible.map(i=>page.events[i]);const ordered=direction>0?visible:[...visible].reverse();
   const next=ordered.find(e=>direction>0?compareTime(e,target)>0:compareTime(e,target)<0);
   if(next)return this.at(next);
  }
  return null;
 }
}
