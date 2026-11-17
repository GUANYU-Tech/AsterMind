// Log order identifies an event even when several records share a timestamp.
export function isPlaybackEvent(event){
 return event.category==='business'||['fault_injected','fault_restored','topology_command'].includes(event.kind);
}
export function createLiveEventCursor(){
 let events=[],shown=-1;
 return {
  update(rows){events=rows||[]},
  get pending(){return events.filter(e=>e.event_order>shown&&isPlaybackEvent(e)).length},
  get order(){return shown},
  catchUp(){shown=events.at(-1)?.event_order??shown},
  get index(){let at=-1;for(let i=0;i<events.length;i++){if(events[i].event_order<=shown)at=i;else break}return at},
  next(){let at=events.findIndex(e=>e.event_order>shown&&isPlaybackEvent(e));if(at<0)return null;
   const tick=events[at].time_ticks;if(tick!=null)while(at+1<events.length&&events[at+1].time_ticks===tick)at++;
   shown=events[at].event_order;return at},
 };
}
