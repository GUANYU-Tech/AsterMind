import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import {acceptsLiveState,requestLiveState,livePlaybackState,liveAdvancePlan} from '../assets/live_playback_state.js';
import * as playback from '../assets/live_playback_state.js';
import {formatTime} from '../assets/communication_clock.js';

test('unversioned preparation accepts startup progress and the ready snapshot',()=>{
 const initial={state:'preparing'};
 assert.equal(acceptsLiveState(null,initial),true);
 assert.equal(acceptsLiveState(initial,{state:'preparing',version:1}),true);
 assert.equal(acceptsLiveState(initial,{state:'paused',version:2}),true);
});

test('late polls cannot undo an advance acknowledgement or a newer snapshot',()=>{
 const current={state:'advancing',version:5};
 assert.equal(acceptsLiveState(current,{state:'paused',version:4}),false);
 assert.equal(acceptsLiveState(current,{state:'preparing'}),false);
 assert.equal(acceptsLiveState(current,{unchanged:true,version:5}),false);
 assert.equal(acceptsLiveState(current,{state:'paused',version:6}),true);
});

test('idle live playback seeks the next real event, while active traffic keeps bounded look-ahead',()=>{
 const state={state:'paused',time_s:13.81,end_s:600,request_count:30,completed:14,failed:0};
 assert.deepEqual(liveAdvancePlan(state,{stream:true,activeTransfers:false}),{mode:'event'});
 assert.deepEqual(liveAdvancePlan(state,{stream:true,activeTransfers:true}),{mode:'time',until_s:15.81});
 assert.deepEqual(liveAdvancePlan(state,{stream:false}),{mode:'event'});
});

test('business completion fast-forwards the remaining native queue; manual stepping remains event-based',()=>{
 const state={state:'paused',time_s:18,end_s:600,request_count:30,completed:29,failed:1};
 assert.deepEqual(liveAdvancePlan(state,{stream:true}),{mode:'time',until_s:600});
 assert.deepEqual(liveAdvancePlan(state,{stream:true,singleStep:true}),{mode:'event'});
 assert.deepEqual(liveAdvancePlan({...state,request_count:undefined},{stream:true}),{mode:'event'});
});

// Run the actual page controller with a deterministic DOM/network boundary.
// This exercises polling, viewer intent and advance requests together, without
// starting or stopping the user's live simulation.
function pageHarness({autostart=true}={}){
 class Element {
  constructor(){this.value='';this.disabled=false;this.hidden=false;this.className='';this.dataset={};this.children=[];this.clientWidth=600;this.writes=0;this._text='';this.classList={add(){},remove(){},toggle(){}}}
  get textContent(){return this._text}
  set textContent(value){this.writes++;this._text=String(value)}
  append(...nodes){this.children.push(...nodes)}
  after(){}
  setAttribute(){}
  replaceChildren(...nodes){this.children=nodes}
 }
 const elements=new Map(),element=id=>{
  if(!elements.has(id))elements.set(id,new Element());
  return elements.get(id);
 };
 element('eventRate').value='1';element('eventFilter').value='business';
 const timers=[],requests=[],advances=[],playbackUpdates=[];
 let nextFailure=null,postHandler=null,pendingEvents=false,shownEvents=0,remote={id:'test-session',state:'preparing',time_s:0,message:'初始化仿真环境',preparation_started_at:1700000000};
 const main=new Element(),heading=new Element();
 const scene={get:async url=>{
  requests.push(url);
  if(url.endsWith('/input'))return {};
  if(nextFailure){const error=nextFailure;nextFailure=null;throw error}
  return structuredClone(remote);
 }};
 const document={getElementById:element,createElement:()=>new Element(),createElementNS:()=>new Element(),createDocumentFragment:()=>new Element(),querySelector:selector=>selector==='main'?main:heading,documentElement:new Element()};
 const context={document,URLSearchParams,location:{pathname:'/live/test-session',search:autostart?'?autostart=1':''},history:{replaceState(){}},Date:class extends Date{static now(){return 1700000000000}},performance:{now:()=>0},window:{addEventListener(){}},
  ...playback,formatTime,get:url=>scene.get(url),requestLiveState:url=>scene.get(url),post:async(url,payload)=>{
   advances.push({url,payload});return postHandler?postHandler(url,payload):{state:'advancing',version:(remote.version??0)+1};
  },createCellTopology:()=>({clear(){},update(){}}),
  eventGroups:{business:{note:''}},eventDescription:()=>'',renderEventRows(){},
  RuntimeVisual:{stateAt:()=>({activeTransfers:[]})},FaultVisual:{collect:()=>[]},
  updateLiveScene(){},inspectLiveEvent(){},viewingLive:()=>true,hasPendingLiveEvents:()=>pendingEvents,nextLiveEvent(){pendingEvents=false;shownEvents++},showCurrentLiveEvent(){},
  setLivePlayback:(...args)=>playbackUpdates.push(args),setLivePhysicalSource(){},advanceLiveWatch:async()=>false,streamingLive:()=>false,bufferedWatchSeconds:()=>0,watchedThrough:()=>0,executedThroughTime:()=>0,latestCommunicationTime:()=>null,watchDidMove:()=>false,watchingSelectedPath:()=>false,selectedPathComplete:()=>false,liveWatchError:()=>null,resumeLiveWatch(){},
  setTimeout:callback=>{timers.push(callback);return timers.length},
 };
 const source=readFileSync(new URL('../assets/live_runtime.js',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'');
 vm.runInNewContext(source,context,{filename:'live_runtime.js'});
 const settle=()=>new Promise(resolve=>setImmediate(resolve));
 return {element,requests,advances,playbackUpdates,settle,
  state(value){remote={...remote,...value}},
  failNextPoll(error){nextFailure=error},
  pending(value){pendingEvents=value},
  get shownEvents(){return shownEvents},
  postWith(handler){postHandler=handler},
  async tick(name){await settle();const at=timers.findIndex(callback=>callback.name===name);assert.notEqual(at,-1,`${name} scheduled`);const [callback]=timers.splice(at,1);await callback();await settle()},
 };
}

function readyState(){
 return {state:'paused',version:2,time_s:0,end_s:20,step_s:5,nodes:[],links:[],completed:0,failed:0,request_count:1,topology_revision:0,event_groups:{counts:{business:0},recent:{business:[],network:[]}},replay:{initial:{},events:[],frames:[]}};
}

test('preparing -> ready automatically advances on the same page without reload',async()=>{
 const page=pageHarness();await page.settle();
 assert.equal(page.element('sessionState').textContent,'正在准备场景');
 page.state(readyState());await page.tick('refresh');await page.tick('playbackTick');
 assert.ok(page.advances.some(request=>request.url==='/live/test-session/advance'));
 assert.equal(page.element('play').textContent,'暂停观看');
 assert.equal(page.element('play').disabled,false);
});

test('the page sends idle event-seeking and final queue drain, and reveals results only after saving',async()=>{
 const page=pageHarness();await page.settle();
 page.state({...readyState(),request_count:30,completed:14,end_s:600});
 await page.tick('refresh');await page.tick('playbackTick');
 assert.equal(page.advances.at(-1).payload.mode,'event');
 page.state({...readyState(),version:4,request_count:30,completed:30,end_s:600});
 await page.tick('refresh');await page.tick('playbackTick');
 assert.equal(page.advances.at(-1).payload.until_s,600);
 assert.equal(page.element('viewResults').hidden,true);
 page.state({...readyState(),state:'completed',version:6,time_s:600,end_s:600,request_count:30,completed:30});
 await page.tick('refresh');
 assert.equal(page.element('viewResults').hidden,false);
 assert.equal(page.element('viewResults').href,'/results?run=test-session');
});

test('polling and playback ticks keep one preparation notice and stable button text',async()=>{
 const page=pageHarness();await page.settle();
 const notice=page.element('notice'),button=page.element('play'),badge=page.element('sessionState');
 const expected=notice.textContent,writes={notice:notice.writes,button:button.writes,badge:badge.writes};
 for(let i=0;i<4;i++){await page.tick('refresh');assert.equal(notice.textContent,expected);await page.tick('playbackTick');assert.equal(notice.textContent,expected)}
 assert.equal(notice.writes,writes.notice);
 assert.equal(button.writes,writes.button);
 assert.equal(badge.writes,writes.badge);
 assert.match(expected,/准备完成后自动播放/);
});

test('preparation has one notice; traffic statistics appear only when the scene is ready',async()=>{
 const page=pageHarness();await page.settle();
 assert.equal(page.element('runningTraffic').hidden,true);
 assert.equal(page.element('trafficState').textContent,'');
 assert.match(page.element('notice').textContent,/正在准备场景/);
 page.state(readyState());await page.tick('refresh');
 assert.equal(page.element('runningTraffic').hidden,false);
});

test('cancelling autoplay during preparation remains paused when ready',async()=>{
 const page=pageHarness();await page.settle();
 page.element('play').onclick();await page.tick('playbackTick');
 assert.match(page.element('notice').textContent,/准备完成后可开始播放/);
 page.state(readyState());await page.tick('refresh');await page.tick('playbackTick');
 assert.equal(page.advances.length,0);
 assert.equal(page.element('play').textContent,'播放传输');
 assert.equal(page.element('play').disabled,false);
 page.element('play').onclick();await page.tick('playbackTick');
 assert.equal(page.advances.length,1);
});

test('opening without autoplay enables manual playback after preparation',async()=>{
 const page=pageHarness({autostart:false});await page.settle();
 assert.equal(page.element('play').textContent,'播放传输');
 assert.equal(page.element('play').disabled,true);
 page.state(readyState());await page.tick('refresh');
 assert.equal(page.element('play').disabled,false);
 page.element('play').onclick();await page.tick('playbackTick');
 assert.equal(page.advances.length,1);
});

test('an empty buffer keeps viewer intent across solver pause and execution',()=>{
 for(const state of [{state:'preparing'},{state:'paused',time_s:0,end_s:10},{state:'advancing',time_s:0,end_s:10}]){
  const view=livePlaybackState(state,{playing:true,pending:false,stream:false,through:0});
  assert.equal(view.drained,false);assert.equal(view.label,'暂停观看');assert.equal(view.canToggle,true);
 }
});

test('a stalled poll times out instead of blocking all future state updates',async()=>{
 const keepalive=setTimeout(()=>{},1000);
 try{
  await assert.rejects(requestLiveState('/state',{timeoutMs:10,fetcher:async(url,{cache,signal})=>{
   assert.equal(cache,'no-store');
   return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(signal.reason),{once:true}));
  }}),{name:'TimeoutError'});
 }finally{clearTimeout(keepalive)}
});

test('poll timeout is visible and playback resumes after recovery without reload',async()=>{
 const page=pageHarness();await page.settle();
 page.failNextPoll(Object.assign(Error('timeout'),{name:'TimeoutError'}));
 await page.tick('refresh');await page.tick('playbackTick');
 assert.match(page.element('notice').textContent,/状态读取超时/);
 assert.equal(page.element('sessionState').textContent,'连接恢复中');
 page.state(readyState());await page.tick('refresh');await page.tick('playbackTick');
 assert.ok(page.advances.length>0);
 assert.equal(page.element('sessionState').textContent,'在线运行中');
 assert.doesNotMatch(page.element('notice').textContent,/超时|后台计算中/);
});

test('solver paused/advancing transitions do not flash the playback badge or notice',async()=>{
 const page=pageHarness();await page.settle();
 for(const [version,state] of [[2,'paused'],[4,'advancing'],[6,'paused'],[8,'advancing']]){
  page.state({...readyState(),state,version});await page.tick('refresh');await page.tick('playbackTick');
  assert.equal(page.element('sessionState').textContent,'在线运行中');
  assert.equal(page.element('notice').textContent,'正在播放通信，数据就绪后会自动继续。');
  assert.equal(page.element('advance').disabled,false);
  assert.equal(page.element('returnLive').hidden,true);
 }
});

test('Next during automatic look-ahead queues one step without concurrent advances',async()=>{
 const page=pageHarness();await page.settle();
 let release;
 page.postWith(()=>new Promise(resolve=>release=resolve));
 page.state(readyState());await page.tick('refresh');await page.tick('playbackTick');
 assert.equal(page.advances.length,1);
 assert.equal(page.element('advance').disabled,false,'automatic requests do not flash Next');
 page.element('advance').onclick();page.element('advance').onclick();
 await page.tick('playbackTick');
 assert.equal(page.advances.length,1,'the existing advance owns the backend until acknowledged');
 release({state:'advancing',version:3});await page.settle();
 page.state({...readyState(),version:4});page.pending(true);await page.tick('refresh');await page.tick('playbackTick');
 assert.equal(page.shownEvents,1);
 assert.equal(page.element('advance').disabled,false);
 assert.equal(page.element('play').textContent,'播放传输');
 assert.equal(page.advances.length,1);
});
