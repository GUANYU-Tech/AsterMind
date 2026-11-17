import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

test('handoff continuous playback stays live across buffers; only explicit history reveals Return',async()=>{
 const elements=new Map(),requests=[];
 const document={getElementById(id){if(!elements.has(id))elements.set(id,{value:'1',textContent:''});return elements.get(id)}};
 let now=0;
 const start={kind:'v1_admit',id:'attempt',request_id:'request',time_s:1,event_order:0,
  time_semantics:'whole_path_shared_resources',source_node_id:'a',target_node_id:'b',
  path_legs:[{transfer_id:'attempt/segment/0',source_node_id:'a',target_node_id:'b'}]};
 const end={kind:'v1_p2p_complete',id:'attempt',time_s:1.2,event_order:1};
 const page={initial:{},events:[start,end]};
 const index={through_order:2,end_s:1.2,busy_intervals:[[1,1.2]],pages:[{file:'page.json',start_s:1,end_s:1.2,count:2}]};
 const context=vm.createContext({document,console,performance:{now:()=>now},setTimeout,clearTimeout,setInterval(){},
  fetch:async url=>{requests.push(url);return {ok:true,json:async()=>url.endsWith('live-communication.json')?index:url.endsWith('live-paths.json')?{schema:'orbitserve.live_paths.v1',through_event_count:2,paths:[]}:page}}});
 context.window=context;
 const read=name=>readFileSync(new URL(`../assets/${name}`,import.meta.url),'utf8');
 vm.runInContext(read('visual_state.js'),context);
 const empty=new vm.SyntheticModule([],()=>{},{context});await empty.link(()=>{});await empty.evaluate();
 const modules=new Map();
 async function load(name){
  if(modules.has(name))return modules.get(name);
  const module=new vm.SourceTextModule(read(name),{context,identifier:name,importModuleDynamically:async()=>empty});
  modules.set(name,module);
  await module.link(spec=>{
   const dependency=spec.split('/').at(-1);
   return ['live_watch.js','communication_clock.js','live_event_cursor.js','live_traffic_model.js','replay_page_loader.js'].includes(dependency)?load(dependency):empty;
  });
  return module;
 }
 const module=await load('live_scene.js');await module.evaluate();const api=module.namespace;
 const nodes=[{node_id:'a'},{node_id:'b'}],frame={time_s:0,nodes,links:[]};
 const state={id:'native-live',communication_mode:'runtime_comm',state:'paused',version:1,time_s:2,end_s:20,
  event_count:2,nodes,links:[],replay:{initial:{},events:[],frames:[frame]}};
 api.updateLiveScene(state,{});await new Promise(resolve=>setImmediate(resolve));
 assert.ok(requests.some(url=>url.endsWith('live-communication.json')),'V1 uses the original continuous-stream entry point');
 assert.equal(api.streamingLive(),true);
 assert.equal(api.hasPendingLiveEvents(),true);
 assert.equal(api.viewingLive(),true,'ordinary buffering never enters history');
 now+=300;await api.advanceLiveWatch(now,1,-1);
 assert.equal(api.viewingLive(),true,'Previous at the beginning does not reveal a false Return control');
 for(let version=2;version<6;version++){
  api.updateLiveScene({...state,version,time_s:version},{});
  now+=300;await api.advanceLiveWatch(now,.1);
  assert.equal(api.viewingLive(),true);
 }
 now+=300;await api.advanceLiveWatch(now,1,-1);
 assert.equal(api.viewingLive(),false,'Previous explicitly enters a detached view');
 api.updateLiveScene({...state,version:6,time_s:6},{});
 assert.equal(api.viewingLive(),false,'polls cannot silently clear a historical view');
 api.showCurrentLiveEvent();assert.equal(api.viewingLive(),true);
 await context.inspectRuntimePath({id:'attempt',time_semantics:'whole_path_shared_resources',start_s:1,
  terminal_s:1.2,terminal:'completed',hops:[],segments:start.path_legs});
 assert.equal(api.viewingLive(),false,'selected whole-path replay is explicitly detached');
 assert.equal(context.networkReplayState().runtime.time_s,1);
 api.showCurrentLiveEvent();assert.equal(api.viewingLive(),true);
});
