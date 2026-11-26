import assert from 'node:assert/strict';
import test from 'node:test';
import {createRecordedFrameLoader,recordedLinksAt} from '../assets/replay_link_states.js';

const event=(time_s,state)=>({time_s,link_id:'isl',state});
const initial={time_s:0,links:[{link_id:'isl',state:'SETTING_UP',ready_s:3}]};
const page={events:[]};
const frames=times=>times.map(t=>({time_s:t,artifact:'frames/'+t}));
const model={trace_id:'orbit-a',artifacts:{frames:frames([0,30,60])}};

test('uses recorded transitions at the exact time in both playback directions',()=>{
  const frame={...initial,events_until_next_boundary:[event(3,'UP'),event(15,'DOWN')]};
  for(const [time,state] of [[2.999,'SETTING_UP'],[3,'UP'],[15,'DOWN'],[7,'UP'],[0,'SETTING_UP']]){
    assert.equal(recordedLinksAt(frame,time,page,-1)[0].state,state);
  }
  assert.equal(initial.links[0].state,'SETTING_UP');
  assert.equal(recordedLinksAt(initial,8,page,-1)[0].state,'SETTING_UP');
});

test('runtime disconnect and reconnect take effect in order across physical events and page boundaries',()=>{
  const frame={...initial,state_events:[event(3,'UP'),event(10,'DOWN')]};
  const disconnect={kind:'topology_command',action:'disconnect',target_id:'isl',time_s:2,event_order:1};
  const connect={...disconnect,action:'connect',time_s:5,event_order:2,link_state:{state:'UP'}};
  const events={commands:[disconnect],events:[connect]};
  assert.equal(recordedLinksAt(frame,3,events,-1)[0].state,'DOWN');
  assert.equal(recordedLinksAt(frame,5,events,0)[0].state,'UP');
  assert.equal(recordedLinksAt(frame,10,events,0)[0].state,'DOWN');
  const at10={time_s:10,links:[event(10,'DOWN')],state_events:[]};
  assert.equal(recordedLinksAt(at10,11,events,0)[0].state,'DOWN');
  const override={kind:'link_state',...event(3,'DOWN'),event_order:7};
  assert.equal(recordedLinksAt(frame,3,{events:[override]},0)[0].state,'DOWN');
});

test('embedded events including an empty interval need no archived source',async()=>{
  for(const key of ['events_until_next_boundary','state_events']){
    let reads=0;
    const frame={...initial,[key]:[]};
    const load=createRecordedFrameLoader('/runs/scene/',model,async url=>{
      assert.equal(url,'/runs/scene/frames/0');reads++;return frame;
    });
    assert.equal(await load(10),frame);
    assert.equal(await load(3),frame);
    assert.equal(reads,1);
  }
});

test('missing legacy events come from the referenced matching trace, including intermediate source frames',async()=>{
  const source='/runs/physical-1/';
  const files=new Map([
    ['/runs/scene/frames/0',initial],['/runs/scene/input',{physical_dataset:'/old/physical-1/network'}],
    [source+'environment-index',{trace_id:'orbit-a',artifacts:{frames:frames([0,20,30,60])}}],
    [source+'frames/0',{state_events:[event(3,'UP')]}],
    [source+'frames/20',{state_events:[event(22,'DOWN')]}],
  ]);
  const read=async url=>{assert.ok(files.has(url));return files.get(url)};
  const recovered=await createRecordedFrameLoader('/runs/scene/',model,read)(2);
  assert.equal(recordedLinksAt(recovered,3,page,-1)[0].state,'UP');
  assert.equal(recordedLinksAt(recovered,22,page,-1)[0].state,'DOWN');
  files.set(source+'environment-index',{trace_id:'another',artifacts:{frames:frames([0,30,60])}});
  assert.equal(await createRecordedFrameLoader('/runs/scene/',model,read)(10),initial);
});
