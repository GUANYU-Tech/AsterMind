import {createFrameCache} from './runtime_frames.js';

const frameEvents = frame => frame.events_until_next_boundary ?? frame.state_events;

// Older scene exports kept the geometry but omitted the transitions. Read the
// archived physical run named by the scene, after verifying the trace identity.
export function createRecordedFrameLoader(base, model, read) {
  const frames = model.artifacts.frames;
  let physicalSource;
  async function sourceFrames() {
    const input = await read(base+'input');
    const parts = String(input.physical_dataset || '').replaceAll('\\','/').split('/').filter(Boolean);
    if (parts.at(-1)==='network') parts.pop();
    const id = parts.at(-1);
    if (!model.trace_id || !id || !/^[\w-]+$/.test(id)) return null;
    const sourceBase = '/runs/'+id+'/';
    const index = await read(sourceBase+'environment-index');
    if (index.trace_id!==model.trace_id || !index.artifacts?.frames?.length) return null;
    const events = createFrameCache(async descriptor => frameEvents(await read(sourceBase+descriptor.artifact)), 3);
    return {frames:index.artifacts.frames, events};
  }
  const cache = createFrameCache(async descriptor => {
    const frame = await read(base+descriptor.artifact);
    if (Array.isArray(frameEvents(frame))) return frame;
    const start = frame.time_s;
    const end = frames[frames.indexOf(descriptor)+1]?.time_s ?? model.time?.end_s ?? start;
    if (end<=start) return {...frame,state_events:[]};
    if (model.events?.length) {
      return {...frame,state_events:model.events.filter(e=>e.time_s>start && e.time_s<=end)};
    }
    try {
      physicalSource ??= sourceFrames();
      const source = await physicalSource;
      if (source && source.frames[0].time_s<=start && source.frames.at(-1).time_s>=end) {
        const rows = source.frames.filter((row,i)=>row.time_s<end && source.frames[i+1]?.time_s>start);
        const intervals = await Promise.all(rows.map(row=>source.events.get(row)));
        if (intervals.every(Array.isArray)) return {...frame,state_events:intervals.flat()};
      }
    } catch {
      // The scene can outlive its source physical run. Its own frame still works.
      physicalSource = undefined;
    }
    return frame;
  },3);
  return at => cache.get(frames[Math.max(0,frames.findLastIndex(row=>row.time_s<=at))]);
}

export function recordedLinksAt(frame, at, page, cut) {
  const links = new Map(frame.links.map(edge=>[edge.link_id,{...edge}]));
  const updates = (frameEvents(frame) || []).filter(e=>e.time_s>frame.time_s && e.time_s<=at);
  const runtime = [...(page.link_states || []), ...(page.commands || []), ...page.events.slice(0,cut+1)];
  for (const e of runtime) {
    if (e.time_s<=at && (e.kind==='topology_command' || e.kind==='link_state' && e.time_s>=frame.time_s)) updates.push(e);
  }
  // Geometry is the baseline. At equal times, observed runtime changes follow
  // the physical transition, with manual controls applied in execution order.
  updates.sort((a,b)=>a.time_s-b.time_s || (a.event_order ?? -1)-(b.event_order ?? -1));
  for (const e of updates) {
    const command = e.kind==='topology_command';
    const edge = links.get(command ? e.target_id : e.link_id);
    if (!edge) continue;
    if (command) {
      if (e.action==='disconnect') edge.manual_disconnected=true;
      if (e.action==='connect') {
        edge.manual_disconnected=false;
        if (e.time_s>=frame.time_s) Object.assign(edge,e.link_state);
      }
    } else if (e.state) {
      edge.state=e.state;
      edge.reason=e.reason;
      if ('ready_s' in e) edge.ready_s=e.ready_s;
      if ('setup_start_s' in e) edge.setup_start_s=e.setup_start_s;
    }
  }
  for (const edge of links.values()) {
    if (edge.manual_disconnected) Object.assign(edge,{state:'DOWN',reason:'manual_disconnected'});
  }
  return [...links.values()];
}
