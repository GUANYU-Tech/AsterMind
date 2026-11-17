import {isTransmissionEvent} from './communication_clock.js';

let worker, unavailable = false, serial = 0;
const pending = new Map();
function decoder() {
  if (unavailable || typeof Worker === 'undefined') return null;
  if (!worker) {
    try {
      worker = new Worker(new URL('./replay_page_worker.js', import.meta.url), {type: 'module'});
      worker.onmessage = ({data}) => {
        const task = pending.get(data.id);
        if (!task) return;
        pending.delete(data.id);
        data.error ? task.reject(Error(data.error)) : task.resolve(data.value);
      };
      worker.onerror = () => {
        unavailable = true;
        worker.terminate(); worker = null;
        for (const task of pending.values()) task.reject(Error('回放解码线程不可用'));
        pending.clear();
      };
    } catch { unavailable = true; return null; }
  }
  return worker;
}

export async function loadReplayJson(url, {events = false, cache = 'default'} = {}) {
  const instance = decoder();
  if (instance) {
    const id = ++serial;
    try {
      return await new Promise((resolve, reject) => {
        pending.set(id, {resolve, reject});
        instance.postMessage({id, url: new URL(url, location.href).href, events, cache});
      });
    } catch (error) { if (!unavailable) throw error; }
  }
  // Older browsers retain playback and interaction without Worker support.
  const response = await fetch(url, {cache});
  if (!response.ok) throw Error('回放数据读取失败：HTTP ' + response.status);
  let value = await response.json();
  if (value.encoding === 'gzip-base64') {
    const bytes = Uint8Array.from(atob(value.data), c => c.charCodeAt(0));
    value = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
  }
  if (events && value.events) value.visible = value.events.flatMap((event, i) => isTransmissionEvent(event) ? [i] : []);
  return value;
}
