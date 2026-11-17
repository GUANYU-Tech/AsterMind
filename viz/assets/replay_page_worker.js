import {isTransmissionEvent} from './communication_clock.js';

self.onmessage = async ({data: {id, url, events, cache}}) => {
  try {
    const response = await fetch(url, {cache, signal: AbortSignal.timeout(30000)});
    if (!response.ok) throw Error('回放数据读取失败：HTTP ' + response.status);
    let value = await response.json();
    if (value.encoding === 'gzip-base64') {
      const bytes = Uint8Array.from(atob(value.data), c => c.charCodeAt(0));
      value = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
    }
    if (events && value.events) value.visible = value.events.flatMap((event, i) => isTransmissionEvent(event) ? [i] : []);
    self.postMessage({id, value});
  } catch (error) {
    self.postMessage({id, error: error.message});
  }
};
