import {formatTime} from '/assets/communication_clock.js';
export const eventGroups={
 business:{name:'请求与计算通信',note:'请求到达、输入就绪、计算完成、token 交付及业务通信记录。记录条数不等于请求数。'},
 network:{name:'场景与网络',note:'链路状态、物理拓扑采样、故障和人工操作。包括仿真开始时的链路初始化记录。'},
 diagnostic:{name:'内部诊断',note:'程序回调、控制检查点和内部通知。用于排查程序问题，不代表请求、GPU 算子或网络传输次数。'},
};
export function eventDescription(event){
 if(event.category==='diagnostic')return `${event.kind}，系统执行记录`;
 if(event.kind==='link_state')return `${event.state||'状态更新'}${event.reason?'，'+event.reason:''}`;
 if(event.kind==='surrogate_complete')return `${event.operation||'通信'}，${Number(event.data_bytes||0).toLocaleString()} B，${event.devices?.length||0} 个设备`;
 const detail=event.message||event.reason||event.link_id||(event.source_node_id?`${event.source_node_id} → ${event.target_node_id}`:'')||event.cell_id||event.stage||event.operation||'';
 return detail+(event.record_source==='saved_request_summary'?'，来自已保存的请求时间戳':'');
}
export function renderEventRows(host,events,inspect){
 host.replaceChildren();
 for(const event of events.slice().reverse()){
  const row=document.createElement('tr');
  for(const value of [event.time_ticks!=null?formatTime(event):event.time_s.toFixed(6),event.event_label||event.kind,event.request_id||event.correlation_id||event.target_id||event.link_id||'—',eventDescription(event)]){
   const td=document.createElement('td');td.textContent=value;row.append(td);
  }
  if(inspect){row.tabIndex=0;row.onclick=()=>inspect(event);row.onkeydown=e=>{if(e.key==='Enter')inspect(event)}}
  host.append(row);
 }
 if(!events.length){const row=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent='当前仿真时刻尚无此类记录';row.append(td);host.append(row)}
}
