(function(root){
'use strict';
const $=id=>document.getElementById(id);
const names={queued:'排队等待',transmitting:'正在发送',propagating:'传播中'};
const stages={cell_prefill:'cell Prefill 通信',cell_decode:'cell Decode 通信',input:'请求输入',first_token:'首 token 回传',output:'剩余输出回传',kv_copy:'KV 复制'};
function render(state,record){
 const flows=state.globalMode?state.activeTransfers:state.legs.filter(l=>['queued','transmitting','propagating'].includes(l.status));
 $('currentTraffic').replaceChildren();
 $('trafficHint').textContent=flows.length?'逐事件播放；可使用“下一次开始发送”查看每一跳。':state.completed?'此处请求已完成；点击“跳到传输过程”或“播放事件”重新观看。':'此事件没有在途传输。';
 for(const flow of flows){
  const card=document.createElement('div');card.className='card '+flow.status;
  const title=document.createElement('strong');title.style.fontSize='15px';title.textContent='请求 '+(flow.request_id||record.request.request_id)+' '+(stages[flow.stage]||flow.stage||'')+' '+names[flow.status];card.append(title);
  const path=document.createElement('p');path.style.cssText='color:inherit;overflow-wrap:anywhere';path.textContent=flow.source_node_id+' → '+flow.target_node_id;card.append(path);
  const detail=document.createElement('span');detail.textContent=(flow.payload_bytes==null?'':flow.payload_bytes+' B')+(flow.allocated_bandwidth_bps==null?'':' 最近已记录区间 '+(flow.allocated_bandwidth_bps/1e9).toFixed(6)+' Gbit/s');card.append(detail);
  $('currentTraffic').append(card);
 }
}
function initialize(manifest){
 $('openInteractive').href=manifest.native_preset?'/live?preset=qwen3-4b':'/live?from_run='+encodeURIComponent(location.pathname.split('/')[2]);
 $('rerunMode').value=manifest.transport?.backend?.backend||'flowsim';
 $('rerun').onclick=async()=>{
  const button=$('rerun');button.disabled=true;
  try{
   const base=location.pathname.replace(/runtime$/,'');
   $('rerunStatus').textContent='正在读取原输入并启动新运行…';
   const input=await fetch(base+'input');const payload=await input.json();if(!input.ok)throw Error(payload.error);
   payload.physical_network.network.transport_model=$('rerunMode').value;
   const response=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
   const job=await response.json();if(!response.ok)throw Error(job.error);
   location.href='/runs/'+job.id+'/progress';
  }catch(error){button.disabled=false;$('rerunStatus').textContent='重新运行失败：'+error.message}
 };
}
root.RuntimeControls={render,initialize};
})(window);
