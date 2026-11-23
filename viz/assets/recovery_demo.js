(function(root){
'use strict';
let data,stage,buttons=[];
const short=id=>id.replace('starlink-550-','');
function phaseAt(meta,time){
 if(time<meta.fault_s)return '原路径传输';
 if(time<meta.repair_s)return time<meta.input_arrival_s?'故障生效 从缓存节点绕行发送':'绕行输入已到达 等待故障解除';
 if(time<meta.ready_s)return '故障已解除 终端重新建链';
 if(time<meta.restored_route_time_s)return '链路已可用 等待后续传输重新选路';
 return '链路已可用 后续传输重新选择原路径';
}
function initialize(value){
 data=value;const host=document.getElementById('recoveryDemo');if(!host)return;
 host.hidden=false;host.replaceChildren();
 const meta=data.metadata,title=document.createElement('h2');title.style.marginTop='0';
 title.textContent='72×22 大星座 '+(meta.fault_type==='node'?'中继节点故障':'星间链路故障')+' → 绕行 → 原路恢复';host.append(title);
 const info=document.createElement('p');info.textContent='1584 颗通信卫星 物理采样 '+meta.physical_step_s+' s '+meta.fault_s+' s 故障 / 当时换路 '+meta.repair_s+' s 解除 '+meta.ready_s+' s 重新建链完成。故障对象：'+meta.fault_target;info.style.overflowWrap='anywhere';host.append(info);
 const links=document.createElement('div');links.className='controls';
 for(const scenario of data.scenarios){const link=document.createElement('a');link.href='/runs/'+scenario.run_id+'/runtime?view=playback';link.textContent=scenario.kind==='node'?'查看节点故障场景':'查看链路故障场景';links.append(link)}host.append(links);
 const controls=document.createElement('div');controls.className='controls';
 buttons=data.bookmarks.map(bookmark=>{const button=document.createElement('button');button.textContent=bookmark.label+' '+Number(bookmark.time_s.toFixed(3))+' s';button.onclick=async()=>{root.runtimeTimePlayer?.stop();button.disabled=true;try{await root.runtimeContinuousAdapter.at(bookmark.time_s);await root.runtimeGeometryTask;if(bookmark.time_s===meta.fault_s)root.focusRuntimeFault?.('scheduled');else root.focusRuntimeTransmission?.()}catch(error){document.getElementById('status').textContent='定位失败：'+error.message}finally{button.disabled=false}};controls.append(button);return button});host.append(controls);
 stage=document.createElement('p');stage.style.cssText='color:#83edce;font-weight:600';stage.setAttribute('role','status');host.append(stage);
 const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='对比原路、实际改道与恢复后路径';details.append(summary);
 for(const [key,label]of [['original','原定路线'],['detour','故障后改道方案'],['restored','40 s 后续传输重新选路']]){const p=document.createElement('p');p.style.cssText='font:12px ui-monospace,monospace;line-height:1.8;overflow-wrap:anywhere';const route=data.routes[key];p.textContent=label+'：'+[route.path[0].source_node_id,...route.path.map(l=>l.target_node_id)].map(short).join(' → ');details.append(p)}
 const note=document.createElement('p');note.textContent='已完成的 '+meta.preserved_hops+' 跳保留；从 '+short(meta.resume_node)+' 重发当前故障跳。输入在 '+meta.input_arrival_s.toFixed(3)+' s 经绕行到达同一计算 cell。恢复后的新消息实际经过原故障链路；已前进的数据继续前进。两次上传各约 5.56 GiB，用于看清逐跳过程。';details.append(note);host.append(details);
 render(root.currentRuntimeState?.state.time_s||0);
}
function render(time){if(!stage||!data)return;stage.textContent=phaseAt(data.metadata,time)+' 当前 '+time.toFixed(3)+' s';for(let i=0;i<buttons.length;i++)buttons[i].setAttribute('aria-pressed',String(Math.abs(data.bookmarks[i].time_s-time)<1e-5))}
const api={phaseAt,initialize,render};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.LargeRecoveryDemo=api;
})(typeof window!=='undefined'?window:globalThis);
