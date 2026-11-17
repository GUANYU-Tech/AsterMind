(function(root){
'use strict';
const element=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node};
const short=id=>id?.replace('compute-mid-','cell-').replace('-m000','')||'未分配';
async function seek(time){root.runtimeTimePlayer?.stop();await root.runtimeContinuousAdapter.at(time);await root.runtimeGeometryTask;root.focusRuntimeTransmission?.()}
async function initialize(){
 const response=await fetch(location.pathname.replace(/runtime$/,'fault-reference'));
 if(!response.ok)return;const data=await response.json();
 if(data.metadata?.mode!=='global_multiflow_manual_faults')return;
 const audit=data.audit,host=document.getElementById('recoveryDemo');if(!host)return;
 host.hidden=false;host.replaceChildren();
 host.append(element('h2','全球六地区 · 多流综合评分与人工故障恢复'));
 host.append(element('p',`600 s / 10 s 物理采样 · 1584 颗通信星 / 4 个 cell · ${audit.request_count} 个请求，完成 ${audit.completed}，失败 ${audit.failed.length} · ${audit.manual_fault_count} 轮人工断链与恢复 · 随机故障关闭`));
 const note=element('p','上传量为演示而放大。四项权重均为 0.25；可行候选中总分最高者获选。回放来自实际日志，下面的评分是请求到达时的决策记录。');host.append(note);
 for(const failure of audit.late_failure_explanations||[]){
  const explanation=element('p',`${failure.request_id} 在 ${failure.time_s.toFixed(6)} s 到达，剩余 ${failure.remaining_horizon_s.toFixed(6)} s；即使按配置的星地容量上限，上传也至少需要 ${failure.minimum_upload_s_at_configured_ceiling.toFixed(6)} s，因此无法在本次 600 s 窗口内完成。`);explanation.style.color='#ffc665';host.append(explanation);
 }
 const controls=element('div');controls.className='controls';
 for(let i=0;i<audit.fault_cases.length;i++){
  const c=audit.fault_cases[i],button=element('button',`故障 ${i+1} · ${c.start_s.toFixed(3)} s`);
  button.onclick=()=>seek(c.start_s+.15).catch(error=>document.getElementById('status').textContent=error.message);controls.append(button);
 }
 host.append(controls);
 const details=element('details'),summary=element('summary','查看每条请求的候选 cell 综合评分与故障审计');details.append(summary);
 const label=element('label','请求评分 '),select=element('select');select.id='globalScoreRequest';select.setAttribute('aria-label','选择评分请求');
 for(const row of data.scores){const option=element('option',`${row.request_id} @ ${row.time_s.toFixed(3)} s → ${short(row.selected_node)}`);option.value=row.request_id;select.append(option)}label.append(select);details.append(label);
 const wrap=element('div');wrap.className='scroll';const table=element('table'),head=element('thead'),tr=element('tr');
 for(const name of ['候选 cell','时延','KV复用','能量','计算负载','加权总分','结果'])tr.append(element('th',name));head.append(tr);table.append(head);const body=element('tbody');table.append(body);wrap.append(table);details.append(wrap);
 const reason=element('p');details.append(reason);
 details.append(element('p','本次网络并发充分，但计算队列没有形成等待，计算负载分数均为 1；高计算拥塞下的负载均衡仍需单独验证。'));
 function renderScores(){
  const row=data.scores.find(r=>r.request_id===select.value);body.replaceChildren();
  for(const c of row.candidates){const tr=element('tr');if(c.selected)tr.className='active';
   for(const value of [short(c.node),c.latency_score.toFixed(6),c.kv_reuse_score.toFixed(6),c.energy_score.toFixed(6),c.compute_load_score.toFixed(6),c.total.toFixed(6),c.selected?'已选择':c.feasible?'可行':'约束未通过'])tr.append(element('td',value));body.append(tr)}
  reason.textContent=`独立核对：${audit.score_audit.evaluated} 次决策，评分或实际计算目标不一致 ${audit.score_audit.mismatches.length} 次；本请求选中 ${short(row.selected_node)}。`;
 }select.onchange=renderScores;renderScores();
 const recovery=element('table'),recoveryHead=element('tr');for(const text of ['注入 / 解除 s','实际中断流数','立即恢复尝试','解除前继续发送','最终完成'])recoveryHead.append(element('th',text));recovery.append(recoveryHead);
 for(const c of audit.fault_cases){const row=element('tr');for(const text of [`${c.start_s.toFixed(3)} / ${c.repair_s.toFixed(3)}`,c.interrupted.length,c.immediate_recoveries.length,c.resumed_before_repair.length,`${c.completed_requests.length}/${c.affected_requests.length}`])row.append(element('td',String(text)));recovery.append(row)}
 const recoveryWrap=element('div');recoveryWrap.className='scroll';recoveryWrap.append(recovery);details.append(recoveryWrap);host.append(details);
}
initialize().catch(error=>console.warn('global multiflow guide:',error));
})(window);
