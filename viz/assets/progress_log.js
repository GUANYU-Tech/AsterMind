const stages=new Map([
 ['Materializing physical nodes','正在生成物理节点'],
 ['Compute formation clearance preflight','正在检查计算卫星编队安全间距'],
 ['SGP4 state precompute','正在预计算卫星轨道状态（SGP4）'],
 ['Visibility, sunlight and candidate geometry','正在计算可见性、日照及候选链路几何'],
 ['Canonical orbit trace export','正在导出标准轨道轨迹数据'],
 ['Persisting complete backend-independent physical attributes','正在保存完整物理属性'],
 ['Dynamic link acquisition and terminal allocation','正在计算动态建链与终端分配'],
 ['Core physical validation and snapshot store','正在校验物理数据并生成快照索引'],
 ['Route probes','正在执行路由探测'],
 ['Physical core dataset ready','核心物理数据已生成'],
 ['Parallel snapshot artifact export','正在并行导出物理场景快照'],
]);

export function formatPhysicalSummary(checks){
 const rows=[checks.passed===true?'物理数据校验通过':checks.passed===false?'物理数据校验未通过':'物理数据校验结果已生成'];
 for(const [key,label,unit] of [['physical_satellites','物理卫星','颗'],['ground_nodes','地面站','个'],['snapshot_count','快照','个']]){
  if(Number.isFinite(checks[key]))rows.push(`${label}：${checks[key]} ${unit}`);
 }
 for(const error of checks.errors||[])rows.push(typeof error==='string'?error:JSON.stringify(error));
 return rows.join('\n');
}

export function formatProgressLine(line,{knownOnly=false}={}){
 let text=String(line);
 // Old workers wrote JSON strings; new workers write plain text. Decode only
 // valid JSON strings so Windows paths and ordinary traceback text stay intact.
 for(let i=0;i<2;i++){
  if(!text.trim().startsWith('"'))break;
  try{const value=JSON.parse(text);if(typeof value!=='string')break;text=value}catch{break}
 }
 const value=text.trim();
 if(value.startsWith('{')){
  try{const result=JSON.parse(value);if(typeof result.checks?.passed==='boolean')return formatPhysicalSummary(result.checks)}catch{}
 }
 if(stages.has(value))return stages.get(value);
 let match=value.match(/^Orbit batch (\d+)\/(\d+): (.+?)–(.+?) s$/);
 if(match)return `轨道预计算：第 ${match[1]}/${match[2]} 批｜仿真时间 ${match[3]}～${match[4]} 秒`;
 match=value.match(/^Physical attributes: (\d+)\/(\d+) links; (\d+) workers?$/);
 if(match)return `物理属性计算：已完成 ${match[1]}/${match[2]} 条链路｜并行进程：${match[3]}`;
 match=value.match(/^Visibility screening: (\d+) structural pairs -> (\d+) horizon candidates$/);
 if(match)return `候选链路筛选：${match[1]} 对结构连接 → ${match[2]} 条全时域候选链路`;
 return knownOnly ? null : text;
}

export function formatProgressLog(text,{discardPartialFirstLine=false,knownOnly=false}={}){
 const lines=String(text||'').split('\n');
 if(discardPartialFirstLine&&lines.length>1)lines.shift();
 return lines.map(line=>formatProgressLine(line,{knownOnly})).filter(line=>line!==null).join('\n');
}

const states = {
 received:'已接收', queued:'排队中', running:'正在运行', stopping:'正在停止', cancelled:'已手动停止',
 paused:'已暂停', completed:'运行完成', success:'运行完成', failed:'运行失败', rejected:'校验失败',
 interrupted:'运行已中断', aborted:'已提前结束', terminated:'已终止', unknown:'状态暂不可用'
};
const phases = {
 queued:'等待执行', preparing_environment:'准备运行环境', configuration:'解析配置', validation:'校验配置',
 bootstrap:'加载执行环境', physics:'准备物理拓扑', network:'建立通信资源与全局时间轴',
 topology:'构建网络', cell:'初始化计算编队', cells:'初始化计算编队', ready:'初始化完成，准备推进仿真',
 serving:'请求调度与计算 / 通信仿真', completed:'运行完成'
};

export function taskPhase(event) {
 const phase=event.phase, stage=event.startup_stage?.phase===phase ? event.startup_stage : event;
 if ((phase==='cell'||phase==='cells') && Number.isInteger(stage.total)) {
   const count=phase==='cell' ? stage.index : stage.completed;
   if (Number.isInteger(count)) return `初始化计算编队（${count}/${stage.total}）`;
 }
 return phases[phase] || (phase ? formatProgressLine(phase) : '');
}

export function taskFailure(job) {
 const reason=String(job.error || job.replay_error || '').trim();
 const raw=String(job.log_tail || '').trim();
 if (!reason && !['failed','rejected'].includes(job.state || job.status) && job.replay_state!=='failed') return '';
 // Older physical workers stored the last 6 KB of stdout/stderr as their error.
 if (!reason || reason.includes('Traceback (most recent call last):') ||
     reason===raw || (reason.includes('\n') && raw.endsWith(reason))) {
   return (reason || raw).split('\n').reverse().find(line=>/^(?:[\w.]+(?:Error|Exception|Interrupt|Exit)|Exception):/.test(line))
     || (raw ? '任务失败，详见原始诊断日志。' : '失败原因未记录。');
 }
 return reason;
}

function eventText(event) {
 const state=states[event.state] || event.state || '';
 const phase=taskPhase(event);
 const parts=[...new Set([state,phase].filter(Boolean))];
 if (Number.isFinite(event.progress)) parts.push(`${event.progress}%`);
 return parts.join(' · ');
}

export function formatTaskLog(job) {
 const history=job.progress_history || [];
 const lines=job.progress_history_truncated ? ['…'] : [];
 for (const {at,event} of history) {
   lines.push(`[${at.slice(11,19)} UTC] ${eventText(event)}`);
 }
 const current={...job,state:job.state || job.status};
 const latest=eventText(current);
 const last=history.at(-1);
 if (latest && (!last || eventText(last.event)!==latest)) lines.push(latest);
 const oldTail=job.log_tail_truncated===undefined && new TextEncoder().encode(job.log_tail||'').length>=12000;
 const physical=formatProgressLog(job.log_tail,{knownOnly:true,discardPartialFirstLine:oldTail});
 // Physical-generation workers also report real counts between status writes.
 if (physical) {
   if (['running','stopping'].includes(current.state)) lines.push(physical);
   else lines.splice(Math.max(0,lines.length-1),0,physical);
 }
 if (job.checks) {
   const summary=formatPhysicalSummary(job.checks);
   if (!physical.includes(summary)) lines.push(summary);
 }
 if (job.simulation_status && job.simulation_status!=='completed') {
   lines.push(`仿真结果：${states[job.simulation_status] || job.simulation_status}`);
 }
 const failure=taskFailure(job);
 if (failure) lines.push(failure);
 return lines.join('\n');
}
