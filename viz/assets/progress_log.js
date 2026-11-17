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

export function formatProgressLine(line){
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
 return text;
}

export function formatProgressLog(text,{discardPartialFirstLine=false}={}){
 const lines=String(text||'').split('\n');
 if(discardPartialFirstLine&&lines.length>1)lines.shift();
 return lines.map(formatProgressLine).join('\n');
}
