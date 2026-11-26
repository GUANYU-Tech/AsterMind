import assert from 'node:assert/strict';
import test from 'node:test';
import {formatProgressLine,formatProgressLog,formatPhysicalSummary,formatTaskLog,taskFailure} from '../assets/progress_log.js';
import {translate} from '../assets/language-en.js';

test('legacy JSON strings display readable stages and simulation time ranges',()=>{
 const log=String.raw`"Materializing physical nodes"
"Compute formation clearance preflight"
"Orbit batch 1/10: 0\u201361 s"
"Physical attributes: 1720/4429 links; 1 workers"`;
 const display=formatProgressLog(log);
 assert.match(display,/正在生成物理节点/);
 assert.match(display,/正在检查计算卫星编队安全间距/);
 assert.match(display,/第 1\/10 批｜仿真时间 0～61 秒/);
 assert.match(display,/已完成 1720\/4429 条链路｜并行进程：1/);
 assert.doesNotMatch(display,/\\u2013|"/);
});

test('plain new logs display the actual parallel count rather than an invented core count',()=>{
 assert.equal(formatProgressLine('Physical attributes: 240/4429 links; 15 workers'),
  '物理属性计算：已完成 240/4429 条链路｜并行进程：15');
 assert.equal(formatProgressLine('Orbit batch 10/10: 549–600 s'),
  '轨道预计算：第 10/10 批｜仿真时间 549～600 秒');
});

test('unknown errors, Windows paths, markup and summary data remain plain text',()=>{
 for(const line of [String.raw`File "D:\new\trace.py", line 1`,
  'ValueError: invalid input', '<script>alert(1)</script>', '{"diagnostic":{"raw":true}}', '']){
  assert.equal(formatProgressLine(line),line);
 }
 assert.equal(formatProgressLine(JSON.stringify('准备完成')),'准备完成');
 assert.equal(formatProgressLog('first\nsecond\n'),'first\nsecond\n');
});

test('completed physical logs and failed checks remain readable without dumping per-link data',()=>{
 assert.equal(formatProgressLine('{"checks":{"passed":true,"physical_satellites":1604,"ground_nodes":6,"snapshot_count":600}}'),
  '物理数据校验通过\n物理卫星：1604 颗\n地面站：6 个\n快照：600 个');
 assert.match(formatPhysicalSummary({passed:false,errors:['invalid orbit']}),/校验未通过\ninvalid orbit/);
});

test('legacy byte tails discard an incomplete first line without corrupting later progress',()=>{
 assert.equal(formatProgressLog('429 links; 1 workers"\n"Physical attributes: 4408/4429 links; 1 workers"',
  {discardPartialFirstLine:true}),'物理属性计算：已完成 4408/4429 条链路｜并行进程：1');
 assert.equal(formatProgressLine('"Physical core dataset ready"'),'核心物理数据已生成');
});

test('historical tasks show only recorded facts, regardless of host diagnostics',()=>{
 const log_tail="Using 'pin_memory=False' as WSL is detected. This may slow down performance.\nmacOS host diagnostic";
 assert.equal(formatTaskLog({state:'completed',phase:'结果与回放已就绪',log_tail}),'运行完成 · 结果与回放已就绪');
 assert.equal(formatTaskLog({status:'success',phase:'completed',progress:100,log_tail}),'运行完成 · 100%');
 assert.equal(formatTaskLog({state:'cancelled',log_tail}),'已手动停止');
});

test('recorded stages keep their UTC time, actual cell counts and reported percentages',()=>{
 const event={state:'running',phase:'cells',completed:2,total:4,progress:20};
 const job={status:'running',phase:'cells',startup_stage:{phase:'cells',completed:2,total:4},progress:20,
   progress_history:[{at:'2026-11-26T05:02:07+00:00',event}]};
 const text=formatTaskLog(job);
 assert.equal(text,'[05:02:07 UTC] 正在运行 · 初始化计算编队（2/4） · 20%');
 assert.doesNotMatch(translate(text,true),/\p{Script=Han}/u);
});

test('physical stages are ordered relative to the job state and diagnostics are omitted',()=>{
 const log_tail='Materializing physical nodes\nGPU driver initialization diagnostic';
 assert.equal(formatTaskLog({state:'running',log_tail}),'正在运行\n正在生成物理节点');
 assert.equal(formatTaskLog({state:'completed',log_tail}),'正在生成物理节点\n运行完成');
});

test('preserve structured failures and extract only the exception from a legacy diagnostic tail',()=>{
 const log_tail='Host performance warning\nTraceback (most recent call last):\n  File "worker.py"\nValueError: invalid orbit';
 assert.equal(formatTaskLog({state:'failed',error:log_tail,log_tail}),'运行失败\nValueError: invalid orbit');
 assert.equal(taskFailure({status:'rejected',phase:'validation',error:'invalid configuration'}),'invalid configuration');
 assert.equal(taskFailure({state:'completed',replay_state:'failed',replay_error:'invalid replay index'}),'invalid replay index');
 assert.equal(taskFailure({state:'failed',log_tail:'host diagnostic only'}),'任务失败，详见原始诊断日志。');
});
