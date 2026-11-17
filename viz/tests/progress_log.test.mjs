import assert from 'node:assert/strict';
import test from 'node:test';
import {formatProgressLine,formatProgressLog,formatPhysicalSummary} from '../assets/progress_log.js';

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
