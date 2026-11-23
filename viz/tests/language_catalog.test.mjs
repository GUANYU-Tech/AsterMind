import assert from 'node:assert/strict';
import test from 'node:test';
import {translate} from '../assets/language-en.js';
import {patterns} from '../assets/language-en-catalog.js';

test('localizes reviewed labels and formats while preserving measured values', () => {
  assert.equal(translate('随机故障'), 'Random Faults');
  assert.equal(translate('指定时间故障'), 'Scheduled Faults');
  assert.equal(translate('GPU 端口传播时延（微秒）'), 'GPU Port Propagation Delay (µs)');
  assert.equal(translate('端到端延迟 1：603.068 ms'), 'End-to-End Latency 1: 603.068 ms');
  assert.equal(translate('选择链路 卫星 002 到 卫星 003'), 'Select Link from Satellite 002 to Satellite 003');
  assert.equal(translate('打开任务'), 'Open Task');
  assert.equal(translate('  查看结果  '), '  View Results  ');
  assert.equal(translate('configs/model/qwen3-4b.json'), 'configs/model/qwen3-4b.json');
});

test('every locale pattern keeps its value placeholders', () => {
  const slots = text => [...new Set(text.match(/\{\d+\}/g))].sort();
  for (const [source, english] of patterns) assert.deepEqual(slots(source), slots(english), source);
});
