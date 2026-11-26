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

test('translates the shared L2/L3 configuration labels and dynamic summaries', () => {
  const expected = {
    'Prefill 分块上限': 'Prefill Chunk Limit',
    'KV 缓存（高级）': 'KV Cache Config',
    '4 颗卫星': '4 satellites',
    '8 卡/星': '8 GPUs/satellite',
    '自定义相对位置': 'Custom Relative Positions',
    '星座模板': 'Constellation Template',
    '计算池 1': 'Worker Pool 1',
    '候选模型 15 个': '15 candidate models',
    '编辑算力编队 dawn_dusk-p000-s0000': 'Edit Compute Cell dawn_dusk-p000-s0000',
    '计算星—通信星 有效可用带宽比例': 'Compute–Communication Satellite Effective Bandwidth Ratio',
    '上行 0.4 → 0.36 Gbps': 'Uplink 0.4 → 0.36 Gbps',
    'Nearest（逐轮最近距离）': 'Nearest (Per Turn)',
    'V1 路径 / #16/p2p/16': 'V1 Path / #16/p2p/16',
    '传输 #16/p2p/16 / V1 整条路径共享资源 / 路径传输中': 'Transfer #16/p2p/16 / V1 Whole-Path Resources / In Transit',
    '18 条活跃路径段': '18 active path segments',
  };
  for (const [source, english] of Object.entries(expected)) assert.equal(translate(source), english, source);
});
