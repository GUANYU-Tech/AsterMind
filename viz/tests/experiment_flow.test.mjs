import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import {startBatchExperiment, completedDestination} from '../assets/experiment_flow.js';

test('a saved scene launches automatic execution and opens progress before replay', async () => {
  const calls = [];
  await startBatchExperiment(async (...args) => {
    calls.push(args);
    return {id: 'batch-42', url: '/live/batch-42'};
  }, 'saved-scene', url => calls.push(url));
  assert.deepEqual(calls, [
    ['/launch-scene', {id: 'saved-scene'}],
    '/runs/batch-42/progress?next=replay',
  ]);
  assert.equal(completedDestination({id: 'batch-42'}, '?next=replay'),
    '/results?run=batch-42&view=scene');
});

test('invalid scenes and launch errors do not navigate to an empty result', async () => {
  let called = false;
  const unexpected = () => { called = true; };
  await assert.rejects(startBatchExperiment(unexpected, '', unexpected), /完整场景/);
  await assert.rejects(startBatchExperiment(async () => { throw Error('配置未通过'); }, 'scene', unexpected), /配置未通过/);
  assert.equal(called, false);
});

test('physical generation still opens step two, and historical result links remain valid', () => {
  assert.equal(completedDestination({physical_model_id: 'physical-id'}, '?next=scene'), '/scene?model=physical-id');
  assert.equal(completedDestination({physical_model_id: 'old'}, '?next=scene&model=new'), '/scene?model=new');
  assert.equal(completedDestination({native_scene: true, result_url: '/runs/old/result'}, ''), '/runs/old/result');
  assert.equal(completedDestination({preview_url: '/runs/physical/viewer'}, ''), '/runs/physical/viewer');
});

test('configuration launches the selected scene once and recovers its button after an error', async () => {
  const ids = ['models', 'scenes', 'editModel', 'editScene', 'createScene', 'configurePhysical', 'runExperiment', 'status'];
  const elements = Object.fromEntries(ids.map(id => [id, {value: '', addEventListener() {}}]));
  let launchCount = 0, rejectLaunch;
  const context = {
    document: {getElementById: id => elements[id], querySelector: () => ({dataset: {}})},
    get: async () => [], fillOptions() {}, enhanceSelects() {},
    post: () => { launchCount++; return new Promise((_, reject) => { rejectLaunch = reject; }); },
    startBatchExperiment, location: {assign() { assert.fail('must not navigate on failure'); }},
  };
  vm.createContext(context);
  vm.runInContext(readFileSync(new URL('../assets/configuration.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/gm, ''), context);
  await new Promise(resolve => setImmediate(resolve));
  elements.scenes.value = 'selected-scene';
  const pending = elements.runExperiment.onclick();
  assert.equal(elements.runExperiment.disabled, true);
  await elements.runExperiment.onclick();
  assert.equal(launchCount, 1);
  rejectLaunch(Error('已有任务'));
  await pending;
  assert.equal(elements.runExperiment.disabled, false);
  assert.match(elements.status.textContent, /启动失败：已有任务/);
});

test('scene editor saves validated changes before starting batch, with live debugging still optional', async () => {
  const calls = [], buttons = {save: {}, run: {}, liveRun: {}, status: {}};
  const context = {
    $: id => buttons[id], validate: async () => calls.push('validate'), payload: () => ({name: 'changed scene'}),
    post: async (url, value) => {
      calls.push([url, value]);
      return url === '/save-scene' ? {id: 'new-scene'} : {id: 'new-run', url: '/live/new-run'};
    },
    perform: action => action(), startBatchExperiment,
    history: {replaceState() {}}, location: {assign: url => calls.push(url)},
  };
  const source = readFileSync(new URL('../assets/scene_editor.js', import.meta.url), 'utf8');
  const block = source.slice(source.indexOf('async function save(kind)'), source.indexOf("$('generateFaults')", source.indexOf('async function save(kind)')));
  // Only the save function and its three footer handlers belong to this check.
  const end = block.indexOf("$('liveRun').onclick=()=>perform(()=>save('live'));", 0);
  vm.createContext(context);
  vm.runInContext(block.slice(0, end + "$('liveRun').onclick=()=>perform(()=>save('live'));".length), context);
  await buttons.run.onclick();
  assert.deepEqual(JSON.parse(JSON.stringify(calls)), ['validate', ['/save-scene', {name: 'changed scene'}], ['/launch-scene', {id: 'new-scene'}], '/runs/new-run/progress?next=replay']);
  calls.length = 0;
  await buttons.liveRun.onclick();
  assert.equal(calls[2][0], '/live/start');
  assert.equal(calls.at(-1), '/live/new-run?autostart=1');
});
