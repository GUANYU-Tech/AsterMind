import assert from 'node:assert/strict';
import test from 'node:test';
import {bindFormationSize} from '../assets/formation_size.js';

function editor(count = 10, shape = '5x2', options) {
  const cluster = {satellites_per_cluster:count, grid_shape:shape, formation_pitch_m:500};
  const input = value => ({value:String(value), setCustomValidity(message) { this.validationMessage = message; }});
  const countInput = input(count), shapeInput = input(shape), snapshots = [];
  bindFormationSize(cluster, countInput, shapeInput, () => snapshots.push(JSON.parse(JSON.stringify(cluster))), options);
  return {cluster, countInput, shapeInput, snapshots};
}

test('changing 10 satellites to 4 updates the visible shape and serialized model before submission', () => {
  const form = editor();
  form.countInput.value = '4';
  form.countInput.oninput();
  assert.equal(form.shapeInput.value, '2x2');
  assert.deepEqual(form.snapshots.at(-1), {satellites_per_cluster:4, grid_shape:'2x2', formation_pitch_m:500});
  form.countInput.onchange();
  assert.equal(form.cluster.grid_shape, '2x2');
  assert.equal(form.shapeInput.validationMessage, '');
  assert.equal(form.countInput.min, '1');
  assert.equal(form.countInput.step, '1');
});

test('older mismatched drafts are repaired; valid custom arrangements remain unchanged', () => {
  assert.equal(editor(4, '5x2').shapeInput.value, '2x2');
  assert.equal(editor(4, '4x1').shapeInput.value, '4x1');
  assert.equal(editor(10, '2 × 5').cluster.grid_shape, '2x5');
  for (const [count, shape] of [[1,'1x1'],[6,'3x2'],[7,'7x1'],[12,'4x3']]) {
    const form = editor();
    form.countInput.value = String(count);
    form.countInput.oninput();
    assert.equal(form.cluster.grid_shape, shape);
  }
});

test('manual sizes accept common multiplication symbols and reject incompatible dimensions locally', () => {
  const form = editor(4, '2x2');
  form.shapeInput.value = '2*3';
  form.shapeInput.oninput();
  assert.match(form.shapeInput.validationMessage, /当前为 4 颗/);
  form.shapeInput.value = '4*1';
  form.shapeInput.onchange();
  assert.equal(form.shapeInput.validationMessage, '');
  assert.equal(form.snapshots.at(-1).grid_shape, '4x1');
  assert.equal(form.cluster.satellites_per_cluster, 4);
});

test('temporary invalid counts do not create a zero-size grid; observed TLE membership is not rewritten', () => {
  const form = editor();
  for (const count of ['', '0', '-1', '1.5']) {
    form.countInput.value = count;
    form.countInput.oninput();
    assert.equal(form.shapeInput.value, '5x2');
  }
  const tle = editor(4, '5x2', {synthetic:false});
  assert.equal(tle.cluster.grid_shape, '5x2');
  assert.equal(tle.shapeInput.validationMessage, '');
});
