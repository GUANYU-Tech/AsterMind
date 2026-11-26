import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {editDependencies} from '../cell/config-edit.mjs';
import {optionLabels} from '../cell/config-labels.mjs';
import {visibleField} from '../cell/config-layout.mjs';
import {configurationErrors, fieldError, schemaAt} from '../cell/config-validation.mjs';

const capabilities = JSON.parse(readFileSync(new URL('../cell/presentation.json', import.meta.url)));
const {schema} = capabilities;
const distributionPath = ['requests', 'synthetic', 'arrival_distribution'];
const ratePath = ['requests', 'synthetic', 'request_rate'];
const scalePath = ['requests', 'arrival_rate_scale'];
const workloadErrors = config => configurationErrors(config, schema, 'workload');

test('simultaneous is a valid dropdown selection without an arrival rate', () => {
  const config = structuredClone(capabilities.default_configuration);
  const distribution = schemaAt(schema, distributionPath);
  assert.deepEqual(distribution.enum, ['poisson', 'constant', 'simultaneous']);
  assert.equal(optionLabels.simultaneous, '同时到达');
  config.requests.synthetic.arrival_distribution = 'simultaneous';
  delete config.requests.synthetic.request_rate;
  assert.equal(fieldError('simultaneous', distributionPath, config, distribution), '');
  assert.notEqual(fieldError('invalid', distributionPath, config, distribution), '');
  assert.equal(visibleField(ratePath, config), false);
  assert.equal(visibleField(scalePath, config), false);
  assert.deepEqual(workloadErrors(config), []);
});

test('switching back to spaced arrivals restores the rate field and a missing rate', () => {
  const config = structuredClone(capabilities.default_configuration);
  delete config.requests.synthetic.request_rate;
  config.requests.synthetic.arrival_distribution = 'poisson';
  assert.ok(workloadErrors(config).some(issue => issue.loc.join('.') === ratePath.join('.')));
  assert.equal(editDependencies(config, distributionPath, 'poisson', capabilities), true);
  assert.equal(config.requests.synthetic.request_rate, capabilities.default_configuration.requests.synthetic.request_rate);
  assert.equal(visibleField(ratePath, config), true);
  assert.equal(visibleField(scalePath, config), true);
  assert.deepEqual(workloadErrors(config), []);
});
