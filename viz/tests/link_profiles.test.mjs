import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {ensureInterCellProfile} from '../assets/link_profiles.js';

for (const name of ['starlink_550', 'sharing_demo', 'fault_recovery']) {
  test(`${name}: enabling inter-cell links supplies a separate complete budget`, () => {
    const {network} = JSON.parse(readFileSync(new URL(`../templates/${name}.json`, import.meta.url)));
    const original = structuredClone(network);
    assert.equal(ensureInterCellProfile(network), false);
    assert.deepEqual(network, original);
    network.inter_cell_links = true;
    assert.equal(ensureInterCellProfile(network), true);
    assert.deepEqual(network.link_profiles.inter_cell, original.link_profiles.cell_isl);
    network.link_profiles.inter_cell.max_distance_km = 1234;
    network.link_profiles.inter_cell.assumptions.push('Independent inter-cell budget');
    assert.deepEqual(network.link_profiles.cell_isl, original.link_profiles.cell_isl);
    assert.equal(ensureInterCellProfile(network), false);
    network.inter_cell_links = false;
    ensureInterCellProfile(network);
    network.inter_cell_links = true;
    ensureInterCellProfile(network);
    assert.equal(network.link_profiles.inter_cell.max_distance_km, 1234);
  });
}

test('existing inter-cell parameters survive loading and re-enabling', () => {
  const profile = {capacity_model:'fixed', bandwidth_bps:123, setup_time_s:7, max_distance_km:456};
  const network = {inter_cell_links:true, link_profiles:{inter_cell:profile}};
  assert.equal(ensureInterCellProfile(network), false);
  assert.equal(network.link_profiles.inter_cell, profile);
});

test('missing compute budget does not fabricate a usable link profile', () => {
  for (const link_profiles of [undefined, {}, {cell_isl:null}, {cell_isl:[]}]) {
    const network = {inter_cell_links:true, link_profiles};
    const original = structuredClone(network);
    assert.equal(ensureInterCellProfile(network), false);
    assert.deepEqual(network, original);
  }
});
