import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {linkMode, setLinkMode, rememberEmptyLinkMode, newInterSatelliteLink, applyOpticalDefaults} from '../cell/link-mode.mjs';
import {syncFormationLinks, editorChoices} from '../cell/config-edit.mjs';
import {visibleField} from '../cell/config-layout.mjs';
const presentation = JSON.parse(readFileSync(new URL('../cell/presentation.json', import.meta.url)));
const fixture = () => structuredClone(presentation.default_configuration);

test('one choice applies to all links while preserving their endpoints and configured rates', () => {
  const config = fixture();
  config.topology.inter_satellite.links = [
    {source:'a',target:'b',mode:'fixed',bandwidth_gbps:123},
    {source:'b',target:'c',mode:'islsim',bandwidth_gbps:null},
  ];
  assert.equal(linkMode(config),null);
  setLinkMode(config,'fixed');
  assert.equal(linkMode(config),'fixed');
  assert.deepEqual(config.topology.inter_satellite.links.map(link => link.bandwidth_gbps),[123,400]);
  setLinkMode(config,'islsim');
  assert.ok(config.topology.inter_satellite.links.every(link => link.mode === 'islsim'));
  assert.deepEqual(config.topology.inter_satellite.links.map(link => [link.source,link.target,link.bandwidth_gbps]),[['a','b',123],['b','c',400]]);
});

test('last selection survives removing every link and is inherited by canvas-created links', () => {
  const config = fixture(); setLinkMode(config,'fixed');
  config.topology.inter_satellite.links = [];
  assert.equal(newInterSatelliteLink(config,'a','b').mode,'fixed');
  setLinkMode(config,'islsim');
  assert.deepEqual(newInterSatelliteLink(config,'a','b'),{source:'a',target:'b',mode:'islsim',bandwidth_gbps:null});
});

test('loaded uniform mode and saved empty-list preference are retained', () => {
  const config = fixture(); setLinkMode(config,'fixed');
  const loaded = structuredClone(config); assert.equal(linkMode(loaded),'fixed');
  loaded.topology.inter_satellite.links = []; assert.equal(linkMode(loaded),'fixed');
  const empty = fixture(); empty.topology.inter_satellite.links = [];
  rememberEmptyLinkMode(empty,'fixed'); assert.equal(linkMode(empty),'fixed');
});

test('formation generation and form add-link action inherit the global mode', () => {
  const config = fixture(); setLinkMode(config,'fixed'); config.topology.inter_satellite.links = [];
  config.hardware.satellite_ids = ['a','b','c']; config.constellation.formation.layout = 'ring';
  syncFormationLinks(config);
  assert.equal(config.topology.inter_satellite.links.length,3);
  assert.ok(config.topology.inter_satellite.links.every(link => link.mode === 'fixed'));
  setLinkMode(config,'islsim');
  const choices = editorChoices(config,{default_configuration:presentation.default_configuration,schema:presentation.schema});
  assert.equal(choices.seed(['topology','inter_satellite','links',3],{}).mode,'islsim');
});

test('mixed imported modes remain explicit until the user chooses; invalid choices fail atomically', () => {
  const config = fixture();
  config.topology.inter_satellite.links = [{mode:'fixed'}, {mode:'islsim'}];
  const before = structuredClone(config);
  assert.equal(linkMode(config),null);
  assert.throws(() => newInterSatelliteLink(config,'a','b'),/统一/);
  assert.throws(() => setLinkMode(config,'fake'),/有效/);
  assert.deepEqual(config,before);
});

test('ISLSim keeps imported optical parameters and defaults to automatic baud selection', () => {
  const config = fixture();
  Object.assign(config.fso.tx,{beam_waist_m:.07,baud_rate_options_ghz:[20,40,50]}); delete config.fso.tx.baud_rate_ghz;
  Object.assign(config.fso.channel,{wdm_loss_db:6,pointing_error_rad:.001}); delete config.fso.rx.preamp_gain_db;
  config.fso.tx.tx_power_w = 7;
  applyOpticalDefaults(config,presentation.default_configuration.fso);
  assert.equal(config.fso.tx.baud_rate_ghz,null);
  assert.deepEqual(config.fso.tx.baud_rate_options_ghz,[20,40,50]);
  assert.equal(config.fso.tx.beam_waist_m,.07);
  assert.equal(config.fso.channel.pointing_error_rad,.001);
  assert.equal(config.fso.tx.tx_power_w,7);
  assert.equal(config.fso.rx.preamp_gain_db,40);
  for (const [section,definition] of [['tx','FSOTxConfig'],['rx','FSORxConfig'],['channel','FSOChannelConfig']]) {
    assert.ok(Object.keys(config.fso[section]).every(key => Object.hasOwn(presentation.schema.$defs[definition].properties,key)));
  }
  config.fso.tx.baud_rate_ghz=40;
  applyOpticalDefaults(config,presentation.default_configuration.fso);
  assert.equal(config.fso.tx.baud_rate_ghz,40);
  assert.equal(visibleField(['fso','tx','baud_rate_options_ghz']),true);
  assert.equal(visibleField(['fso','tx','beam_waist_m']),false);
});
