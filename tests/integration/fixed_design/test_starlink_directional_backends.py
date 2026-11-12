"""A single canonical request traverses asymmetric RF in every real backend."""
from dataclasses import replace
import importlib.util
import unittest

from astermind.engine import LinkProfile
from tests.network_case import run_traffic
from tests.integration.fixed_design.test_network_acceptance import acceptance_spec
from tests.unit.runtime.test_starlink_presets import generated_profiles


class StarlinkDirectionalBackendTests(unittest.TestCase):
    def verify(self, mode):
        base=acceptance_spec()
        generated,_=generated_profiles('starlink_v3_design_160_20261021')
        reference=next(n for n in generated.nodes if n.shell_id=='comm-low')
        nodes=tuple(replace(n,radio_profile=reference.radio_profile,sgl_profiles=reference.sgl_profiles,
                    radio_capacity_bps=reference.radio_capacity_bps,capacity_assumptions=reference.capacity_assumptions)
                    if n.role=='communication' else replace(n,ground_kind='user') if n.role=='ground' else n
                    for n in base.network.nodes)
        profiles={**generated.link_profiles,'sgl':LinkProfile(1e12,.5),
                  'communication_isl':LinkProfile(1e12,.5),'cross_layer':LinkProfile(1e12,.5)}
        scenario=replace(base,network=replace(base.network,nodes=nodes,link_profiles=profiles,faults=(),transport_model=mode),
                         traffic=({**base.traffic[0],'input_bytes':1000000,'output_bytes':1000000,'output_tokens':1},))
        result=run_traffic(scenario)
        self.assertEqual(result.completed,1)
        legs=result.transfer_jobs[0]['path']
        uplink=next(p for p in legs if p['source_node_id']=='ground-00')
        downlink=next(p for p in legs if p['target_node_id']=='ground-00')
        self.assertEqual(uplink['payload_bytes'],downlink['payload_bytes'])
        self.assertGreater(uplink['transmission_s'],2*downlink['transmission_s'])
        for leg in legs:
            self.assertAlmostEqual(sum(s['bytes'] for s in leg['segments']),leg['payload_bytes'],places=5)

    def test_queue_uses_different_upload_and_download_budgets(self): self.verify('queue')
    def test_fluid_uses_different_upload_and_download_budgets(self): self.verify('fluid')

    @unittest.skipUnless(importlib.util.find_spec('simgrid'),'requires native SimGrid')
    def test_simgrid_uses_different_upload_and_download_budgets(self): self.verify('simgrid')

    @unittest.skipUnless(importlib.util.find_spec('ns'),'requires native ns-3')
    def test_ns3_uses_different_upload_and_download_budgets(self): self.verify('ns3')
