import unittest
from dataclasses import replace
import numpy as np
from astermind.builder.orbit.sgp4.initialization import walker_delta_elements
from astermind.builder.orbit.sgp4.elements import load_sgp4_element_bundle
from astermind.builder.orbit.sgp4.propagation import propagate_sgp4_teme

class FixedTests(unittest.TestCase):
    def test_square_day_velocity_and_partition_independence(self):
        epoch='2026-07-20T00:00:00.000000Z'
        bundle=load_sgp4_element_bundle(walker_delta_elements(name='ref',altitude_km=550.,planes=1,
            satellites_per_plane=1,epoch_utc=epoch,inclination_deg=53.))
        reference=bundle.satellites[0]
        members=tuple(replace(reference,satellite_id=f'm{i}',satnum=i+1,
            fixed_lvlh_offset_km=(0.,i%2-.5,i//2-.5)) for i in range(4))
        bundle=replace(bundle,satellites=members)
        self.assertEqual(load_sgp4_element_bundle(bundle.to_dict()),bundle)
        ts=[0.,1000.,1000.1,86400.]
        result=propagate_sgp4_teme(bundle,simulation_epoch_utc=epoch,sample_times_s=ts)
        p=np.array([s.position_teme_km for s in result.samples]).reshape(4,4,3)
        for a,b in ((0,1),(1,3),(3,2),(2,0)):
            np.testing.assert_allclose(np.linalg.norm(p[:,a]-p[:,b],axis=1),1.,atol=1e-9)
        finite=(p[2]-p[1])/.1
        velocity=np.array([s.velocity_teme_km_s for s in result.samples[4:8]])
        np.testing.assert_allclose(finite,velocity,atol=.001)
        for member in members:
            single=propagate_sgp4_teme(replace(bundle,satellites=(member,)),simulation_epoch_utc=epoch,sample_times_s=ts)
            self.assertEqual(single.samples,tuple(s for s in result.samples if s.satellite_id==member.satellite_id))

if __name__=='__main__':unittest.main()

def test_reference_identity_and_rotating_velocity_both_orbits():
    for inclination in (53., 97.5930238001833):
        epoch='2026-07-20T00:00:00.000000Z'
        bundle=load_sgp4_element_bundle(walker_delta_elements(name='reference',altitude_km=550.,planes=1,
            satellites_per_plane=1,epoch_utc=epoch,inclination_deg=inclination))
        reference=bundle.satellites[0]
        baseline=propagate_sgp4_teme(bundle,simulation_epoch_utc=epoch,sample_times_s=[43200.])
        zero=replace(reference,fixed_lvlh_offset_km=(0.,0.,0.))
        actual=propagate_sgp4_teme(replace(bundle,satellites=(zero,)),simulation_epoch_utc=epoch,sample_times_s=[43200.])
        assert baseline.samples == actual.samples
        offset=replace(reference,fixed_lvlh_offset_km=(.3,1.,.7))
        samples=propagate_sgp4_teme(replace(bundle,satellites=(offset,)),simulation_epoch_utc=epoch,
            sample_times_s=[43199.95,43200.,43200.05]).samples
        numerical=(np.array(samples[2].position_teme_km)-np.array(samples[0].position_teme_km))/.1
        np.testing.assert_allclose(numerical,samples[1].velocity_teme_km_s,atol=5e-5)
