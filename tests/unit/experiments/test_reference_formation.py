import json
from copy import deepcopy
from pathlib import Path

import pytest

from astermind.builder.physical.network_inputs import materialize_network_input
from astermind.engine import NetworkSpec, candidate_links


def manifest():
    p = json.loads((Path(__file__).parents[2] / 'fixtures/network/multi_shell.json').read_text())
    c = p['constellations']['compute_formations'][0]['cluster']
    c.update(motion_model='reference_fixed_lvlh', reference_member=0,
             member_offsets_m=[[0, 0, 0], [0, 1000, 0], [300, 1000, 700], [0, 0, 700]])
    return p


def test_arbitrary_offsets_and_ring_are_separate():
    result = materialize_network_input(manifest())
    members = [s for s in result['elements']['satellites'] if s['satellite_id'].startswith('compute')]
    assert members[0]['fixed_lvlh_offset_km'] == [0, 0, 0]
    assert members[2]['fixed_lvlh_offset_km'] == [.3, 1, .7]
    assert len({s['mean_anomaly_deg'] for s in members[:4]}) == 1
    edges = [e for e in candidate_links(NetworkSpec.from_value(result['network'])) if e['kind']=='cell_isl']
    assert len(edges) == 8
    assert {tuple(sorted((e['endpoint_a'][-3:], e['endpoint_b'][-3:]))) for e in edges} == {
        ('000','001'), ('001','002'), ('002','003'), ('000','003')}


@pytest.mark.parametrize('offsets', [None, [[0,0,0]], [[0,0,0]]*4,
    [[0,0,0],[0,1000,0],[0,float('nan'),0],[0,0,1000]],
    [[1,0,0],[0,1000,0],[0,1000,1000],[0,0,1000]]])
def test_invalid_offsets_rejected(offsets):
    p=manifest(); p['constellations']['compute_formations'][0]['cluster']['member_offsets_m']=offsets
    with pytest.raises(ValueError): materialize_network_input(p)


def test_free_mode_rejects_silently_ignored_offsets():
    p=manifest(); p['constellations']['compute_formations'][0]['cluster']['motion_model']='free_sgp4'
    with pytest.raises(ValueError): materialize_network_input(p)
