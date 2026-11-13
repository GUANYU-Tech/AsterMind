"""Exercise retained policy APIs and their existing correctness assertions."""
import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
import sys
from structure_test_support import run_suites, write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from astermind.engine.association import select_ground_links
    from astermind.engine.topology import candidate_links
    from tests.unit.runtime.test_network_policies import AssociationTests, SparseTopologyTests
    policies = ('nearest_distance', 'longest_available', 'highest_elevation', 'highest_capacity')
    rows = []
    fixture = AssociationTests()
    for matching, ground, satellite in product(('greedy', 'deferred_acceptance'), policies, policies):
        state = fixture.state(matching)
        state.spec = replace(state.spec, ground_access_policy=ground, satellite_access_policy=satellite)
        positions = {'u1': (6378., 0., 0.), 'u2': (0., 6378., 0.), 'g': (0., 0., 6357.),
                     's1': (7000., 500., 300.), 's2': (300., 7000., 500.)}
        state.position = lambda node, time: positions[node]
        # Explicit measured attributes are test inputs to the policy, not a
        # replacement for the physical model exercised by the network suites.
        state.directional_attributes = lambda edge, source, target, time: {
            'bandwidth_bps': {'s1': 200., 's2': 400.}[target if target.startswith('s') else source]}
        edges = fixture.edges()
        selected = select_ground_links(state, edges, 0)
        assert selected == select_ground_links(state, list(reversed(edges)), 0)
        assert selected and selected <= {e['link_id'] for e in edges}
        for name, node in state.nodes.items():
            attached = [e for e in edges if e['link_id'] in selected and name in (e['endpoint_a'], e['endpoint_b'])]
            assert len(attached) <= node.sgl_terminals
            if node.role != 'ground':
                for service, quota in [('user', node.max_user_links), ('gateway', node.max_gateway_links)]:
                    assert sum(state.nodes[e['endpoint_a']].ground_kind == service for e in attached) <= quota
        rows.append(dict(matching=matching, ground=ground, satellite=satellite, selected=sorted(selected)))
    topologies = []
    for topology, repair, wrap in product(('grid', 'honeycomb'), ('none', 'ronchetti_intra_orbit'), (False, True)):
        spec = SparseTopologyTests().grid(topology, repair, wrap)
        topologies.append(dict(topology=topology, repair=repair, wrap=wrap, edges=candidate_links(spec)))
    suites = run_suites([
        'tests.unit.runtime.test_network_policies', 'tests.unit.runtime.test_dynamic_network',
        'tests.unit.runtime.test_persistent_routing', 'tests.unit.runtime.test_device_network',
        'tests.unit.runtime.test_request_prerequisites', 'tests.unit.runtime.test_cell_migration',
        'tests.integration.fixed_design.test_network_acceptance',
        'tests.integration.fixed_design.test_continuous_control',
        'tests.integration.fixed_design.test_boundary_control'])
    write_result(args.output, dict(association=rows, topologies=topologies, suites=suites))


if __name__ == '__main__':
    main()
