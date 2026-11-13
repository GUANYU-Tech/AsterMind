"""Exercise the original two-file ISLSim core, not the external adapter.

The external adapter still requires the original validator.py. This probe never
supplies a substitute validator or claims adapter integration coverage.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    identity = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('config.py', 'link_model.py')}
    # This process is dedicated to the external core; do not contaminate a
    # Simulation process with its unqualified `config` import.
    sys.path.insert(0, str(root))
    from config import FSOConfig
    from link_model import compute_link, C
    assert Path(sys.modules['config'].__file__).resolve() == root/'config.py'
    assert Path(sys.modules['link_model'].__file__).resolve() == root/'link_model.py'
    config = FSOConfig()
    rows = []
    for distance in (100., 500., 1000., 2000., 4000., 8000.):
        result = compute_link(config, distance)
        assert math.isclose(result.latency_ms, distance*1e6/C, rel_tol=1e-12)
        assert math.isfinite(result.snr_db) and result.total_bw_gbps >= 0
        assert result.total_bw_gbps <= config.terminal_cap_gbps
        rows.append(dict(distance_km=distance, result=asdict(result)))
    powers = [r['result']['received_power_dbm'] for r in rows]
    assert all(a > b for a, b in zip(powers, powers[1:]))
    weaker = deepcopy(config)
    weaker.tx_power_w /= 5
    a, b = compute_link(config, 1000.), compute_link(weaker, 1000.)
    assert a.received_power_dbm > b.received_power_dbm and a.snr_db > b.snr_db
    fixed = deepcopy(config)
    fixed.modulation_pool = [(4, 'DP-QPSK')]
    fixed.terminal_cap_gbps = None
    result = compute_link(fixed, 1000.)
    assert result.modulation_order == 4
    payload = dict(scope='original_core_only', source_sha256=identity,
        results=rows, weaker_power=asdict(b), fixed_qpsk=asdict(result),
        external_adapter=dict(status='unverified_missing_dependency',
            missing=['validator.py'], user_accepted_scope_exception=True))
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2)+'\n')
    print(json.dumps(dict(core_cases=len(rows)+2, external_adapter=payload['external_adapter'])))


if __name__ == '__main__':
    main()
