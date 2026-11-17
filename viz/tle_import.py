"""Offline sample catalog and TLE inspection for the personal console."""
import json
from pathlib import Path
import re

from astermind.builder.physical.tle_inspection import inspect_tle_constellation


def sample_paths(root, identity):
    if not re.fullmatch(r'[a-z0-9-]+', identity):
        raise ValueError('invalid TLE sample ID')
    directory = Path(root)/'local/inputs/tle'
    return directory/(identity+'.summary.json'),directory/(identity+'.network.json'),directory/(identity+'.tle')


def samples(root):
    result=[]
    for path in sorted((Path(root)/'local/inputs/tle').glob('*.summary.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        _,manifest,tle=sample_paths(root,value['id'])
        if manifest.is_file() and tle.is_file():
            result.append({k:value[k] for k in ('id','title','selected_count','nominal_reference','historical_count_difference',
                'grid_placed_count','grid_unplaced_count','nominal_grid_empty_positions','inferred_grid_shape')})
    return result


def sample(root, identity):
    summary,manifest,_=sample_paths(root,identity)
    return dict(summary=json.loads(summary.read_text(encoding='utf-8')),
                physical_network=json.loads(manifest.read_text(encoding='utf-8')))


def inspect(text):
    if not isinstance(text,str) or len(text.encode('utf-8'))>1_500_000:
        raise ValueError('TLE text is missing or too large; select one shell')
    return inspect_tle_constellation(text)
