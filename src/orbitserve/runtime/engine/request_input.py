"""Validate canonical requests before allocating simulation/backend resources."""
import math
from collections.abc import Mapping


def _number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be finite and nonnegative') from None
    if isinstance(value, bool) or not math.isfinite(result) or result < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return result


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def normalize_runtime_request(raw_request):
    if not isinstance(raw_request, Mapping):
        raise ValueError('canonical request must be an object')
    if 'prompt_tokens' not in raw_request or 'output_tokens' not in raw_request:
        raise ValueError('canonical request requires prompt_tokens and output_tokens')
    identity = str(raw_request.get('request_id') or '')
    if not identity:
        raise ValueError('canonical request requires request_id')
    mode = raw_request.get('arrival_time_mode', 'absolute')
    if not isinstance(mode, str) or mode not in {'absolute', 'after_dependency_completion'}:
        raise ValueError('unsupported request arrival_time_mode')
    result = {
        **dict(raw_request), 'request_id': identity, 'arrival_time_mode': mode,
        'prompt_tokens': _integer(raw_request['prompt_tokens'], 'prompt_tokens'),
        'output_tokens': _integer(raw_request['output_tokens'], 'output_tokens', 1),
        'context_tokens_before_turn': _integer(raw_request.get('context_tokens_before_turn', 0),
                                               'context_tokens_before_turn'),
    }
    for name in ('arrival_time_s', 'input_bytes', 'output_bytes'):
        result[name] = _number(raw_request.get(name, 0.), name)
    if mode == 'after_dependency_completion':
        result['inter_turn_gap_s'] = _number(raw_request.get('inter_turn_gap_s', 0.), 'inter_turn_gap_s')
    return result


def prepare_runtime_requests(rows):
    """Validate a dependency forest in linear time without Python recursion."""
    requests = [normalize_runtime_request(row) for row in rows]
    by_id = {row['request_id']: row for row in requests}
    if len(by_id) != len(requests):
        raise ValueError('duplicate request_id')
    visited = set()
    for identity in by_id:
        current, visiting = identity, set()
        while current not in visited:
            if current in visiting:
                raise ValueError('request dependency cycle')
            visiting.add(current)
            request = by_id[current]
            if request['arrival_time_mode'] == 'absolute':
                break
            current = request.get('depends_on_request_id')
            if not isinstance(current, str) or current not in by_id:
                raise ValueError('unknown request predecessor')
        visited.update(visiting)
    return requests
