"""Measured static-batch tables; internal communication is included exactly once."""
from dataclasses import replace
import hashlib
import json
import math


def from_astermind_measurements(base, document):
    if document.get('schema_version') != 'orbitserve.astermind_measurements.v1':
        raise ValueError('expected orbitserve.astermind_measurements.v1; a single AsterMind total_latency_s is not a Serving profile')
    if document.get('model_name') != base.model_name or document.get('formation') != base.formation or document.get('parallelism') != base.parallelism:
        raise ValueError('AsterMind measurements must match model, formation and parallelism')
    if document.get('hardware') != base.hardware:
        raise ValueError('AsterMind measurements must match resolved hardware')
    if document.get('scope') != 'phase_latency_including_cell_communication':
        raise ValueError('AsterMind measurement scope must include cell communication')
    prefill, decode, identities = [], [], set()
    for row in document.get('samples', []):
        phase, length, concurrency = row.get('phase'), row.get('length'), row.get('concurrency')
        if phase not in {'prefill', 'decode'} or type(length) is not int or length < 1 or type(concurrency) is not int or not 1 <= concurrency <= base.max_concurrency:
            raise ValueError('invalid phase, length or concurrency in AsterMind measurement')
        identity = phase, length, concurrency
        if identity in identities:
            raise ValueError('duplicate AsterMind measurement')
        identities.add(identity)
        result = row.get('result', {})
        latency = result.get('metrics', {}).get('total_latency_s')
        if result.get('status') != 'success' or not result.get('versions', {}).get('astermind') or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError('AsterMind measurement requires successful versioned result and finite positive latency')
        target, key, rate, amount = (prefill, 'prompt_length', 'prefill_tokens_per_s', length) if phase == 'prefill' else (decode, 'context_length', 'decode_tokens_per_s', 1)
        if phase == 'decode' and row.get('decode_tokens') != 1:
            raise ValueError('decode measurement must contain exactly one generated token')
        target.append({key: length, 'concurrency': concurrency, rate: amount / latency})
    for phase in ('prefill', 'decode'):
        for concurrency in range(1, base.max_concurrency + 1):
            if not any(i[0] == phase and i[2] == concurrency for i in identities):
                raise ValueError('AsterMind profile must cover both phases at every configured concurrency')
    fingerprint = hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return replace(base, prefill_rate_table=tuple(prefill), decode_rate_table=tuple(decode), internal_communication_accounting='embedded', provenance={**base.provenance, 'execution_model': 'astermind_measured_static_batch', 'measurement_sha256': fingerprint, 'formation_communication': 'included_in_astermind_phase_latency_no_analytical_addition', 'prefill_model': 'measured_astermind_phase_latency', 'decode_model': 'measured_astermind_single_token_latency', 'rate_semantics': 'per_request'})
