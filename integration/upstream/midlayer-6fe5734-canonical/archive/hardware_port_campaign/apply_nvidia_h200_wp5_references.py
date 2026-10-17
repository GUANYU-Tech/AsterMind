#!/usr/bin/env python3
"""Apply WP5 (vector / RMSNorm / Add-RMSNorm / SiLU / block reduction) H200 references."""
import hashlib
import re

from nvidia_h200_yaml_edit import REFS, load, ref, save

H = 'microbench/nvidia_h200/references/'


def sha(name):
    return hashlib.sha256((REFS / name).read_bytes()).hexdigest()


def set_rate(text, key, value, rename_from=None):
    """Set a unique service-rate line `<indent><key>: value`, optionally renaming it."""
    old_key = rename_from or key
    pattern = re.compile(r'^(\s+)' + re.escape(old_key) + r':[ \t]*[^\n]*$', re.M)
    hits = pattern.findall(text)
    if len(hits) != 1:
        raise SystemExit(f'{old_key} matched {len(hits)} times')
    return pattern.sub(lambda m: f'{m.group(1)}{key}: {value}', text, count=1)


def set_block_line(text, block, key, value):
    start = text.index(f'\n  {block}:\n') + 1
    end_match = re.search(r'^  \S', text[start + 1:], re.M)
    end = start + 1 + end_match.start() if end_match else len(text)
    body = text[start:end]
    pattern = re.compile(r'^(    ' + re.escape(key) + r':[ \t]*)[^\n#]*?(\s*#[^\n]*)?$', re.M)
    if len(pattern.findall(body)) != 1:
        raise SystemExit(f'{block}.{key} not unique')
    body = pattern.sub(lambda m: f'{m.group(1)}{value}{m.group(2) or ""}', body, count=1)
    return text[:start] + body + text[end:]


def main():
    t = load()
    pipeline = ref('add_rmsnorm_block1024_pipeline_sm90.json')
    renamed_to = {new: old for old, new in pipeline['yaml_key_rename_needed'].items()}
    for key, service in pipeline['selected_services'].items():
        t = set_rate(t, key, service['rate_per_s_at_1_98ghz'], renamed_to.get(key))

    block256 = ref('add_rmsnorm_block256_pipeline_sm90.json')['service_parameters']
    for function in ('add_square', 'scale_multiply'):
        for phase in ('first', 'resident'):
            t = set_rate(t, f'bf16_{function}_{phase}_256_block256',
                         1e9 / block256[f'{function}_{phase}'])
    pair = ref('add_rmsnorm_block256_full_tail_pair_sm90.json')['service_parameters']
    for width in (64, 128):
        for function in ('add_square', 'scale_multiply'):
            for phase in ('first', 'resident'):
                # One full-tail pair retires two source rounds.
                t = set_rate(t, f'bf16_{function}_{phase}_{width}_block256',
                             2e9 / pair[f'{function}_{phase}_{width}'])

    for row in ref('block_reduce_latency_sm90.json')['parameters']:
        threads = row['threads']
        t = set_rate(t, f'block_reduce_first_{threads}', 1e9 / row['first_block_latency_ns'])
        t = set_rate(t, f'block_reduce_resident_{threads}', row['resident_block_rate_per_sm_s'])
        t = set_rate(t, f'block_reduce_queued_{threads}', row['queued_block_rate_per_sm_s'])

    t = set_rate(t, 'silu_scalar_round',
                 ref('silu_scalar_latency_sm90.json')['selected_services']['silu_scalar_round']
                 ['operations_per_second_at_1_98ghz'])
    for key, service in ref('silu_nonrecurrent_segmented_service_sm90.json')['selected_services'].items():
        t = set_rate(t, key, service['operations_per_second_at_1_98ghz'])

    block = 'add_rmsnorm_source_round_pipeline'
    t = set_block_line(t, block, 'software_revision', pipeline['software_revision'])
    for curve, points in (('active_unit_points', '[1, 4, 16, 132]'),
                          ('additional_active_unit_points', '[1, 4, 16, 132]'),
                          ('second_additional_active_unit_points', '[1, 4, 16, 132]'),
                          ('third_additional_active_unit_points', '[1, 4, 8, 16, 132]'),
                          ('fourth_additional_active_unit_points', '[1, 4, 8, 16, 132]')):
        t = set_block_line(t, block, curve, points)
    t = set_block_line(t, block, 'reference_sha256', sha('add_rmsnorm_block1024_pipeline_sm90.json'))
    t = set_block_line(t, block, 'fourth_additional_reference_sha256',
                       sha('add_rmsnorm_block1024_pipeline_sm90.json'))
    block = 'add_rmsnorm_block256_source_round_pipeline'
    t = set_block_line(t, block, 'software_revision',
                       ref('add_rmsnorm_block256_pipeline_sm90.json')['software_revision'])
    t = set_block_line(t, block, 'service_reference_sha256', sha('add_rmsnorm_round_service_sm90.json'))
    t = set_block_line(t, block, 'full_tail_pair_reference_sha256',
                       sha('add_rmsnorm_block256_full_tail_pair_sm90.json'))
    t = set_block_line(t, block, 'reference_sha256', sha('add_rmsnorm_block256_pipeline_sm90.json'))
    save(t)


if __name__ == '__main__':
    main()
