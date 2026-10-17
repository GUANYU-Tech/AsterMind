#!/usr/bin/env python3
"""Apply WP7 (launch / runtime / task-floor) H200 references to the H200 yaml.

Also restores the three scheduling blocks the A800 config carries but the
H200 copy dropped (cuda_graph_small_node_span, cta_dispatch_cost,
runtime_task_latency_models), in the A800 layout with H200 data.
"""
import hashlib
import re

from nvidia_h200_yaml_edit import REFS, load, ref, save

H = 'microbench/nvidia_h200/references/'


def sha(name):
    return hashlib.sha256((REFS / name).read_bytes()).hexdigest()


def set_in_block(text, block, key, value, indent=4):
    start = text.index(f'\n  {block}:\n') + 1
    end_match = re.search(r'^  \S', text[start + 1:], re.M)
    end = start + 1 + end_match.start() if end_match else len(text)
    body = text[start:end]
    pattern = re.compile(r'^(' + ' ' * indent + re.escape(key) + r':[ \t]*)[^\n]*$', re.M)
    if len(pattern.findall(body)) != 1:
        raise SystemExit(f'{block}.{key} not unique')
    body = pattern.sub(lambda m: f'{m.group(1)}{value}', body, count=1)
    return text[:start] + body + text[end:]


def main():
    t = load()
    launch = ref('isolated_empty_task_scheduler_sm90.json')['parameters']
    for key in ('submit_ns', 'cta_round_ns', 'wave_round_ns', 'extra_grid_dimension_ns', 'extra_block_dimension_ns'):
        t = set_in_block(t, 'launch_latency_model', key, launch[key])

    memdep = ref('memory_dependency_task_sm90.json')
    for key in ('base_ns', 'fanout_activation_ns', 'cta_ns', 'warp_ns'):
        t = set_in_block(t, 'memory_dependency_task_latency_model', key, memdep['parameters'][key])
    t = set_in_block(t, 'memory_dependency_task_latency_model', 'maximum_grid_size', memdep['support']['maximum_grid'])
    t = set_in_block(t, 'memory_dependency_task_latency_model', 'software_revision', 'cuda_12.2_sm90')

    copy = ref('tensor_iterator_copy_task_sm90.json')
    for key in ('software_revision', 'reference_sm_clock_hz', 'maximum_grid_size', 'resident_ctas_per_sm'):
        t = set_in_block(t, 'tensor_iterator_copy_task_latency_model', key, copy[key])
    block_start = t.index('\n  tensor_iterator_copy_task_latency_model:\n')
    for entry in copy['entries']:
        entry_start = t.index(f'contract: {entry["contract"]}', block_start)
        entry_end = t.index('cta_cycles:', entry_start)
        entry_end = t.index('\n', entry_end)
        segment = t[entry_start:entry_end]
        for key in ('base_cycles', 'fanout_activation_cycles', 'cta_cycles'):
            segment = re.sub(r'(' + key + r':[ \t]*)[^\n]*', lambda m: f'{m.group(1)}{entry[key]}', segment, count=1)
        t = t[:entry_start] + segment + t[entry_end:]

    barrier = ref('sm80_depth3_cta_barrier_service_sm90.json')['microbenchmark']
    t = set_in_block(t, 'sm80_depth3_cta_barrier_service_cycles', 'barrier_cycles', barrier['barrier_cycles'])
    t = set_in_block(t, 'sm80_depth3_cta_barrier_service_cycles', 'software_revision', 'cuda_12.2_sm90')
    t = set_in_block(t, 'sm80_depth3_cta_barrier_service_cycles', 'source_sha256', barrier['source_sha256'])
    t = set_in_block(t, 'sm80_depth3_cta_barrier_service_cycles', 'report_sha256', barrier['report_sha256'])

    envelope = ref('fa2_source_task_outer_envelope_sm90.json')
    for key in ('software_revision', 'minimum_grid_size', 'maximum_grid_size', 'maximum_outer_envelope_ns',
                'source_sha256', 'binary_sha256'):
        t = set_in_block(t, 'fa2_source_task_outer_envelope', key, envelope[key])

    graph = ref('cuda_graph_small_node_span_sm90.json')
    dispatch = ref('cta_dispatch_cost_sm90.json')
    binary = ref('bf16_binary_tensor_iterator_task_floor_curve_sm90.json')
    graph_block = (
        '  # H200-measured graph-node span (cuda_graph_small_node_span_sm90.json).\n'
        '  cuda_graph_small_node_span:\n'
        '    model: cuda_graph_small_node_span_v1\n'
        '    timer_domain: cuda_event_gpu_timeline\n'
        '    threads_per_cta: 128\n'
        '    nodes_per_batch: 64\n'
        f'    reference_path: {H}cuda_graph_small_node_span_sm90.json\n'
        f'    reference_sha256: {sha("cuda_graph_small_node_span_sm90.json")}\n'
        '    production_or_evaluation_duration_consumed: false\n')
    rows = ''.join(f'      {g}: {dispatch["cost_ns_by_grid"][g]}\n'
                   for g in sorted(dispatch['cost_ns_by_grid'], key=int))
    dispatch_block = (
        '  # H200-measured CTA dispatch cost (cta_dispatch_cost_sm90.json).\n'
        '  cta_dispatch_cost:\n'
        f'    model: {dispatch["model"]}\n'
        f'    duration_unit: {dispatch["duration_unit"]}\n'
        f'    timer_domain: {dispatch["timer_domain"]}\n'
        f'    software_revision: {dispatch["software_revision"]}\n'
        f'    derivation_rule: {dispatch["derivation_rule"]}\n'
        f'    lookup_rule: {dispatch["lookup_rule"]}\n'
        '    cost_ns_by_grid:\n' + rows)
    # A floor composed by maximum with a step-down lookup must not decrease
    # with element count.  The H200 curve has one noise-level inversion
    # (16384 < 8192 by 6 ns); the monotone lower envelope min over e' >= e
    # stays below every measured minimum, so it is still a valid floor.
    measured = sorted(((int(e), v) for e, v in binary['floor_ns_by_elements'].items()))
    envelope = []
    running = float('inf')
    for elements, floor_ns in reversed(measured):
        running = min(running, floor_ns)
        envelope.append((elements, running))
    floor_rows = ''.join(f'        {e}: {v}\n' for e, v in sorted(envelope))
    runtime_block = (
        '  # H200-measured BF16 binary TensorIterator task floor\n'
        '  # (bf16_binary_tensor_iterator_task_floor_curve_sm90.json), as its\n'
        '  # monotone lower envelope (8192 takes the 16384 minimum, 6 ns lower).\n'
        '  runtime_task_latency_models:\n'
        '    pytorch_bf16_binary_tensor_iterator:\n'
        f'      model: {binary["model"]}\n'
        f'      duration_unit: {binary["duration_unit"]}\n'
        f'      timer_domain: {binary["timer_domain"]}\n'
        f'      software_revision: {binary["software_revision"]}\n'
        f'      derivation_rule: {binary["derivation_rule"]}\n'
        f'      lookup_rule: {binary["lookup_rule"]}\n'
        f'      probe_block_threads: {binary["probe_launch_geometry"]["block_threads"]}\n'
        '      floor_ns_by_elements:\n' + floor_rows)
    anchor = '\n  small_kernel_span_floor:\n'
    # Insert above the comment paragraph that documents small_kernel_span_floor.
    comment_start = t.index(anchor)
    while True:
        previous = t.rfind('\n', 0, comment_start)
        if t[previous + 1:comment_start].lstrip().startswith('#'):
            comment_start = previous
        else:
            break
    for block, name in ((graph_block, 'cuda_graph_small_node_span'),
                        (dispatch_block, 'cta_dispatch_cost'),
                        (runtime_block, 'runtime_task_latency_models')):
        if f'\n  {name}:\n' in t:
            raise SystemExit(f'{name} already present')
    t = t[:comment_start + 1] + graph_block + dispatch_block + runtime_block + t[comment_start + 1:]
    save(t)


if __name__ == '__main__':
    main()
