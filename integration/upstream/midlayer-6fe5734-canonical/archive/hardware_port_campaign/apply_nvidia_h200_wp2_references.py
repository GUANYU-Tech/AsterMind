#!/usr/bin/env python3
"""Apply WP2 (MoE CUDA probes: grouped operand services, BM16 tile pipeline,
MoE routing atomics) H200 references to the H200 yaml."""
import hashlib
import re

from nvidia_h200_yaml_edit import REFS, load, ref, save

A = 'microbench/nvidia_a800/references/'
H = 'microbench/nvidia_h200/references/'


def sha(name):
    return hashlib.sha256((REFS / name).read_bytes()).hexdigest()


def set_service_by_evidence(text, a800_evidence, h200_evidence, capacity, occurrences):
    """Rewrite each service entry citing `a800_evidence`: capacity + evidence."""
    lines = text.split('\n')
    hits = [i for i, line in enumerate(lines) if line.strip() == f'evidence_ref: {a800_evidence}']
    if len(hits) != occurrences:
        raise SystemExit(f'{a800_evidence}: {len(hits)} entries, expected {occurrences}')
    for i in hits:
        j = i
        while not lines[j].lstrip().startswith('logical_byte_capacity_per_s:'):
            j -= 1
            if lines[j].lstrip().startswith('- service:'):
                raise SystemExit(f'no capacity line above {a800_evidence}')
        indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
        lines[j] = f'{indent}logical_byte_capacity_per_s: {capacity}'
        lines[i] = lines[i].replace(a800_evidence, h200_evidence)
    return '\n'.join(lines)


def set_block_line(text, block, key, value, indent=4):
    start = text.index(f'\n  {block}:\n') + 1
    end_match = re.search(r'^  \S', text[start + 1:], re.M)
    end = start + 1 + end_match.start() if end_match else len(text)
    body = text[start:end]
    pattern = re.compile(r'^(' + ' ' * indent + re.escape(key) + r':[ \t]*)[^\n#]*?(\s*#[^\n]*)?$', re.M)
    if len(pattern.findall(body)) != 1:
        raise SystemExit(f'{block}.{key} not unique')
    body = pattern.sub(lambda m: f'{m.group(1)}{value}{m.group(2) or ""}', body, count=1)
    return text[:start] + body + text[end:]


def main():
    t = load()
    residency = ref('bm64_operand_residency_sm90.json')['selected_services']
    lines_hit = A + 'bm64_operand_residency_sm80.json#selected_services'
    # The first residency entry is the combined L2-hit service; the other two
    # are the weight-role DRAM-miss services (same value, as on A800).
    text_lines = t.split('\n')
    hit_indices = [i for i, line in enumerate(text_lines) if line.strip() == f'evidence_ref: {lines_hit}']
    if len(hit_indices) != 3:
        raise SystemExit(f'residency evidence entries: {len(hit_indices)}')
    values = (residency['grouped_operand_l2_hit_bytes_per_s'],
              residency['grouped_weight_dram_miss_bytes_per_s'],
              residency['grouped_weight_dram_miss_bytes_per_s'])
    for index, value in zip(hit_indices, values):
        j = index
        while not text_lines[j].lstrip().startswith('logical_byte_capacity_per_s:'):
            j -= 1
        indent = text_lines[j][:len(text_lines[j]) - len(text_lines[j].lstrip())]
        text_lines[j] = f'{indent}logical_byte_capacity_per_s: {value}'
        text_lines[index] = text_lines[index].replace(lines_hit, H + 'bm64_operand_residency_sm90.json#selected_services')
    t = '\n'.join(text_lines)
    t = set_service_by_evidence(
        t, A + 'bm64_full_group_pipeline_sm80.json#selected_service',
        H + 'bm64_full_group_pipeline_sm90.json#selected_service',
        ref('bm64_full_group_pipeline_sm90.json')['selected_service']['logical_byte_capacity_per_s'], 1)
    mixed = ref('bm64_mixed_group_pipeline_sm90.json')['selected_services_by_weight_streams']
    for streams in ('2', '4', '8'):
        t = set_service_by_evidence(
            t, A + f'bm64_mixed_group_pipeline_sm80.json#selected_services_by_weight_streams/{streams}',
            H + f'bm64_mixed_group_pipeline_sm90.json#selected_services_by_weight_streams/{streams}',
            mixed[streams], 1)

    for key, value in ref('moe_routing_atomic_service_sm90.json')['yaml_values_ops_per_s_per_subcore'].items():
        pattern = re.compile(r'^(\s+' + re.escape(key) + r':[ \t]*)[^\n]*$', re.M)
        if len(pattern.findall(t)) != 1:
            raise SystemExit(f'{key} not unique')
        t = pattern.sub(lambda m: f'{m.group(1)}{value}', t, count=1)

    block = 'grouped_gemm_tile_pipeline'
    pipeline = ref('bm16_grouped_gemm_pipeline_sm90.json')
    long_k = ref('bm16_grouped_gemm_long_k_pipeline_sm90.json')['cross_device_upper_envelope']
    empty = ref('bm16_grouped_gemm_empty_tail_sm90.json')['service_parameters']
    t = set_block_line(t, block, 'software_revision', 'vllm_0.24.0_triton_3.6_sm90')
    t = set_block_line(t, block, 'registers_per_thread', 45)
    old_registers = '    registers_per_thread_by_variant:\n      w1: 40\n      w2: 48\n'
    if t.count(old_registers) != 1:
        raise SystemExit('registers_per_thread_by_variant block not found once')
    t = t.replace(old_registers, '    registers_per_thread_by_variant:\n      w1: 39\n      w2: 45\n')
    t = set_block_line(t, block, 'sparse_refill_wave_count', 1)
    t = set_block_line(t, block, 'fixed_span_ns', pipeline['service_parameters']['fixed_span_ns'])
    t = set_block_line(t, block, 'local_k_round_interval_ns', pipeline['service_parameters']['local_k_round_interval_ns'])
    t = set_block_line(t, block, 'empty_tail_round_ns', empty['empty_tail_round_ns'])
    t = set_block_line(t, block, 'long_k_anchor_completion_ns', long_k['anchor_completion_ns'])
    t = set_block_line(t, block, 'long_k_round_interval_ns', long_k['tail_round_interval_ns'])
    t = set_block_line(t, block, 'production_source_sha256', pipeline['production_source_sha256'])
    t = set_block_line(t, block, 'reference_sha256', sha('bm16_grouped_gemm_pipeline_sm90.json'))
    t = set_block_line(t, block, 'empty_tail_reference_sha256', sha('bm16_grouped_gemm_empty_tail_sm90.json'))
    t = set_block_line(t, block, 'long_k_reference_sha256', sha('bm16_grouped_gemm_long_k_pipeline_sm90.json'))
    t = set_block_line(t, block, 'sparse_refill_reference_sha256', sha('bm16_grouped_gemm_sparse_second_wave_sm90.json'))
    save(t)


if __name__ == '__main__':
    main()
