#!/usr/bin/env python3
"""Apply WP6 (FA2 scalar/resident/spill/combine/Oaccum dependency) H200 references.

Each block is rebuilt from the H200 reference's `h200_yaml_values`, with its
evidence comment pointing at the H200 reference.  The spill blocks keep the
pinned FA2 2.8.3 SM80 production binary/SASS digests: they identify the
source program whose static stack/site contract the H200 probe reproduces.
"""
import hashlib
import re

from nvidia_h200_yaml_edit import REFS, load, ref, save

H = 'microbench/nvidia_h200/references/'
SPILL_REVISION = 'flash_attn_2_8_3_source_cuda_12_2_sm90_probe'


def sha(name):
    return hashlib.sha256((REFS / name).read_bytes()).hexdigest()


def probe_source_sha(reference, suffix):
    files = [f for key, f in reference['h200_provenance']['files'].items() if key.endswith(suffix)]
    if len(files) != 1:
        raise SystemExit(f'{suffix}: {len(files)} provenance files')
    return files[0]['sha256']


def replace_block(text, key, new_block):
    """Replace `  key:` plus its preceding comment paragraph with new_block."""
    anchor = f'\n  {key}:\n'
    if text.count(anchor) != 1:
        raise SystemExit(f'{key} not unique')
    start = text.index(anchor) + 1
    while True:
        previous = text.rfind('\n', 0, start - 1)
        if text[previous + 1:start].startswith('  #'):
            start = previous + 1
        else:
            break
    end_match = re.search(r'^  \S', text[text.index(anchor) + len(anchor):], re.M)
    end = text.index(anchor) + len(anchor) + end_match.start()
    return text[:start] + new_block + text[end:]


def old_field(text, key, field):
    body = text[text.index(f'\n  {key}:\n'):]
    match = re.search(r'^    ' + re.escape(field) + r': (\S+)$', body, re.M)
    return match.group(1)


STAGES = ('fp32_max_quad4_cycles', 'fp32_local_sum_cycles', 'fp32_scale_exp2_sum_cycles',
          'cta_barrier_cycles', 'fp32_quad4_sum_cycles', 'integer_column_predicate_cycles',
          'fp32_to_bf16_conversion_cycles')
FACTORS = ('resident_ctas_per_sm', 'fp32_max_quad4', 'cta_barrier', 'fp32_quad4_sum',
           'fp32_scale_exp2_sum', 'integer_column_predicate', 'fp32_to_bf16_conversion')


def scalar_block():
    scalar = ref('fa2_scalar_dependency_sm90.json')
    resident = ref('fa2_resident_dependency_overlap_sm90.json')
    factors = resident['h200_yaml_values']['two_resident_demand_factors']
    lines = [
        '  # H200 re-measurement of the isolated FA2 scalar dependency probe',
        '  # (fa2_scalar_dependency_sm90.json; trials 0-1 define, trial 2 validates)',
        '  # and of the two-resident overlap factors for the (4, 16) geometry',
        '  # (fa2_resident_dependency_overlap_sm90.json: 264/132 CTA grid-span ratio',
        '  # divided by two).  Neither consumes a production or evaluation duration.',
        '  register_reduction_dependency_cycles:',
        '    model: isolated_register_reduction_and_cta_barrier_v1',
        '    cycle_unit: cycles_per_128_thread_cta_stage',
        '    timer_domain: clock64_same_cta_envelope_cycles',
        '    software_revision: cuda_12.2_sm90',
        '    threads_per_cta: 128',
        '    quad_reduction_width: 4',
        f'    source_sha256: {probe_source_sha(scalar, "fa2_scalar_dependency.cu")}',
        f'    report_sha256: {scalar["h200_provenance"]["report_sha256"]}',
        f'    two_resident_reference_sha256: {sha("fa2_resident_dependency_overlap_sm90.json")}',
        '    entries:',
    ]
    for entry in scalar['h200_yaml_values']['entries']:
        lines.append(f'      - lane_local_rows_per_thread: {entry["lane_local_rows_per_thread"]}')
        lines.append(f'        score_values_per_lane_local_row: {entry["score_values_per_lane_local_row"]}')
        lines += [f'        {s}: {entry[s]}' for s in STAGES]
        geometry = (entry['lane_local_rows_per_thread'], entry['score_values_per_lane_local_row'])
        if geometry == (4, 16):
            if entry['two_resident_demand_factors'] != factors:
                raise SystemExit('scalar and resident references disagree on (4, 16) factors')
            lines.append('        two_resident_demand_factors:')
            lines += [f'          {f}: {factors[f]}' for f in FACTORS]
        elif 'two_resident_demand_factors' in entry:
            raise SystemExit(f'unexpected factors for {geometry}')
    return '\n'.join(lines) + '\n'


def spill_block(text, key, model, cycle_unit, reference_name, description):
    reference = ref(reference_name)
    values = reference['h200_yaml_values']
    if values['yaml_key'] != key or values['software_revision'] != SPILL_REVISION:
        raise SystemExit(f'{reference_name}: yaml key/revision mismatch')
    lines = [f'  # {line}' for line in description] + [
        f'  {key}:',
        f'    model: {model}',
        f'    cycle_unit: {cycle_unit}',
        '    timer_domain: clock64_same_cta_envelope_cycles',
        f'    software_revision: {SPILL_REVISION}',
        '    threads_per_cta: 128',
        f'    stack_bytes_per_thread: {values["stack_bytes_per_thread"]}',
        f'    scalar_store_sites: {values["scalar_store_sites"]}',
        f'    scalar_load_sites: {values["scalar_load_sites"]}',
        f'    spill_round_cycles: {values["spill_round_cycles"]}',
        f'    source_sha256: {probe_source_sha(reference, "fa2_local_spill_round.cu")}',
        f'    report_sha256: {reference["h200_provenance"]["report_sha256"]}',
        f'    production_binary_sha256: {old_field(text, key, "production_binary_sha256")}',
        f'    production_sass_sha256: {old_field(text, key, "production_sass_sha256")}',
    ]
    return '\n'.join(lines) + '\n'


FRAGMENT_FIELDS = ('slse_publish_transpose_cycles', 'post_max_exp_local_sum_cycles',
                   'post_sum_log_cycles', 'scale_exp_smem_barrier_cycles')


def combine_block():
    reference = ref('fa2_combine_dependency_sm90.json')
    values = reference['h200_yaml_values']
    lines = [
        '  # H200 re-measurement of the FA2 2.8.3 split-reducer dependency probe',
        '  # (fa2_combine_dependency_sm90.json).  source_fragment_entries holds the',
        '  # A800 split set plus every other split the FA2 heuristic can choose with',
        '  # 132 SMs inside the probe range (D64 2..16, D128 2..32), measured in a',
        '  # separate capture of the same probe.  No production duration is consumed.',
        '  fa2_combine_dependency_cycles:',
        '    model: isolated_fa2_combine_source_dependency_v2',
        '    cycle_unit: cycles_per_128_thread_cta_stage',
        '    timer_domain: clock64_same_cta_envelope_cycles',
        '    software_revision: cuda_12.2_sm90',
        '    threads_per_cta: 128',
        f'    source_sha256: {probe_source_sha(reference, "fa2_combine_dependency.cu")}',
        f'    report_sha256: {reference["h200_provenance"]["report_sha256"]}',
        '    reduction_entries:',
    ]
    for entry in values['reduction_entries']:
        lines += [f'      - allreduce_width: {entry["allreduce_width"]}',
                  f'        fp32_max_allreduce_cycles: {entry["fp32_max_allreduce_cycles"]}',
                  f'        fp32_sum_allreduce_cycles: {entry["fp32_sum_allreduce_cycles"]}']
    lines.append('    conversion_entries:')
    for entry in values['conversion_entries']:
        lines += [f'      - block_m: {entry["block_m"]}',
                  f'        head_dim: {entry["head_dim"]}',
                  f'        output_values_per_lane: {entry["output_values_per_lane"]}',
                  f'        fp32_to_bf16_output_conversion_cycles: {entry["fp32_to_bf16_output_conversion_cycles"]}']
    lines.append('    source_fragment_entries:')
    fragments = values['source_fragment_entries'] + values['heuristic_extension_source_fragment_entries']
    fragments.sort(key=lambda e: (e['head_dim'] != 64, e['num_splits']))
    for entry in fragments:
        lines += [f'      - block_m: {entry["block_m"]}',
                  f'        head_dim: {entry["head_dim"]}',
                  f'        num_splits: {entry["num_splits"]}',
                  f'        max_splits: {entry["max_splits"]}']
        lines += [f'        {f}: {entry[f]}' for f in FRAGMENT_FIELDS]
    return '\n'.join(lines) + '\n'


def oaccum_block():
    reference = ref('fa2_oaccum_ldg128_load_use_sm90.json')
    values = reference['h200_yaml_values']
    if values['software_revision'] != 'cuda_12.2_sm90':
        raise SystemExit('Oaccum software revision mismatch')
    lines = [
        '  # H200 re-measurement of the FA2 Oaccum LDG128 + four dependent FMA probe',
        '  # (fa2_oaccum_ldg128_load_use_sm90.json).  Trials 0-1 define, trial 2',
        '  # validates: MAPE/max APE 3.27%/17.73% (L2-warm), 6.47%/17.29% (DRAM).',
        '  # The NCU cache-tier audit is replaced by a clock64 DRAM/L2 separation',
        '  # gate (ncu is not installed).  No production duration is consumed.',
        '  fa2_oaccum_load_use_cycles:',
        '    model: isolated_ldg128_fp32x4_load_to_operands_ready_v1',
        '    cycle_unit: sm_cycles_per_128_thread_cta_iteration',
        '    timer_domain: clock64_same_cta_envelope_differential',
        '    software_revision: cuda_12.2_sm90',
        '    threads_per_cta: 128',
        '    load_bytes_per_lane: 16',
        '    values_per_lane: 4',
        '    dependent_fp32_fma_per_lane: 4',
        f'    source_sha256: {probe_source_sha(reference, "fa2_oaccum_ldg128_load_use.cu")}',
        f'    report_sha256: {reference["h200_provenance"]["report_sha256"]}',
        '    cache_entries:',
    ]
    for entry in values['cache_entries']:
        lines += [f'      - cache_tier: {entry["cache_tier"]}',
                  f'        model: {entry["model"]}',
                  f'        first_load_ready_cycles: {entry["first_load_ready_cycles"]}',
                  f'        additional_load_interval_cycles: {entry["additional_load_interval_cycles"]}']
    return '\n'.join(lines) + '\n'


def main():
    t = load()
    t = replace_block(t, 'register_reduction_dependency_cycles', scalar_block())
    t = replace_block(t, 'fa2_local_spill_dependency_cycles', spill_block(
        t, 'fa2_local_spill_dependency_cycles', 'isolated_fa2_local_spill_round_v1',
        'cycles_per_128_thread_cta_spill_round', 'fa2_local_spill_round_d64_chunked_sm90.json',
        ['H200 re-measurement of the FA2 2.8.3 D64 chunked-prefill local-stack',
         'round (fa2_local_spill_round_d64_chunked_sm90.json): 56-byte stack, 13',
         'STL / 13 LDL sites, local-spill minus register-only control.  The',
         'production digests identify the pinned SM80 source program whose static',
         'contract the sm_90 probe build reproduces.']))
    t = replace_block(t, 'fa2_dense_d128_causal_local_spill_dependency_cycles', spill_block(
        t, 'fa2_dense_d128_causal_local_spill_dependency_cycles',
        'isolated_fa2_dense_d128_causal_local_spill_round_v1',
        'cycles_per_128_thread_cta_local_stack_round', 'fa2_local_spill_round_d128_causal_sm90.json',
        ['H200 re-measurement of the FA2 2.8.3 dense causal D128 local-stack round',
         '(fa2_local_spill_round_d128_causal_sm90.json): 176-byte stack, 51 STL /',
         '53 LDL sites.  Production digests as for the D64 block above.']))
    t = replace_block(t, 'fa2_combine_dependency_cycles', combine_block())
    t = replace_block(t, 'fa2_oaccum_load_use_cycles', oaccum_block())
    save(t)


if __name__ == '__main__':
    main()
