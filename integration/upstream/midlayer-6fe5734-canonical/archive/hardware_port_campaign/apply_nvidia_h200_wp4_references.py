#!/usr/bin/env python3
"""Apply WP4 (DRAM / L2) H200 references to configs/hardware/nvidia_h200.yaml.

Every number is read from `microbench/nvidia_h200/references/*_sm90.json`;
cycle conversions use the H200 1.98 GHz SM clock and the 3.125 GHz DRAM
clock of the yaml.  Apply once to the A800-proxy yaml: the patterns match
the A800 values being replaced and fail loudly if already applied.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
YAML = ROOT / 'configs/hardware/nvidia_h200.yaml'
REFS = ROOT / 'microbench/nvidia_h200/references'
SM_HZ = 1.98e9
DRAM_HZ = 3.125e9
L2_BYTES = 62914560  # device-reported H200 L2 (cudaDeviceProp.l2CacheSize)


def ref(name):
    return json.loads((REFS / name).read_text())


def sub_once(text, pattern, replacement):
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if count != 1:
        raise SystemExit(f'pattern not found: {pattern}')
    return new


def main():
    text = YAML.read_text()
    l2 = ref('l2_read_bandwidth_sm90.json')
    l2_rate = l2['throughput_per_clk']
    text = sub_once(text, r'^(    throughput_per_clk: )2563\.037314779177.*$',
                    rf'\g<1>{l2_rate}  # H200 l2_read_bandwidth_sm90.json (CUDA-event L2-resident ld.global.cg)')
    text = sub_once(text, r'^(    size_bytes: )52428800.*$',
                    rf'\g<1>{L2_BYTES}                  # 60 MiB, device-reported H200 L2')
    ramp = ref('dram_finite_read_stream_ramp_sm90.json')['derived_model']
    dram_clk = ramp['dram_read_capacity_bytes_per_s'] / DRAM_HZ
    text = sub_once(text, r'^(    throughput_per_clk: )1536\s*$', rf'\g<1>{dram_clk}')
    text = sub_once(text, r'^(  dram_read_capacity_per_clk: )1536.*$',
                    rf'\g<1>{dram_clk}  # H200 envelope of dram_read_bw and the finite-stream ramp')
    lat = ref('pointer_chase_latency_sm90.json')
    dram_ns = lat['dram_latency_ns']
    l2_ns = lat['l2_latency_ns']
    text = sub_once(text, r'^(  dram_latency_ns: ).*$', rf'\g<1>{dram_ns}')
    text = sub_once(text, r'^(  l2_latency_ns: ).*$', rf'\g<1>{l2_ns}')
    text = sub_once(text, r'^(  l2_hit_latency_local_cycles: ).*$', rf'\g<1>{l2_ns * SM_HZ / 1e9}')
    text = sub_once(text, r'^(  l2_hit_latency_remote_cycles: ).*$', rf'\g<1>{l2_ns * SM_HZ / 1e9}')
    text = sub_once(text, r'^(  dram_miss_extra_cycles: ).*$',
                    rf'\g<1>{(dram_ns - l2_ns) * SM_HZ / 1e9}')
    gemv = ref('scalar_half_dram_stream_sm90.json')
    text = sub_once(text, r'^(  gemv_scalar_half_dram_transactions_per_sm_s: ).*$',
                    rf"\g<1>{gemv['gemv_scalar_half_dram_transactions_per_sm_s']}")
    text = sub_once(text, r'^(  gemv_scalar_half_source_fill_ns: ).*$',
                    rf"\g<1>{gemv['gemv_scalar_half_source_fill_ns']}")
    for tier in ('l2', 'dram'):
        service = ref(f'add_rmsnorm_mixed_{tier}_service_sm90.json')['selected_service']
        for kind in ('read', 'write'):
            name = f'bf16_vec8_add_rmsnorm_mixed_{tier}_{kind}'
            start = text.index(f'    - service: {name}\n')
            end = text.index('\n    - service:', start + 1) if '\n    - service:' in text[start + 1:] else len(text)
            block = text[start:end]
            capacity = service[f'{kind}_logical_byte_capacity_per_s']
            block = re.sub(r'(      logical_byte_capacity_per_s: ).*', rf'\g<1>{capacity}', block, count=1)
            block = block.replace(
                f'microbench/nvidia_a800/references/add_rmsnorm_mixed_{tier}_service_sm80.json',
                f'microbench/nvidia_h200/references/add_rmsnorm_mixed_{tier}_service_sm90.json')
            if '      logical_byte_capacity_per_s_by_grid_ctas:' in block:
                by_grid = service[f'{kind}_logical_byte_capacity_per_s_by_grid_ctas']
                head, _, tail = block.partition('      logical_byte_capacity_per_s_by_grid_ctas:\n')
                rest = re.sub(r'^(        \d+: .*\n?)+', '', tail, count=1, flags=re.M)
                rows = ''.join(f'        {int(g)}: {v}\n' for g, v in sorted(by_grid.items(), key=lambda kv: int(kv[0])))
                if not rest:
                    rows = rows.rstrip('\n')
                block = head + '      logical_byte_capacity_per_s_by_grid_ctas:\n' + rows + rest
            text = text[:start] + block + text[end:]
    YAML.write_text(text)


if __name__ == '__main__':
    main()
