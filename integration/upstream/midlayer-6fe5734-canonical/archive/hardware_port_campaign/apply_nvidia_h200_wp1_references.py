#!/usr/bin/env python3
"""Apply WP1 (Triton fused-MoE operand-stream services) H200 references.

Every H200 service row that still cites an A800 `*_sm80.json` reference is
rewritten to the same-pointer H200 `*_sm90.json` reference and its measured
capacity; `grouped_fused_moe_operand_read` takes the H200 re-run of the A800
BM16/BN32/BK64 operand-read probe instead of the clock-scaled A800 proxy.
"""
from nvidia_h200_yaml_edit import ref, load, save

A = 'microbench/nvidia_a800/references/'
H = 'microbench/nvidia_h200/references/'
OPERAND_A800 = ('profiler_output/microbench/nvidia_a800/fused_moe_operand_820defa_20260830/'
                'summary.json#operand_bm16_bn32_bk64')
OPERAND_H200 = H + 'grouped_fused_moe_operand_read_sm90.json#selected_services/operand_bm16_bn32_bk64'
OPERAND_PROXY_COMMENT = (
    '      # A800-proxy carried over by cycle-equivalence, not verbatim: the A800\n'
    '      # microbench measured 2427259310000.0 B/s at 108 SMs x 1.14 GHz\n'
    '      # (nvidia_a800.yaml components.sm.count and clock_domains.sm), and the\n'
    '      # two parts are cycle-close, so the absolute capacity scales with\n'
    '      # SM count x clock: 2427259310000.0 x (132 x 1.98e9) / (108 x 1.14e9)\n'
    '      # = 5152603096666.667 B/s (H200 fields: components.sm.count=132,\n'
    '      # clock_domains.sm=1980000000).  The underlying evidence stays the A800\n'
    '      # BM16 BN32/BK64 operand-stream probe below; no H200 profiler capture\n'
    '      # exists for this mechanism.\n')
OPERAND_COMMENT = (
    '      # H200 re-run of the A800 BM16/BN32/BK64 operand-read probe\n'
    '      # (grouped_fused_moe_operand_read_sm90.json), clock-gated at 1980 MHz.\n')


def resolve(evidence):
    name, pointer = evidence.split('#')
    node = ref(name)
    for part in pointer.split('/'):
        node = node[part]
    if isinstance(node, float):
        return node
    if isinstance(node, dict):
        return node['logical_byte_capacity_per_s']
    raise SystemExit(f'{evidence}: unexpected node type {type(node).__name__}')


def set_capacity_above(lines, index, capacity):
    j = index
    while not lines[j].lstrip().startswith('logical_byte_capacity_per_s:'):
        j -= 1
        if lines[j].lstrip().startswith('- service:'):
            raise SystemExit(f'no capacity line above line {index + 1}')
    indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
    lines[j] = f'{indent}logical_byte_capacity_per_s: {capacity}'


def main():
    t = load()
    if t.count(OPERAND_PROXY_COMMENT) != 1:
        raise SystemExit('operand-read proxy comment not found once')
    t = t.replace(OPERAND_PROXY_COMMENT, OPERAND_COMMENT)
    lines = t.split('\n')
    rewritten = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith('evidence_ref:'):
            continue
        evidence = stripped[len('evidence_ref:'):].strip()
        if evidence == OPERAND_A800:
            h200_evidence = OPERAND_H200
        elif evidence.startswith(A):
            name, pointer = evidence[len(A):].split('#')
            if not name.endswith('_sm80.json'):
                raise SystemExit(f'unexpected A800 reference name {name}')
            h200_evidence = H + name[:-len('_sm80.json')] + '_sm90.json#' + pointer
        elif 'nvidia_a800' in evidence:
            raise SystemExit(f'unhandled A800 evidence {evidence}')
        else:
            continue
        set_capacity_above(lines, i, resolve(h200_evidence[len(H):]))
        lines[i] = line.replace(evidence, h200_evidence)
        rewritten += 1
    if rewritten != 46:
        raise SystemExit(f'rewrote {rewritten} evidence rows, expected 46')
    save('\n'.join(lines))
    print(f'rewrote {rewritten} evidence rows')


if __name__ == '__main__':
    main()
