#!/usr/bin/env python3
"""Per-geometry FA3 MLA predicted vs real per-call durations on H200 (wp10).

Evaluation only.  Real: the rank-0 per-call medians written by
``extract_nvidia_h200_attention_kernel_calls.py`` (DeepSeek-V3-L32 TP4 pure-decode
windows C=1/4/16/64 and the prefill window, grouped by launch and by the step
token count).  Predicted: ``predict(KernelSpec, HWSpec, "midlayer")`` of the
H200 launch the selector derives for the same request shape.  The decode
windows do not record per-call KV lengths; each decode geometry is predicted
at the first (7396) and last (8396) decode position of the 1024-token
generation so the table brackets the window.  No duration feeds the model.

Usage:
    PYTHONPATH=. python scripts/evaluation/serving/compare_nvidia_h200_fa3_geometries.py \
        --real-calls attention_calls_r2_prefill.json --output fa3_geometries.json
"""
import argparse
import json
from pathlib import Path

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config

PROMPT = 7373
HEADS = 32
DECODE_KV = (PROMPT + 23, PROMPT + 1023)


def _spec(hardware, variant, q_lens, kv_lens):
    decode = variant == 'mla_decode'
    shape = {'batch': len(q_lens), 'q_len': max(q_lens),
             'kv_len': max(kv_lens), 'qk_head_dim': 576 if decode else 192,
             'v_head_dim': 512 if decode else 128,
             'num_query_heads': HEADS, 'kv_lora_rank': 512}
    attributes = {'causal': True, 'kv_layout': 'mla', 'kv_dist': 'uniform',
                  'q_lens': tuple(q_lens), 'kv_lens': tuple(kv_lens)}
    launch = select_launch_config(
        'nvidia_h200', KernelKind.FLASH_ATTN, shape, 'bf16', variant,
        attributes, hardware)
    return KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id='nvidia_h200',
                      shape=shape, dtype='bf16', variant=variant,
                      launch=launch, attributes=attributes)


def _combine(hardware, main):
    shape = {'batch': main.shape['batch'], 'q_len': 1,
             'kv_len': main.shape['kv_len'], 'head_dim': 512,
             'num_query_heads': HEADS, 'num_kv_heads': 1}
    attributes = {**main.attributes, 'fa_ksteps': None, 'fa_eff_mfrac': 1.0,
                  'num_kv_splits': main.launch.tiling.num_kv_splits}
    launch = select_launch_config(
        'nvidia_h200', KernelKind.FLASH_ATTN, shape, 'bf16', 'combine',
        attributes, hardware)
    return KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id='nvidia_h200',
                      shape=shape, dtype='bf16', variant='combine',
                      launch=launch, attributes={
                          name: attributes[name] for name in (
                              'causal', 'kv_layout', 'kv_dist', 'fa_ksteps',
                              'fa_eff_mfrac', 'q_lens', 'kv_lens')})


def _real(window, name_part, tokens):
    rows = [row for row in window['launches']
            if name_part in row['name']
            and row['step_tokens_from_kv_write'] == tokens]
    if len(rows) != 1:
        raise ValueError(f'expected one real launch for {name_part}/{tokens}')
    return rows[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-calls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    real = json.loads(args.real_calls.read_text())['windows']
    hardware = HWSpec('configs/hardware/nvidia_h200.yaml')
    rows = []
    for batch in (1, 4, 16, 64):
        window = real[f'C={batch}']
        for kind, name_part in (('main', 'FlashAttnFwdSm90'),
                                ('combine', 'FlashAttnFwdCombine')):
            observed = _real(window, name_part, batch)
            predicted = []
            for kv in DECODE_KV:
                spec = _spec(hardware, 'mla_decode', (1,) * batch,
                             (kv,) * batch)
                if kind == 'combine':
                    spec = _combine(hardware, spec)
                predicted.append((spec, predict(spec, hardware, 'midlayer')))
            rows.append({
                'geometry': f'decode C={batch} {kind}',
                'algorithm': predicted[0][0].launch.algorithm,
                'predicted_grid': list(predicted[0][0].launch.grid),
                'real_grid': observed['grid'],
                'real_median_us': observed['median_us'],
                'real_calls': observed['calls'],
                'predicted_us_kv_first': predicted[0][1].t_total_s * 1e6,
                'predicted_us_kv_last': predicted[1][1].t_total_s * 1e6,
                'bottleneck': predicted[0][1].bottleneck_group,
            })
    for prompts in (1, 3, 4):
        tokens = prompts * PROMPT
        observed = _real(real['prefill'], 'FlashAttnFwdSm90', tokens)
        spec = _spec(hardware, 'mla_prefill', (PROMPT,) * prompts,
                     (PROMPT,) * prompts)
        prediction = predict(spec, hardware, 'midlayer')
        rows.append({
            'geometry': f'prefill {prompts}x{PROMPT} main',
            'algorithm': spec.launch.algorithm,
            'predicted_grid': list(spec.launch.grid),
            'real_grid': observed['grid'],
            'real_median_us': observed['median_us'],
            'real_calls': observed['calls'],
            'predicted_us_kv_first': prediction.t_total_s * 1e6,
            'predicted_us_kv_last': prediction.t_total_s * 1e6,
            'bottleneck': prediction.bottleneck_group,
        })
    for row in rows:
        for key in ('kv_first', 'kv_last'):
            row[f'signed_error_{key}'] = (
                row[f'predicted_us_{key}'] / row['real_median_us'] - 1.0)
    args.output.write_text(json.dumps({
        'evaluation_only': True,
        'production_or_evaluation_duration_consumed_by_model': False,
        'rows': rows}, indent=1) + '\n')
    for row in rows:
        print(f"{row['geometry']:<26} real {row['real_median_us']:9.2f} us "
              f"pred {row['predicted_us_kv_first']:9.2f}..{row['predicted_us_kv_last']:9.2f} us "
              f"err {row['signed_error_kv_first']:+.1%}..{row['signed_error_kv_last']:+.1%} "
              f"grid {row['predicted_grid']} vs {row['real_grid']}")


if __name__ == '__main__':
    main()
