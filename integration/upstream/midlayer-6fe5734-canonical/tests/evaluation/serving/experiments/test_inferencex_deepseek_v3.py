"""Published framework recipes keep one immutable input and explicit metrics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.serving.experiments import inferencex_deepseek_v3 as experiment


def test_all_settings_share_one_loaded_input_per_concurrency():
    loaded = []

    def loader(path):
        loaded.append(path)
        return SimpleNamespace(
            case_dir=path,
            manifest=json.loads((path / 'manifest.json').read_text()))

    _, jobs = experiment.load_suite(
        'configs/evaluation/experiments/deepseek_v3_h200_closedloop.json',
        case_loader=loader)
    assert len(jobs) == 30
    assert len(loaded) == len(set(loaded)) == 5
    for concurrency in (4, 8, 16, 32, 64):
        cases = [case for _, c, case in jobs if c == concurrency]
        assert len(cases) == 6
        assert all(case is cases[0] for case in cases)


@pytest.mark.parametrize('setting_id,expected_backend', (
    ('vllm_colocated_mtp_off', 'vllm'),
    ('vllm_pd_mtp_on', 'vllm'),
    ('sglang_colocated_mtp_on', 'sglang_mirror'),
))
def test_execution_preserves_inputs_and_excludes_warmup(
        tmp_path, monkeypatch, setting_id, expected_backend):
    suite = json.loads(Path(
        'configs/evaluation/experiments/deepseek_v3_h200_closedloop.json').read_text())
    setting = next(s for s in suite['settings'] if s['id'] == setting_id)
    case_dir = Path(suite['dataset_root']) / 'c4'
    manifest = json.loads((case_dir / 'manifest.json').read_text())
    case = SimpleNamespace(case_dir=case_dir, manifest=manifest)
    before = {p.name: p.read_bytes() for p in case_dir.iterdir()}
    commands = []

    def simulate(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'metrics': {
            'token_throughput_per_s': 160,
            'output_token_throughput_per_s': 80,
            'latency_distributions': {'tpot_s': {'quantiles': {
                'percentiles': [90], 'values_s': [0.01]}}},
        }}))

    monkeypatch.setattr(experiment.subprocess, 'run', simulate)
    row = experiment.run_point(setting, 4, case, tmp_path)
    assert len(commands) == 2
    for command, field in zip(commands, ('warmup_requests', 'requests')):
        assert command[command.index('--backend') + 1] == expected_backend
        assert command[command.index('--canonical-requests') + 1] == str(
            case_dir / manifest[field]['path'])
        assert command[command.index('--canonical-requests-sha256') + 1] == (
            manifest[field]['sha256'])
        if setting['group'].endswith('_on'):
            assert command[command.index('--mtp-num-speculative-tokens') + 1] == '2'
            assert command[command.index('--mtp-mean-acceptance-rate') + 1] == '0.673'
    assert row['interactivity_p90_tok_s_user'] == 100
    assert row['output_token_throughput_per_decode_gpu_tok_s'] == 10
    assert row['total_token_throughput_per_decode_gpu_tok_s'] == 20
    assert row['gpu_count'] == (16 if setting['group'].startswith('pd_') else 8)
    assert {p.name: p.read_bytes() for p in case_dir.iterdir()} == before
