import os
import sys
from pathlib import Path

import pytest


def test_serving_emulation_rejects_positive_stagger_before_outputs(tmp_path, monkeypatch, capsys):
    import scripts.serve_trace.trace_serving as serving

    trace_path = tmp_path / "trace.json"
    monkeypatch.delenv("CUDA_SIDECAR_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "run_vllm_serving_emulation.py",
        "--stagger", "0.25",
        "--trace-output", str(trace_path),
    ])

    with pytest.raises(SystemExit) as exc:
        serving.main()

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--stagger > 0" in err
    assert "not supported" in err
    assert "true staggered continuous batching" in err
    assert "CUDA_SIDECAR_PATH" not in os.environ
    assert not trace_path.exists()
    assert not Path(str(trace_path) + "_provenance.json").exists()
    assert not Path(str(trace_path) + "_scheduler.jsonl").exists()


def test_serving_emulation_default_stagger_passes_validation():
    import scripts.serve_trace.trace_serving as serving

    parser = serving.build_arg_parser()
    args = parser.parse_args([])
    serving.validate_args(args, parser)

    assert args.stagger == 0.0


def test_prompts_file_updates_effective_num_requests(tmp_path):
    import json
    from types import SimpleNamespace

    import scripts.serve_trace.trace_serving as serving

    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps({
        "prompts": ["alpha", "beta"],
        "prompt_lengths": [3, 4],
    }))
    args = SimpleNamespace(
        prompts_file=str(workload),
        dataset=None,
        prompt_len=16,
        num_requests=10,
        max_tokens=1,
        seed=42,
    )

    prompts, actual_lens = serving.load_prompts_and_update_request_count(
        args, tokenizer=None)

    assert prompts == ["alpha", "beta"]
    assert actual_lens == [3, 4]
    assert args.num_requests == 2
