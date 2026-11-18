"""A single real request still has a complete WideEP plan with idle replicas."""

import math
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(('model', 'tp', 'pp'), [
    ('deepseek-v3', 1, 1),
    ('deepseek-v4-flash', 2, 1),
    ('deepseek-v4-flash', 1, 2),
    ('kimi-k3', 2, 1),
    ('kimi-k3', 1, 2),
])
def test_single_real_request_with_an_idle_replica(model, tp, pp):
    from astermind.common.bootstrap import activate

    root = activate()
    from astermind.engine.compute.runtime import (
        MidlayerCellConfig, create_midlayer_runtime, prepare_scheduled_step,
    )
    from src.serving.kernel_lowering import (
        k3_dp_contract, k3_sp_contract, k3_tp_contract,
        v3_dp_contract, v4_dp_contract, v4_modelsp_contract,
    )
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    from src.serving.workload.runtime_inputs import K3DataParallelSourceInputs, K3SourceInputs

    config = MidlayerCellConfig(
        root, model_file=f'configs/model/{model}.json',
        hardware_file='configs/hardware/nvidia_h200.yaml', hardware_id='nvidia_h200',
        scheduler_adapter='vllm_0_30', compute_mode='midlayer',
        tensor_parallel_size=tp, pipeline_parallel_size=pp, data_parallel_size=2,
        enable_expert_parallel=True, expert_routing_seed=42,
        max_model_len=256, max_batched_tokens=16, kv_block_size_tokens=256,
    )
    _, execution, _ = create_midlayer_runtime(config, with_scheduler=False)
    for phase in ('prefill', 'decode'):
        prefill = phase == 'prefill'
        request = ScheduledRequest('r', phase, 2 if prefill else 0, 0 if prefill else 1,
                                   computed_tokens=0 if prefill else 2, data_parallel_rank=0)
        schedule = ScheduleStep(phase, 'main', (request,))
        source = None
        if model == 'kimi-k3':
            layers = execution.model.kimi_k3_semantics.layer_attention_types
            active = K3SourceInputs(
                ('r',), (0, 1) if prefill else (2,),
                tuple((1,) if layer == 'kda' else () for layer in layers),
                256, 8, 'supports_out' if prefill else None,
            )
            source = K3DataParallelSourceInputs((active, None), (256, 256), (8, 8))
        step = SimpleNamespace(schedule=schedule, embedding_token_ids=(42,) * schedule.token_count,
                               model_source_inputs=source, kv_cache_events=())
        plan = prepare_scheduled_step(execution, step, (2,))
        specs = tuple(call.kernel_spec for call in plan.calls)
        assert any(spec.transfer is not None and spec.transfer.operation == 'all_gatherv'
                   and len(spec.transfer.participants) >= 2 for spec in specs)
        if model == 'deepseek-v4-flash':
            assert any(spec.attributes.get('mathematical_program') == 'v4_idle_swa_insert_v1'
                       and spec.attributes.get('valid_token_count') == 0 for spec in specs)
        if model == 'kimi-k3':
            assert any(spec.variant == 'kimi_k3_nope_decode' and spec.attributes['slot_mapping'] == (-1,)
                       for spec in specs)
        predictions = execution.predict_local_plan(plan)
        assert predictions and all(math.isfinite(p.t_total_s) and p.t_total_s >= 0
                                   for p in predictions.values())
        for proof in (k3_dp_contract._validated_variables, k3_sp_contract._validated_variables,
                      k3_tp_contract._PROOF, v3_dp_contract._PROOF,
                      v4_dp_contract._validated_variables, v4_modelsp_contract._validated_variables):
            assert proof.get() is None
