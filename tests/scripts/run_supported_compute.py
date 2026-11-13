"""Probe the branch's real compute entry; native DP/PD run in the DS matrix.

Discovery is an explicit preflight, never an acceptance result. Unexpected errors
fail normal runs; accepted rejections must be frozen and reviewed in a manifest.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


def candidates():
    cases = []
    def add(model, hardware, *, entry='host', **options):
        cases.append(dict(model=model, hardware=hardware, options=options, entry=entry))
    devices = ('nvidia_a800', 'nvidia_h200', 'ascend_910b', 'metax_c500')
    for model in ('qwen3-4b', 'qwen3-8b', 'qwen3-32b', 'llama-3.1-8b-instruct'):
        for hardware in devices:
            add(model, hardware)
            if model.startswith('qwen'):
                add(model, hardware, tensor_parallel_size=2)
    for hardware in devices:
        add('qwen3-30b-a3b', hardware, tensor_parallel_size=2,
            enable_expert_parallel=True, expert_routing_seed=42)
    for hardware in ('intel_agmf039r47', 'nvidia_rtx_pro_5500', 'samsung_hbm_pim',
                     'groq_lpu', 'ucb_gemmini', 'hygon_bw1100'):
        add('qwen3-4b', hardware)
    for tp in (1, 2):
        add('qwen3-32b-fp16', 'nvidia_a800', tensor_parallel_size=tp, scheduler_adapter='vllm_0_11')
        for hardware in ('nvidia_a800', 'nvidia_h200'):
            add('qwen3-8b', hardware, tensor_parallel_size=tp, pipeline_parallel_size=2,
                scheduler_adapter='vllm_0_30')
    add('qwen3-8b', 'nvidia_h200', tensor_parallel_size=8, scheduler_adapter='vllm_0_30')
    for model, tp, ep in (
        ('glm-5.2', 8, False), ('glm-5.2-fp8', 8, False), ('glm-5.3', 8, False),
        ('kimi-k2.6', 8, True), ('kimi-k2.7-code', 8, True), ('kimi-k3', 1, False),
        ('deepseek-v3', 8, False), ('deepseek-v4-flash', 4, False), ('deepseek-v4-pro', 8, False),
        ('deepseek-v3-l32', 1, False), ('deepseek-v4-flash-dspark', 1, False),
        ('deepseek-v4-pro-dspark', 1, False)):
        add(model, 'nvidia_h200', tensor_parallel_size=tp, enable_expert_parallel=ep,
            expert_routing_seed=42)
    for hardware in devices:
        add('qwen3-30b-a3b', hardware, expert_routing_seed=42)
    add('glm-5.2', 'nvidia_a800', tensor_parallel_size=8, pipeline_parallel_size=4,
        scheduler_adapter='vllm_0_30', expert_routing_seed=42)
    for hardware in ('ascend_910b', 'metax_c500'):
        for model in ('qwen3-4b', 'qwen3-8b', 'qwen3-32b', 'qwen3-30b-a3b'):
            add(model, hardware, entry='upstream_vendor', tensor_parallel_size=2,
                enable_expert_parallel=model=='qwen3-30b-a3b')
    for model in ('glm-5.2', 'glm-5.2-fp8', 'glm-5.3'):
        add(model, 'nvidia_h200', entry='upstream_glm', tensor_parallel_size=8)
    return cases


def upstream_context(root, case, mode):
    """Explicit public source API fixtures, separate from host-entry support.

The host currently wires NVLink for every multi-rank deployment and omits
single-replica GLM declarations. These fixtures test supported imported modules;
they do not repair the host wrapper or turn its rejection into a success.
"""
    from src.kernel.hardware_spec import HWSpec
    from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, WorkerPoolConfig
    from src.serving.deployment.deployment_plan import build_deployment_plan
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.serving.workload.model_config import load_model_config
    from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs
    from src.serving.execution import ServingExecutionContext
    model = load_model_config(root/f"configs/model/{case['model']}.json")
    hardware = HWSpec(root/f"configs/hardware/{case['hardware']}.yaml")
    parallel = ParallelConfig(**case['options'])
    link = {'ascend_910b': 'hccs', 'metax_c500': 'metaxlink'}.get(case['hardware'], 'nvlink')
    collectives = tuple(CollectiveConfig(op, 'ring', link, 'ring', 'source-fabric', 'analytical_roofline_v1')
                        for op in ('all_reduce', 'all_gather'))
    deployment = build_deployment_plan(DeploymentConfig('colocated', (
        WorkerPoolConfig('main', 'colocated', case['hardware'], parallel,
                         ranks=tuple(range(parallel.worker_count)), collectives=collectives),), ()),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    routing = (ExpertRoutingInput.seeded_assumption('structure-explicit-source-fixture', 42)
               if model.num_experts else ExpertRoutingInput.not_applicable())
    if case['entry']=='upstream_vendor':
        version = '0_18' if case['hardware']=='ascend_910b' else '0_24'
        inputs = RuntimeInputs(routing, 'paged_varlen', 'uniform',
            model_execution_version=f'vllm_{version}_0', target_sampling_source=f'vllm_{version}_plain_greedy_bf16')
    else:
        inputs = RuntimeInputs(routing, 'mla', 'uniform', model_execution_version='vllm_0_30_0',
            target_sampling_source='vllm_0_30_plain_greedy_bf16',
            symbolic_forward_source=f'vllm_0_30_v32_{model.weight_dtype}_sync_explicit_allreduce',
            symbolic_source_max_model_len=128, symbolic_source_max_batched_tokens=32,
            glm_indexer_selection_source='declared_causal_topk',
            symbolic_workspace_initial_capacity_bytes=(0,)*parallel.worker_count)
    return ServingExecutionContext(model, deployment, {case['hardware']: hardware},
        'analytical_launch', mode, inputs, execution_mode='accelerated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python011', type=Path, required=True,
                        help='Recorded native PD interpreter; native execution is a separate gate')
    parser.add_argument('--discover', action='store_true')
    parser.add_argument('--expectations', type=Path)
    parser.add_argument('--cases', type=int, nargs='+', help='Preflight subset; never full acceptance')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from astermind.common.bootstrap import activate
    root = activate()
    from astermind.engine.compute.runtime import MidlayerCellConfig, create_midlayer_runtime, prepare_scheduled_step
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    from tests.scripts.probe_compute_source import canonical
    args.output.mkdir(parents=True, exist_ok=True)
    expected = json.loads(args.expectations.read_text()) if args.expectations else {}
    rows, errors = [], []
    for index, case in enumerate(candidates()):
        if args.cases and index not in args.cases:
            continue
        for mode in ('midlayer', 'roofline'):
            key = f'{index:03d}-{mode}'
            row = dict(id=key, **case, compute_mode=mode,
                scope='host_compute_only' if case['entry']=='host' else 'upstream_compute_only')
            stage = 'construct'
            try:
                if case['entry']=='host':
                    _, execution, _ = create_midlayer_runtime(MidlayerCellConfig(root,
                        model_file=f"configs/model/{case['model']}.json",
                        hardware_file=f"configs/hardware/{case['hardware']}.yaml",
                        hardware_id=case['hardware'], compute_mode=mode, **case['options']), with_scheduler=False)
                else:
                    execution = upstream_context(root, case, mode)
                observations = []
                for phase, computed in (('prefill', 0), ('decode', 2)):
                    stage = phase+'-plan'
                    schedule = ScheduleStep('support-'+phase, 'main', (
                        ScheduledRequest('r', phase, 2 if phase=='prefill' else 0,
                            0 if phase=='prefill' else 1, computed_tokens=computed),))
                    step = SimpleNamespace(schedule=schedule, embedding_token_ids=(42,)*schedule.token_count,
                        model_source_inputs=None, kv_cache_events=())
                    plan = prepare_scheduled_step(execution, step, (2,))
                    assert plan.calls
                    stage = phase+'-prediction'
                    predictions = execution.predict_local_plan(plan)
                    assert predictions and all(math.isfinite(p.t_total_s) and p.t_total_s >= 0 for p in predictions.values())
                    payload = dict(calls=canonical(tuple(plan.calls)),
                        predictions={k:canonical(v) for k,v in predictions.items()})
                    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    observations.append(dict(phase=phase, calls=len(plan.calls), predictions=len(predictions),
                        logical_calls=plan.logical_call_count, exact_output_sha256=digest,
                        total_prediction_s=sum(p.t_total_s for p in predictions.values())))
                row.update(status='passed', observations=observations)
                if key in expected:
                    raise AssertionError('previous explicit rejection now succeeds; review support boundary')
            except Exception as error:
                failure = dict(stage=stage, type=type(error).__name__, message=str(error))
                row.update(status='rejected' if expected.get(key)==failure else 'unexpected_error', error=failure)
                if row['status'] == 'unexpected_error':
                    errors.append(row)
            rows.append(row)
            (args.output/(key+'.json')).write_text(json.dumps(row, sort_keys=True, indent=2)+'\n')
            print(json.dumps(dict(id=key, model=case['model'], hardware=case['hardware'], status=row['status'],
                                  stage=stage)), flush=True)
    if not args.cases and not args.discover:
        assert set(expected)=={r['id'] for r in rows if r['status']=='rejected'}, 'Negative contract inventory changed'
    summary = dict(discovery=args.discover, complete=not errors and not args.discover and not args.cases, cases=rows,
        positive_cases=sum(r['status']=='passed' for r in rows),
        negative_contracts=sum(r['status']=='rejected' for r in rows))
    (args.output/'summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2)+'\n')
    if errors or args.discover:
        raise SystemExit('Discovery/unreviewed errors are not acceptance; inspect summary.json')


if __name__ == '__main__':
    main()
