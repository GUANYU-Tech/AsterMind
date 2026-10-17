"""Local evidence for the adapter contract; no hardware execution."""
import json
from simulator.bootstrap import activate


def main():
    root = activate()
    from midlayer_cell import MidlayerCellConfig, create_midlayer_runtime
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    from src.serving.workload.build_workload import build_workload
    _, execution = create_midlayer_runtime(MidlayerCellConfig(root), with_scheduler=False)
    graph = build_workload(ScheduleStep("audit", "main", (
        ScheduledRequest("p", "prefill", 4, 0, computed_tokens=8),
        ScheduledRequest("d", "decode", 0, 1, computed_tokens=32))),
        execution.model, execution.deployment, launch_policy=execution.launch_policy,
        runtime_inputs=execution.runtime_inputs)
    nodes = []
    for node in graph.nodes[:45]:
        nodes.append(dict(id=node.node_id, kind=node.kind.value, shape=dict(node.shape),
            variant=node.variant, rank=node.rank, attributes=dict(node.attributes)))
    print(json.dumps(dict(nodes=nodes, repeats=[r.node_ids[:3] for r in graph.block_repeats]), default=str, indent=2))


if __name__ == "__main__":
    main()
