"""Narrow launch adaptation; original workload/predictor/timeline code is reused.

M-L 98d207a labels a one-query mixed batch as prefill, but its SM80 FA2
source-order model requires the single-query launch grid. Select the original
decode attention launch for this geometry; never split/reorder the batch or
alter tensor shapes, KV lengths, dependencies, or kernel cost formulas.
"""
from dataclasses import replace
import time

from src.serving.execution import ServingExecutionContext, _runtime_graph_mode
from src.serving.workload.build_workload import build_workload
from src.serving.workload.workload_graph import WorkloadGraph, WorkloadKind


class CellExecutionContext(ServingExecutionContext):
    def execute_step(self, schedule, *, kv_cache_events=(), origin_time_s=0.0,
                     shared_resource_calendar=None, runtime_prepare_events=None):
        if not all(r.scheduled_tokens == 1 for r in schedule.requests):
            return super().execute_step(schedule, kv_cache_events=kv_cache_events,
                                        origin_time_s=origin_time_s,
                                        shared_resource_calendar=shared_resource_calendar,
                                        runtime_prepare_events=runtime_prepare_events)
        started = time.perf_counter()
        graph = build_workload(schedule, self.model, self.deployment,
                               launch_policy=self.launch_policy, kv_cache_events=kv_cache_events,
                               runtime_inputs=self.runtime_inputs,
                               routing_memo=self._routing_memo, block_memo=self._block_memo)
        self._phase_wall_time_s["workload_construction"] += time.perf_counter() - started
        changed = [n.node_id for n in graph.nodes if n.kind == WorkloadKind.FLASH_ATTN and n.variant == "prefill"]
        if changed:
            graph = WorkloadGraph((replace(n, variant="decode") if n.node_id in changed else n for n in graph.nodes),
                                  graph.edges, graph.block_repeats)
            if not hasattr(self, "launch_adaptations"):
                self.launch_adaptations = []
            self.launch_adaptations.append({"step_id": schedule.step_id,
                                           "reason": "single_query_attention_launch",
                                           "changed_node_ids": changed})
        return self.execute_workload(graph, origin_time_s=origin_time_s,
                                     shared_resource_calendar=shared_resource_calendar,
                                     runtime_graph_mode=_runtime_graph_mode(schedule, self.runtime_inputs.graph_execution),
                                     runtime_prepare_events=runtime_prepare_events)
