"""One Kimi Cell with two native schedulers and an external KV handoff."""

from dataclasses import replace

from astermind.engine.cell_events import CellCompletion
from astermind.engine.timeline import nanoseconds
from src.serving.workload.kv_transfer_workload import build_pd_kv_transfer_workload

from .serving import ServingScheduler as MidlayerCell
from .runtime import create_midlayer_runtime


class PrefillDecodeCell:
    def __init__(self, cell_id, loop, config, executor, on_complete, on_output, on_failure, *,
                 submit_handoff):
        backends, self.execution, _ = create_midlayer_runtime(config)
        self.cell_id, self.loop, self.executor = cell_id, loop, executor
        self.on_complete, self.on_output, self.on_failure = on_complete, on_output, on_failure
        self.submit_handoff = submit_handoff
        self.requests, self.prefill_completions, self.handoffs = {}, {}, {}
        self.transferring = set()
        try:
            self.prefill = MidlayerCell(cell_id, loop, backends["prefill"], self.execution,
                self._prefill_complete, on_output=self._prefill_output,
                max_model_len=config.max_model_len, batch_executor=executor,on_failure=on_failure)
            self.decode = MidlayerCell(cell_id, loop, backends["decode"], self.execution,
                self._decode_complete, on_output=self._decode_output,
                max_model_len=config.max_model_len, batch_executor=executor,on_failure=on_failure)
        except BaseException:
            for backend in backends.values():
                backend.close()
            raise

    def submit_ready_request(self, request):
        if request.session_id is not None or request.prefix_hit_tokens:
            raise NotImplementedError("PD smoke requires independent requests without prefix reuse")
        self.prefill.submit_ready_request(request)
        self.requests[request.request_id] = request

    def _prefill_output(self, output):
        self.on_output(replace(output, is_final=self.requests[output.request_id].output_tokens == 1))

    def _decode_output(self, output):
        self.on_output(replace(output, token_start=output.token_start + 1))

    def _prefill_complete(self, completion):
        request_id = completion.request_id
        request = self.requests[request_id]
        if completion.generated_tokens != 1:
            raise RuntimeError("prefill must produce exactly one seed token")
        if request.output_tokens == 1:
            self.on_complete(completion)
            return
        self.prefill_completions[request_id] = completion
        workload = build_pd_kv_transfer_workload(request_id, request.input_tokens,
            self.execution.model, self.execution.deployment, "prefill", "decode")
        partitions = []
        for node in workload.nodes:
            attrs = node.attributes
            source = self.executor.placements[(self.cell_id, "prefill", attrs["source_rank"])].gpu_device_id
            target = self.executor.placements[(self.cell_id, "decode", attrs["destination_rank"])].gpu_device_id
            partitions.append(dict(id=f"{self.cell_id}/{node.node_id}", source=source,
                                   target=target, bytes=attrs["bytes"]))
        record = dict(start_time_ticks=str(self.loop.now_s.ticks),start_time_ns=nanoseconds(self.loop.now_s), partitions=partitions,
                      end_time_ns=None, failures=[])
        self.handoffs[request_id] = record
        if not self.transferring:
            self.prefill.pause_for_migration("pd_kv")
        self.transferring.add(request_id)
        remaining = {part["id"] for part in partitions}

        def terminal(part_id, failure=None):
            remaining.remove(part_id)
            if failure is not None:
                record["failures"].append(failure.model_dump())
            if remaining:
                return
            record["end_time_ns"] = nanoseconds(self.loop.now_s)
            record["end_time_ticks"] = str(self.loop.now_s.ticks)
            self.transferring.remove(request_id)
            if not self.transferring:
                self.prefill.resume_after_migration("pd_kv")
            if record["failures"]:
                self.prefill_completions.pop(request_id)
                self.on_failure(dict(request_id=request_id, reason="pd_kv_failed", handoff=record))
                return
            self.decode.submit_ready_request(replace(request, cell_ready_time_s=self.loop.now_s))

        for part in partitions:
            part_id = part["id"]
            self.submit_handoff(part_id, part["source"], part["target"], part["bytes"],
                "pd_kv", request_id, lambda part_id=part_id: terminal(part_id),
                lambda event, part_id=part_id: terminal(part_id, event))

    def _decode_complete(self, completion):
        prefill = self.prefill_completions.pop(completion.request_id)
        generated = prefill.generated_tokens + completion.generated_tokens
        if generated != self.requests[completion.request_id].output_tokens:
            raise RuntimeError("PD output token count disagrees with request")
        self.on_complete(CellCompletion(self.cell_id, completion.request_id, prefill.cell_ready_time_s,
            prefill.first_execution_s, completion.completion_time_s, prefill.first_token_time_s,
            completion.last_token_time_s, generated, prefill.prefill_active_s, completion.decode_active_s))

    def report(self):
        pools = {name: cell.report() for name, cell in (("prefill", self.prefill), ("decode", self.decode))}
        return dict(pools=pools, steps=sorted(self.prefill.steps + self.decode.steps,
                    key=lambda step: step["start_time_s"]), pd_handoffs=self.handoffs)

    def close(self):
        try:
            self.prefill.close()
        finally:
            self.decode.close()
