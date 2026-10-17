"""All serving and model transfers use the executor's one network pump."""
from network_sim import TransferRequest


def shared_executor(loop, network, placements):
    from network_batch_executor import NetworkBatchExecutor, nanoseconds

    class SharedExecutor(NetworkBatchExecutor):
        def start_batch(self, cell_id, step_id, components, execution, on_complete, **kwargs):
            def finished(at_s):
                on_complete(at_s)
                for notify in tuple(self.boundary_listeners):
                    notify()
            def start():
                return super(SharedExecutor, self).start_batch(cell_id, step_id, components, execution, finished, **kwargs)
            pool_name, pool = next(iter(execution.deployment.worker_pools.items()))
            participants = pool.physical_ranks_for_pipeline_stage(0)
            if len(participants) > 1 and not execution.input_distribution_in_graph:
                count = execution.last_step_token_count
                if type(count) is not int or count <= 0:
                    raise ValueError("native input distribution requires actual scheduled token count")
                record = dict(cell_id=cell_id, step_id=step_id, ready_ns=nanoseconds(loop.now_s), phases=[])
                self.input_distributions.append(record)
                def distributed():
                    record["completed_ns"] = nanoseconds(loop.now_s)
                    start()
                self._phase(cell_id, pool_name, "input_token_ids", f"{cell_id}/{step_id}/input",
                    [(participants[0], rank, count*8) for rank in participants[1:]], distributed, record)
            else:
                start()

        def submit_transport(self, identifier, source, target, size_bytes, operation,
                             correlation_id, on_complete, on_failure=None):
            if identifier in self._callbacks or identifier in self._transport_failures:
                raise ValueError("duplicate active transport")
            if source == target:
                # Ingress/output already at the consuming device: no network copy.
                on_complete()
                return
            request = TransferRequest(id=identifier, source=source, target=target,
                size_bytes=size_bytes, ready_time_ns=nanoseconds(loop.now_s),
                origin="orbitserve", operation=operation, correlation_id=correlation_id)
            network.submit(request)
            self._callbacks[identifier] = on_complete
            if on_failure is not None:
                self._transport_failures[identifier] = on_failure
            self._arm_network()

        def _pump(self, generation, next_ns):
            from network_batch_executor import CommunicationFailed
            if generation != self._generation:
                return
            for event in network.advance_to(next_ns):
                self.transfer_events.append(event.model_dump())
                callback = self._callbacks.pop(event.id)
                failure = self._transport_failures.pop(event.id, None)
                if event.kind == "failed":
                    if failure is None:
                        raise CommunicationFailed(event)
                    failure(event)
                else:
                    if event.completion_time_ns != next_ns:
                        raise RuntimeError("network event delivered at the wrong time")
                    callback()
            self._arm_network()

    executor = SharedExecutor(loop, network, placements)
    executor._transport_failures = {}
    executor.boundary_listeners = []
    executor.input_distributions = []
    return executor
