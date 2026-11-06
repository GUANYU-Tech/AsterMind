"""Experimental Midlayer integration on the OrbitServe event and network clock."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AsterMindCell:
    config: object
    rank_mapping: dict
    repository: object = None


class _HostSimulator:
    def __init__(self, engine, service_job):
        self.engine = engine
        self.job = service_job

    @property
    def now_ns(self):
        return self.engine.now * 1e9

    def schedule(self, delay_ns, callback, priority=0):
        self.engine.push(self.engine.now + delay_ns / 1e9, 30 + priority,
                         'external_callback', payload=callback)

    def start_flow(self, flow_id, source, target, size_bytes, callback, is_control=False):
        self.engine.log.append({'time_s': self.engine.now, 'kind': 'astermind_flow',
                                'flow_id': flow_id, 'source_node_id': str(source),
                                'target_node_id': str(target), 'bytes': size_bytes,
                                'is_control': is_control})
        self.engine.send(self.job, str(source), str(target), float(size_bytes), None,
                         'astermind_collective', 'external_flow_done',
                         context={'callback': callback, 'astermind_flow_id': flow_id})


class AsterMindRuntime:
    """Host one AsterMind deployment per selected compute-node identifier.

    This opt-in bridge currently accepts independent requests only. It aborts
    rather than silently approximating cache reuse, failed cells, or cancelled
    in-flight serving work. It is intended for integration experiments.
    """

    def __init__(self, cells, *, emit=None, record_packets=False):
        self.cells = dict(cells)
        self.emit = emit or (lambda node, kind, row: None)
        self.sessions = {}
        self.jobs = {}
        self.engine = None
        self.record_packets = record_packets

    def initialize(self, engine, requests):
        from astermind.embedded import EmbeddedSession
        from midlayer.serving.request_gen.request_stream import RequestSpec, RequestStream

        if engine.net.spec.transport_model not in {'flowsim', 'ns3'}:
            raise ValueError('AsterMind integration requires FlowSim or ns-3 transport')
        if engine.spec.strategy.shadow_dvfs_enabled:
            raise NotImplementedError('AsterMind integration does not model OrbitServe DVFS')
        if not requests:
            return
        if any(r.get('context_tokens_before_turn', 0) or r.get('depends_on_request_id')
               or r.get('output_tokens', 0) < 1 for r in requests):
            raise NotImplementedError('embedded mode currently supports independent requests with output tokens')
        self.engine = engine
        stream = RequestStream(RequestSpec(max(0., r['arrival_time_s']), r['request_id'],
                                           r['prompt_tokens'], r['output_tokens']) for r in requests)
        used_members = set()
        for node, cell in self.cells.items():
            if node not in engine.eligible:
                raise ValueError(f'AsterMind cell target is not eligible: {node}')
            if any(str(value) not in engine.net.nodes for value in cell.rank_mapping.values()):
                raise ValueError(f'AsterMind rank mapping contains unknown nodes: {node}')
            members = {str(value) for value in cell.rank_mapping.values()}
            if members != set(engine.members(node)) or used_members.intersection(members):
                raise ValueError(f'AsterMind mapping must cover one distinct physical cell: {node}')
            used_members.update(members)
            service_id = f'__astermind_cell__/{node}'
            if service_id in engine.jobs:
                raise ValueError(f'reserved request id: {service_id}')
            service = {'request': {'request_id': service_id}, 'done': False,
                       'path': [], 'ledger': [], 'external_service': True}
            engine.service_jobs[service_id] = service
            self.sessions[node] = EmbeddedSession(
                cell.config, stream, simulator=_HostSimulator(engine, service),
                rank_mapping=cell.rank_mapping, namespace=service_id,
                emit=lambda kind, row, node=node: self.emit(node, kind, row),
                on_tokens=self._tokens, on_complete=self._complete,
                repository=cell.repository,
            )

    def submit(self, engine, job):
        node, request = job['node'], job['request']
        if node not in self.sessions:
            raise ValueError(f'no AsterMind deployment for selected node: {node}')
        if not engine.cell_available(node):
            raise ValueError('compute_cell_disconnected_during_service')
        job.update(local_hit_tokens=0, effective_reuse_tokens=0,
                   recompute_tokens=request['prompt_tokens'], copied_from=None,
                   copied_tokens=0, compute_ready_s=engine.now, prefill_start_s=engine.now,
                   prefill_done_s=engine.now, prefill_service_s=0., decode_service_s=0.,
                   decode_done_s=engine.end,
                   concurrency=1 + sum(not other['done'] and other['node'] == node
                                       for other in self.jobs.values()), dvfs=1.,
                   window={'sunlit': True}, soc_before=engine.state.satellite_for(node).soc,
                   physical_members=engine.members(node))
        # Use OrbitServe's existing per-token energy estimate at admission;
        # Midlayer execution timing does not itself model energy.
        members = job['physical_members']
        updates = {member: engine.debit(
            job, member, prompt=job['recompute_tokens'] / len(members),
            output=request['output_tokens'] / len(members), stage='compute',
        ) for member in members}
        job['soc_before'] = updates[node]['soc_before']
        self.jobs[request['request_id']] = job
        engine.active_compute[request['request_id']] = job
        self.sessions[node].submit(request['request_id'])

    def candidate_load(self, node):
        session = self.sessions.get(node)
        if session is None:
            return 1.0
        outstanding = len(session.submitted - session.runtime.finished)
        return outstanding / (outstanding + 1)

    def _tokens(self, request_id, count, time_s):
        # The host clock is authoritative; avoid an ns -> s float round trip.
        time_s = self.engine.now
        job = self.jobs[request_id]
        if job['done']:
            raise RuntimeError('AsterMind produced tokens for a failed request')
        if not job.get('astermind_first_token'):
            self.engine.log.append({'time_s':time_s, 'kind':'astermind_first_token',
                                    'request_id':request_id, 'compute_node_id':job['node']})
            job['astermind_first_token'] = True
            job['prefill_done_s'] = time_s
            job['prefill_service_s'] = time_s - job['prefill_start_s']
            self.engine.on_first_token_ready(job, None)

    def _complete(self, request_id, time_s):
        time_s = self.engine.now
        job = self.jobs[request_id]
        if job['done'] or not job.get('astermind_first_token'):
            raise RuntimeError('invalid AsterMind completion')
        job['decode_done_s'] = time_s
        self.engine.log.append({'time_s':time_s, 'kind':'astermind_compute_complete',
                                'request_id':request_id, 'compute_node_id':job['node']})
        job['decode_service_s'] = time_s - job['prefill_done_s']
        self.engine.on_compute_complete(job, None)

    def close(self):
        for session in self.sessions.values():
            session.close()
