"""Bounded, actual-event context for the live 3D inspector."""

from copy import deepcopy


def fault_key(event):
    return tuple(event.get(key) for key in ('target_type', 'target_id', 'source', 'start_s', 'end_s'))


class LiveReplayWindow:
    def __init__(self, limit=300):
        self.limit = limit
        self.cursor = 0
        self.start = 0
        self.frames = []
        self.flows = {}
        self.plans = {}
        self.recoveries = {}
        self.losses = {}
        self.faults = {}

    def observe(self, event):
        kind, request = event['kind'], event.get('request_id')
        if kind=='v1_admit':
            for leg in event.get('path_legs',[]):
                self.flows[leg['transfer_id']]={**event,**leg,'status':'transmitting','v1_flow_id':event['id']}
        elif kind in {'v1_p2p_complete','v1_retry','v1_failed'}:
            self.flows={k:v for k,v in self.flows.items() if v.get('v1_flow_id')!=event.get('id')}
        identity = event.get('transfer_id')
        status = {'transfer_queued': 'queued', 'transmit_start': 'transmitting',
                  'transmit_complete': 'propagating'}.get(kind)
        if status:
            self.flows[identity] = {**self.flows.get(identity, {}), **event, 'status': status}
        elif kind in {'hop_arrival', 'transfer_cancelled'}:
            self.flows.pop(identity, None)
        elif kind == 'bandwidth_update' and identity in self.flows:
            self.flows[identity]['sent_bytes'] = self.flows[identity].get('sent_bytes', 0) + event.get('bytes_sent', 0)
            self.flows[identity].update(event)
        if kind == 'route_plan':
            self.plans[(request, event.get('stage'), event.get('reason'))] = event
        elif kind == 'route_recovery':
            self.recoveries[request] = event
        elif kind == 'transfer_cancelled':
            self.losses[request] = event
        elif kind == 'fault_injected':
            self.faults[fault_key(event)] = event
        elif kind == 'fault_restored':
            self.faults.pop(fault_key(event), None)
        if kind == 'cell_transfer_terminal' and event.get('id'):
            request = '__cell_transfer__/' + event['id']
        if kind in {'request_complete', 'request_failed', 'cell_transfer_terminal'}:
            self.flows = {key: flow for key, flow in self.flows.items() if flow.get('request_id') != request}
            self.plans = {key: plan for key, plan in self.plans.items() if key[0] != request}
            self.recoveries.pop(request, None)
            self.losses.pop(request, None)

    def capture(self, log, snapshot, preserve_from=None):
        # Keep recent observed frames and log order, including multiple manual
        # changes at one boundary. Prefix reduction scans each old event once.
        if self.frames and self.frames[-1].get('event_order') == len(log) - 1:
            self.frames.pop()
        # Event stepping often leaves physical geometry unchanged. Keep its
        # original time/order anchor instead of shipping duplicate large frames.
        if (preserve_from is None or not self.frames or self.frames[-1]['nodes'] != snapshot['nodes']
                or self.frames[-1]['links'] != snapshot['links']):
            self.frames.append(dict(snapshot, event_order=len(log) - 1))
        self.frames = self.frames[-8:]
        start = max(self.start, len(log) - self.limit)
        if preserve_from is not None:
            start = max(self.start, min(start, preserve_from))
        while start < len(log) and log[start]['time_s'] < self.frames[0]['time_s']:
            start += 1
        self.start = start
        for event in log[self.cursor:start]:
            self.observe(event)
        self.cursor = start
        context = dict(active_transfers=list(self.flows.values()), route_plans=list(self.plans.values()),
                       recovery_events=list(self.recoveries.values()), loss_events=list(self.losses.values()),
                       faults=list(self.faults.values()))
        from event_catalog import annotate
        return deepcopy(dict(initial=context, frames=self.frames,
                             events=[annotate(dict(event, event_order=index)) for index, event in enumerate(log[start:], start)],
                             first_event_order=start, position_semantics='sample_hold'))
