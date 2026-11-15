"""Native FlowSim max-min allocation over each P2P's complete external path.

Only external resources compete. GPU-to-terminal access is omitted;
intra-only traffic retains surrogate timing. GlobalTimeline owns all events.
"""
from dataclasses import asdict,dataclass
from math import isfinite
from astermind.engine.clock import Time
from astermind.engine.timeline import nanoseconds
from time import perf_counter
from .flow_allocator import FlowAllocator
from .frame_surrogate import NoRouteError,CommunicationRouting
from .planner import plan_communication
from .operator import OperatorEstimate
from .path_budget import path_budget, resource_capacities


@dataclass(frozen=True)
class CommunicationFailure:
    id: str
    reason: str
    failure_time_ns: int
    kind: str='failed'
    def model_dump(self):return asdict(self)


class TickTrace(list):
    def __init__(self, loop):
        self.loop = loop
    def append(self, row):
        super().append(dict(row, time_ticks=str(self.loop.now_s.ticks)))


class RuntimeCommunication(CommunicationRouting):
    def __init__(self,network,loop,policy,allocator=None):
        self.network,self.loop,self.policy=network,loop,policy
        self.allocator=allocator or FlowAllocator()
        self.tasks={};self.flows={};self.active={};self.capacities={};self.rates={}
        self.last_time=self.loop.now_s;self.last_ns=self.now_ns;self.serial=0;self.generation=0;self.trace=TickTrace(loop)
        self.closed=False;self.routes=None;self.frame_deadline=None
        self.delivery_paths={};self.used_ids=set()
        self.stats=dict(peak_active_flows=0,max_path_links=0,rate_allocations=0,
            allocation_wall_s=0.,route_wall_s=0.,retries=0,discarded_application_bytes=0.,
            cell_estimate_calls=0,cell_estimate_wall_s=0.,
            cell_estimate_context_calls=0,cell_estimate_context_wall_s=0.)

    @property
    def now_ns(self):return self.loop.now_s.ticks/10**9

    def _later(self,delay_ns,callback,kind='comm:event',priority=1):
        if not isfinite(delay_ns) or delay_ns<0:raise ValueError('invalid communication delay')
        at=self.loop.now_s+Time.from_ns(delay_ns)
        self.loop.schedule(at,priority,kind,callback)

    def _refresh(self,force=False):
        if self.routes is not None and not force:
            valid_until=self.routes.frame.valid_until_s
            if valid_until is None or self.loop.now_s<valid_until:return self.routes
        frame=self.network.frame_at(self.loop.now_s);self.routes=self.network.routes(frame)
        self.capacities=resource_capacities(self.routes,self.policy)
        deadline=frame.valid_until_s
        if deadline is not None and deadline>self.loop.now_s and deadline!=self.frame_deadline:
            self.frame_deadline=deadline
            def update():
                if self.closed or not self.tasks:return
                self.frame_deadline=None;self.network_changed()
            self.loop.schedule(deadline,0,'comm:network_frame',update)
        return self.routes

    def _valid(self,f,attempt):
        return self.flows.get(f['id']) is f and f['attempt']==attempt and f['task'] in self.tasks

    def _settle(self):
        dt=max(0.,(self.loop.now_s.ticks-self.last_time.ticks)/10**9);finished=[]
        for identity,f in tuple(self.active.items()):
            rate=self.rates.get(identity,0.)
            sent=min(f['remaining'],rate*dt);f['remaining']-=sent;f['sent']+=sent
            if sent and self.policy.trace_service:
                self.trace.append(dict(kind='service',id=identity,task=f['task'],attempt_id=f['attempt'],
                    start_ns=self.last_ns,end_ns=self.now_ns,start_ticks=str(self.last_time.ticks),
                    end_ticks=str(self.loop.now_s.ticks),application_bytes=sent,rate_gbps=rate*8,
                    resources=[str(k) for k in f['budget']['demand']]))
            if rate>0 and f['remaining']<=max(1e-9,f['size']*1e-12):finished.append(f)
        self.last_time=self.loop.now_s
        self.last_ns=self.now_ns
        for f in finished:
            self.active.pop(f['id'],None);f['state']='tail';attempt=f['attempt']
            self._later(f['budget']['tail'],lambda f=f,a=attempt:self._arrive(f,a),'comm:arrival')

    def _rebalance(self):
        self.generation+=1;generation=self.generation
        demands={i:dict(f['budget']['demand']) for i,f in self.active.items()}
        capacities=dict(self.capacities)
        started=perf_counter();self.rates=self.allocator.rates(demands,capacities)
        self.stats['allocation_wall_s']+=perf_counter()-started;self.stats['rate_allocations']+=1
        if self.policy.trace_service:
            self.trace.append(dict(kind='allocation',capacities={str(k):v for k,v in capacities.items()},
                flows={i:dict(rate_bytes_per_ns=self.rates[i],demand={str(k):v for k,v in d.items()})
                       for i,d in demands.items()}))
        self.stats['peak_active_flows']=max(self.stats['peak_active_flows'],len(self.active))
        waits=[f['remaining']/self.rates[i] for i,f in self.active.items() if self.rates[i]>0]
        if waits:
            def wake():
                if generation!=self.generation or self.closed:return
                self._settle();self._rebalance()
            self._later(max(min(waits),1e-6),wake,'comm:serialization_complete')

    def _arrive(self,f,attempt):
        if not self._valid(f,attempt):return
        self.flows.pop(f['id']);f['state']='completed'
        self.delivery_paths[f['task']]=tuple(f['path'])
        self.trace.append(dict(kind='p2p_complete',id=f['id'],task=f['task'],attempt_id=attempt,time_ns=self.now_ns))
        f['done']()

    def _attempt(self,f):
        if f['task'] not in self.tasks:return
        routes=self._refresh();started=perf_counter()
        first_hop=self.tasks[f['task']].get('first_hop')
        try:route=routes.route(f['source'],f['target'],f['size'],first_hop=first_hop)
        except NoRouteError:
            f['state']='waiting'
            self.trace.append(dict(kind='route_wait',id=f['id'],task=f['task'],time_ns=self.now_ns));return
        finally:self.stats['route_wall_s']+=perf_counter()-started
        f['attempt']+=1;attempt=f['attempt'];f['remaining']=float(f['size']);f['sent']=0.
        f['path_keys']=tuple(route['resources'])
        f['path']=tuple(route['path'])
        self.stats['max_path_links']=max(self.stats['max_path_links'],len(route['resources']))
        same=routes.topology.devices[f['source']].satellite==routes.topology.devices[f['target']].satellite
        if same and all(e['source_owner']==e['target_owner'] for e in routes.route_resources(route)):
            f['state']='local'
            if all(routes.topology.devices[n].kind=='gpu' for n in (f['source'],f['target'])):
                from .intra_topology import p2p_ns
                from astermind.engine.comm.intra_hierarchy import classify_participants
                _, hierarchy, _, _ = classify_participants(routes.frame, (f['source'], f['target']))
                duration=p2p_ns(routes,route,f['size'],hierarchy)
            else:duration=8*f['size']/route['bandwidth_gbps']+route['latency_ns']
            if self.policy.trace_service:
                self.trace.append(dict(kind='intra_service',id=f['id'],task=f['task'],path=route['path'],
                    bandwidth_gbps=route['bandwidth_gbps'],latency_ns=route['latency_ns'],duration_ns=duration,
                    intra_competition=False))
            self._later(duration,lambda:self._arrive(f,attempt),'comm:intra_complete');return
        f['budget']=path_budget(routes,route,f['size'],self.policy);f['state']='startup'
        def admit():
            if not self._valid(f,attempt):return
            self._settle()
            if any(k not in self.capacities for k in f['budget']['demand']):
                self._fail(f['task'],'missing shared capacity pool');return
            f['state']='sending';self.active[f['id']]=f
            self.trace.append(dict(kind='admit',id=f['id'],task=f['task'],attempt_id=attempt,time_ns=self.now_ns,
                path=f['budget']['path'],budgets=f['budget']['budgets'],startup_ns=f['budget']['startup'],
                tail_ns=f['budget']['tail'],sampled_errors=False,ack_flows=False))
            self._rebalance()
        self._later(f['budget']['startup'],admit,'comm:admit')

    def _p2p(self,task,source,target,size,done):
        if not size or source==target:self._later(0,lambda:done() if task in self.tasks else None);return
        self.serial+=1;identity=f'{task}/p2p/{self.serial}'
        f=dict(id=identity,task=task,source=source,target=target,size=size,done=done,
               attempt=0,retries=0,sent=0.,state='new',path_keys=())
        self.flows[identity]=f;self._attempt(f)

    def network_changed(self):
        self._settle();routes=self._refresh(force=True)
        # Updating path validity never rewrites an in-flight attempt. Only
        # actual lost resources below trigger retry; capacity changes rebalance.
        host = getattr(self.network, 'host', None)
        cache = getattr(host.net, '_path_route_cache', None) if host is not None else None
        if cache is not None:
            cache.synchronize(host.net, host.now)
        for f in tuple(self.flows.values()):
            if f['task'] not in self.tasks:continue
            if f['state']=='waiting':self._attempt(f);continue
            if any(k not in routes.edges for k in f['path_keys']):
                self.active.pop(f['id'],None);f['attempt']+=1;f['retries']+=1
                self.stats['retries']+=1;self.stats['discarded_application_bytes']+=f['sent']
                self.trace.append(dict(kind='retry',id=f['id'],task=f['task'],time_ns=self.now_ns,
                    discarded_application_bytes=f['sent'],retry_count=f['retries']))
                if f['retries']>self.policy.retry_limit:self._fail(f['task'],'route retry limit exceeded')
                else:self._attempt(f)
            elif f['state'] in ('sending','startup'):
                route=dict(path=f['budget']['path'],resources=f['path_keys'])
                updated=path_budget(routes,route,f['size'],self.policy)
                if updated['demand']!=f['budget']['demand']:
                    self._fail(f['task'],'protocol/resource membership changed during active attempt')
                else:
                    # Sending/startup phases use the latest sampled propagation.
                    # Once the tail is scheduled, its arrival time is not rewritten.
                    f['budget']['tail']=updated['tail']
        self._rebalance()

    def _register(self,identity,on_failure,deadline_s):
        if self.closed:raise RuntimeError('communication runtime closed')
        if identity in self.used_ids:raise ValueError('duplicate communication ID')
        if deadline_s is None and self.policy.task_timeout_s is not None:deadline_s=self.loop.now_s+self.policy.task_timeout_s
        horizon=getattr(self.network,'horizon',None)
        horizon_cleanup = horizon is not None and (deadline_s is None or deadline_s > horizon)
        if horizon_cleanup:deadline_s=horizon
        if deadline_s is not None and deadline_s<self.loop.now_s:raise ValueError('deadline is in the past')
        self.tasks[identity]=dict(start=self.loop.now_s,failure=on_failure)
        self.used_ids.add(identity)
        if deadline_s is not None:
            token=self.tasks[identity]
            def expire():
                if self.tasks.get(identity) is token:self._fail(identity,'communication deadline exceeded')
            # The inclusive host horizon drains same-tick completions before cleanup.
            # Explicit task deadlines retain the upstream priority-zero policy.
            self.loop.schedule(deadline_s,100 if horizon_cleanup else 0,'comm:deadline',expire)

    def _fail(self,identity,reason):
        self._settle()
        task=self.tasks.pop(identity,None)
        if task is None:return
        for key,f in tuple(self.flows.items()):
            if f['task']==identity:self.flows.pop(key);self.active.pop(key,None)
        self._rebalance()
        failure=CommunicationFailure(identity,str(reason),nanoseconds(self.loop.now_s));self.trace.append(failure.model_dump())
        if task['failure']:task['failure'](failure)
        else:raise RuntimeError(f'communication {identity} failed: {reason}')

    def cancel(self,identity,reason='cancelled'):
        self._settle();self._fail(identity,reason)

    def submit(self,operator,on_complete,on_failure=None,*,task_id=None,deadline_s=None):
        self.serial+=1;identity=task_id or f'communication-{self.serial}'
        self._register(identity,on_failure,deadline_s)
        try:
            self._settle();routes=self._refresh()
            plan=plan_communication(self.network,self.policy,operator,self.loop.now_s,routes=routes)
            self.tasks[identity]['policy']=plan.policy
            self.trace.append(dict(kind='comm_plan',id=identity,time_ns=self.now_ns,**plan.selection,
                model=plan.estimate.mode,whole=not bool(plan.waves),model_inputs=plan.estimate.model_inputs))
            def complete():
                task=self.tasks.pop(identity,None)
                if task is None:return
                e=plan.estimate
                on_complete(OperatorEstimate(nanoseconds(self.loop.now_s-task['start']),e.rounds,e.flow_count,
                    e.cross_satellite_flows,e.cross_satellite_bytes))
            if not plan.waves:self._later(plan.estimate.duration_ns,complete,'comm:whole_intra_complete')
            else:
                def wave(index):
                    if identity not in self.tasks:return
                    if index==len(plan.waves):complete();return
                    pending=len(plan.waves[index])
                    if not pending:self._later(0,lambda:wave(index+1));return
                    def arrived():
                        nonlocal pending
                        pending-=1
                        if not pending:self._later(0,lambda:wave(index+1))
                    for route,size in plan.waves[index]:self._p2p(identity,route['path'][0],route['path'][-1],size,arrived)
                wave(0)
        except Exception as error:
            if identity not in self.tasks:raise
            self._fail(identity,error)
        return identity

    def submit_transfer(self,identity,source,target,size,on_complete,on_failure=None,*,deadline_s=None,
                        first_hop=None):
        if type(size) is not int or size<0:raise ValueError('payload must be nonnegative integer bytes')
        if first_hop is not None:
            if first_hop==source:raise ValueError('first hop must differ from source')
            routes=self._refresh()
            if first_hop not in routes.adj:raise ValueError(f'unknown first hop {first_hop}')
            if not any(key[2]==first_hop and routes.edges[key]['bandwidth_gbps']>0
                       for key in routes.adj.get(source,())):
                raise NoRouteError(f'first hop {first_hop} is not an established neighbor of {source}')
        self._register(identity,on_failure,deadline_s)
        self.tasks[identity]['first_hop']=first_hop
        try:
            self._settle();self._refresh()
            def complete():
                if self.tasks.pop(identity,None) is not None:on_complete()
            self._p2p(identity,source,target,size,complete)
        except Exception as error:
            if identity not in self.tasks:raise
            self._fail(identity,error)
        return identity

    def close(self):
        self.closed=True;self.active.clear();self.flows.clear();self.tasks.clear();self.generation+=1
