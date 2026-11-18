"""Event-driven external executor consuming the existing unpriced KernelPlan."""
from dataclasses import asdict, replace
from astermind.engine.clock import Time
from astermind.engine.compute.executor_base import NetworkBatchExecutor
from astermind.engine.comm.operator import CommunicationOperator
from astermind.engine.comm.native_intra import clear_native_contexts, native_enabled
from astermind.engine.compute.ep_routing import direct_ep_matrices
from astermind.engine.timeline import nanoseconds
from src.kernel.kernel_spec import KernelKind
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.kernel_lowering.execution_timeline import (
    _plan_schedule_inputs,_logical_schedule,call_reserved_resource_ids,call_host_lanes,
)


class RuntimeBatchExecutor(NetworkBatchExecutor):
    def start_batch(self,cell_id,step_id,components,execution,on_complete,*,token_count,
                    pool_name="main",on_next_batch=None,replica_tokens=None,on_failure=None):
        if cell_id!=self.cell_id or pool_name in self._active:
            raise ValueError("invalid Cell or batch already in flight")
        state = dict(alive=True, tasks=set(), pool=pool_name, step=step_id,
                     failure=on_failure, origin=self.loop.now_s, record=None)
        try:
            return self._start_batch(cell_id,step_id,components,execution,on_complete,
                token_count=token_count,pool_name=pool_name,on_next_batch=on_next_batch,
                replica_tokens=replica_tokens,state=state)
        except Exception as error:
            if not state['alive']:raise
            self._abort_batch(state,error)

    def _abort_batch(self,state,error):
        if not state['alive']:
            return
        state['alive']=False
        # Cancellation callbacks may re-enter this method; mark terminal first.
        for identity in tuple(state['tasks']):
            if identity in getattr(self.comm,'tasks',{}):
                self.comm.cancel(identity)
        self._active.discard(state['pool'])
        clear_native_contexts(getattr(self.comm,'network',None),self.cell_id,
                              f"{self.cell_id}/{state['pool']}/{state['step']}/")
        detail=error.model_dump() if hasattr(error,'model_dump') else dict(reason=str(error))
        failure=dict(cell_id=self.cell_id,pool_name=state['pool'],step_id=state['step'],
                     time_s=self.loop.now_s,reason=detail.get('reason',str(error)),communication=detail)
        record=state['record'] or dict(cell_id=self.cell_id,pool_name=state['pool'],step_id=state['step'],
            start_time_ns=nanoseconds(state['origin']),start_time_ticks=str(state['origin'].ticks),calls=[],
            native_communication_duration_charged_s=0.)
        record.update(status='failed',end_time_ns=nanoseconds(self.loop.now_s),
                      end_time_ticks=str(self.loop.now_s.ticks),failure=failure)
        self.records.append(record)
        if state['failure'] is None:
            raise RuntimeError(f"batch communication failed: {failure['reason']}")
        state['failure'](failure)
        if self._on_batch_boundary:self._on_batch_boundary()

    def _submit_batch_task(self,state,operator,callback,identity):
        if not state['alive']:return
        state['tasks'].add(identity)
        try:
            self.comm.submit(operator,callback,lambda error:self._abort_batch(state,error),task_id=identity)
        except Exception as error:
            if not state['alive']:raise
            self._abort_batch(state,error)

    def _start_batch(self,cell_id,step_id,components,execution,on_complete,*,token_count,
                    pool_name,on_next_batch,replica_tokens,state):
        clear_native_contexts(getattr(self.comm,'network',None),self.cell_id,f'{self.cell_id}/{pool_name}/')
        if not isinstance(components,KernelPlan):raise ValueError("runtime requires an unpriced KernelPlan")
        if execution.host_runtime_prepare_profiles is not None:
            raise NotImplementedError("Host runtime preparation is outside this executor")
        predictions=execution.predict_local_plan(components)
        if any(c.call_id in predictions for c in components.calls if c.kernel_spec.kind is KernelKind.COMMUNICATION):
            raise ValueError("local prediction must exclude communication")
        pool=execution.deployment.worker_pools[pool_name]
        dp=pool.parallel_plan.config.data_parallel_size
        counts=(token_count,) if dp==1 and replica_tokens is None else replica_tokens
        if not counts or len(counts)!=dp or sum(counts)!=token_count or any(type(n) is not int or n<0 for n in counts):
            raise ValueError("input distribution needs valid per-replica token counts")
        ranks=pool.physical_ranks_for_pipeline_stage(0)
        groups=(ranks,) if dp==1 else tuple(dict.fromkeys(pool.physical_rank_group("tensor",
            pool.logical_rank_for_physical_rank(r)) for r in ranks))
        if len(groups)!=dp:raise ValueError("input distribution replica count mismatch")
        self._active.add(pool_name)
        origin=nanoseconds(self.loop.now_s)
        origin_ticks=str(self.loop.now_s.ticks)
        pending=[]
        for replica,(group,count) in enumerate(zip(groups,counts,strict=True)):
            if count and len(group)>1:
                pending.append((replica,CommunicationOperator("broadcast",None,
                    tuple(self._device(pool_name,r) for r in group),count*8,"payload",
                    cell_id=self.cell_id,task_id=f'{self.cell_id}/{pool_name}/{step_id}/input/{replica}')))
        deliveries=[]
        def start():
            if not state['alive']:return
            if deliveries:self.input_distributions.append(dict(cell_id=cell_id,pool_name=pool_name,step_id=step_id,
                ready_ns=origin,ready_ticks=origin_ticks,completed_ticks=str(self.loop.now_s.ticks),
                completed_ns=nanoseconds(self.loop.now_s),replicas=deliveries))
            try:self._run_plan(step_id,components,predictions,execution,pool_name,on_complete,on_next_batch,state)
            except Exception as error:
                if not state['alive']:raise
                self._abort_batch(state,error)
        if not pending:start();return
        left=len(pending)
        def delivered(replica,estimate):
            nonlocal left
            if not state['alive']:return
            deliveries.append(dict(replica=replica,communication_estimate=asdict(estimate)))
            left-=1
            if not left:start()
        for replica,operator in pending:
            if not state['alive']:break
            self._bind_input_distribution(operator,execution,pool)
            self._submit_batch_task(state,operator,lambda estimate,r=replica:delivered(r,estimate),
                                    f"{cell_id}/{pool_name}/{step_id}/input/{replica}")

    def _run_plan(self,step_id,plan,predictions,execution,pool_name,on_complete,on_next_batch,state):
        start_time=self.loop.now_s
        start_ns=nanoseconds(start_time)
        inputs=_plan_schedule_inputs(plan);calls=inputs[1]
        logical=tuple(_logical_schedule(plan,*inputs))
        policy = getattr(self.comm, "policy", None)
        routed = (direct_ep_matrices(calls, execution, pool_name, step_id)
                  if getattr(policy, "ep_strategy", "allgather") == "direct" else {})
        if routed and native_enabled(policy, self.cell_id):
            raise NotImplementedError("Direct DeepSeek V4 EP requires surrogate communication")
        host_end={};gaps=None
        if execution.host_launch_profiles:
            from src.serving.kernel_lowering.execution_timeline import _DeviceLaunchGapTracker,eager_host_ready_times
            gaps=_DeviceLaunchGapTracker(execution.host_launch_profiles)
            host_end={k:Time(t) for k,t in eager_host_ready_times(inputs[0],tuple(k for k,_,_,_ in logical),execution.host_launch_profiles).items()}
        rows={};last={};successors={};remaining={};operators={};records=[]
        end_times={};host_ready={};launch_gaps={}
        for key,deps,position,_ in logical:
            call=calls[key[0]];communication=call.kernel_spec.kind is KernelKind.COMMUNICATION
            deps=set(deps)
            if communication:
                operator=self._operator(call,execution,step_id)
                if key[0] in routed:
                    matrix = routed[key[0]]
                    operator = replace(operator, operation="all_to_all", algorithm=None,
                                       bytes=sum(map(sum, matrix)), bytes_semantics="aggregate_input",
                                       reduction=None, partition_bytes=(), traffic_matrix_bytes=matrix)
                operators[key]=operator
                resources=tuple(f"network-endpoint:{g}" for g in operator.participants)
                prediction=None
            else:
                self._device(call.pool_name,call.rank)
                resources=call_reserved_resource_ids(call);prediction=predictions[call.call_id].t_total_s
            for resource in resources:
                if resource in last:deps.add(last[resource])
                last[resource]=key
            gap=Time(gaps.observe(call.kernel_spec.hardware_id,tuple((lane,None) for lane in call_host_lanes(call)))) if gaps else Time()
            host_ready[key]=start_time+host_end.get(key,Time())
            launch_gaps[key]=gap
            row=dict(pool_name=pool_name,call_id=call.call_id,occurrence=key[1],rank=call.rank,kind=call.kernel_spec.kind.value,
                communication=communication,dependencies=sorted(deps),position=position,start_time_ns=None,
                end_time_ns=None,native_prediction_s=prediction,host_ready_time_ns=nanoseconds(host_ready[key]),
                host_ready_time_ticks=str(host_ready[key].ticks),launch_gap_ticks=str(gap.ticks),launch_gap_ns=nanoseconds(gap))
            rows[key]=row;records.append(row);remaining[key]=len(deps)
            for dep in deps:successors.setdefault(dep,[]).append(key)
        if len(rows)!=plan.logical_call_count or not rows:raise ValueError("KernelPlan did not expand completely")
        batch=dict(cell_id=self.cell_id,pool_name=pool_name,step_id=step_id,start_time_ticks=str(start_time.ticks),start_time_ns=start_ns,end_time_ns=None,calls=records,
            native_batch_duration_s=None,native_communication_duration_charged_s=0.,model="native compute and runtime surrogate communication")
        state['record']=batch
        finished=set()
        def finish(key,estimate=None):
            if not state['alive']:return
            if key in finished:raise RuntimeError("duplicate task completion")
            end_times[key]=self.loop.now_s
            row=rows[key];row["end_time_ns"]=nanoseconds(self.loop.now_s)
            row["end_time_ticks"]=str(self.loop.now_s.ticks)
            if estimate is not None:
                row["communication_estimate"]=asdict(estimate)
                row["communication_operation"]=operators[key].operation
                if key[0] in routed:
                    row["ep_strategy"] = "direct"
            finished.add(key)
            for child in successors.get(key,()):
                remaining[child]-=1
                if not remaining[child]:arm(child)
            if len(finished)==len(rows):
                batch["end_time_ns"]=nanoseconds(self.loop.now_s)
                batch["end_time_ticks"]=str(self.loop.now_s.ticks)
                def publish():
                    if not state['alive']:return
                    state['alive']=False
                    clear_native_contexts(getattr(self.comm,'network',None),self.cell_id,f'{self.cell_id}/{pool_name}/{step_id}/')
                    self._active.remove(pool_name);self.records.append(batch)
                    on_complete(self.loop.now_s)
                    if self._on_batch_boundary:self._on_batch_boundary()
                    if on_next_batch:on_next_batch()
                self.loop.schedule(self.loop.now_s,1,"batch:complete",publish,owner_id=self.cell_id)
        def arm(key):
            if not state['alive']:return
            row=rows[key]
            dep_end=max((end_times[d] for d in row["dependencies"]),default=start_time)
            when=max(dep_end+launch_gaps[key],host_ready[key])
            def begin():
                if not state['alive']:return
                row["start_time_ns"]=nanoseconds(self.loop.now_s)
                row["start_time_ticks"]=str(self.loop.now_s.ticks)
                if row["communication"]:
                    self._submit_batch_task(state,operators[key],lambda estimate:finish(key,estimate),
                        f"{self.cell_id}/{pool_name}/{step_id}/{key[0]}/{key[1]}")
                else:self.loop.schedule(self.loop.now_s+row["native_prediction_s"],1,"kernel:complete",lambda:finish(key))
            self.loop.schedule(max(self.loop.now_s,when),3,"kernel:start",begin)
        for key in rows:
            if remaining[key]==0:arm(key)
