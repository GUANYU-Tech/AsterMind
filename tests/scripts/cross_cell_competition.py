"""DS02: real independent native schedulers and shared whole-path Comm."""
from astermind.common.bootstrap import activate
from astermind.builder.config import SimulationConfig
from astermind.engine.network_host import DeviceRuntimeHost,DeviceCommunicationRuntime
from astermind.engine.cell_events import CellRequest
from astermind.engine.audit import audit
from examples.native.compute_comm_matrix import cross_cell_competition


def run(compute_mode,intra_mode,*,concurrent):
    root=activate()
    from astermind.engine.compute.runtime import MidlayerCellConfig
    from astermind.engine.compute.serving import ServingScheduler
    from astermind.engine.compute.placement import RankPlacement
    from astermind.engine.compute.batch_executor import RuntimeBatchExecutor
    raw,placements=cross_cell_competition(compute_mode=compute_mode,intra_mode=intra_mode)
    config=SimulationConfig.model_validate(raw);topology=config.topology()
    host=DeviceRuntimeHost(topology,config);network=DeviceCommunicationRuntime(host,topology)
    cells={};executors={};completed=[];outputs=[]
    def fail(error):raise AssertionError(error)
    try:
        for identity,devices in list(placements.items())[:2 if concurrent else 1]:
            ranks=tuple(RankPlacement(identity,'main',rank,topology.devices[device].satellite,device)
                        for rank,device in enumerate(devices))
            executor=RuntimeBatchExecutor(identity,host.timeline,network.communication_policy,ranks)
            executors[identity]=executor
            native=MidlayerCellConfig(root,model_file='configs/model/qwen3-8b.json',
                hardware_file='configs/hardware/nvidia_h200.yaml',hardware_id='nvidia_h200',
                tensor_parallel_size=2,compute_mode=compute_mode,max_model_len=32)
            cells[identity]=ServingScheduler.create(identity,host.timeline,native,completed.append,
                on_output=outputs.append,on_failure=fail,batch_executor=executor)
        for identity,cell in cells.items():
            cell.submit_ready_request(CellRequest(identity,host.now,2,2))
        host.run()
        assert len(completed)==len(cells)
        assert all(c.generated_tokens==2 for c in completed)
        assert sum(output.token_count for output in outputs)==2*len(cells)
        comm=network.communication_policy
        assert not (comm.tasks or comm.active or comm.flows)
        if concurrent:assert comm.stats['peak_active_flows']>=2,comm.stats
        result=dict(configuration=raw,requests={},transfers=[],migrations={},
            request_state_events=[],event_log=host.timeline.events,
            batches=[b for executor in executors.values() for b in executor.records],
            input_distributions=[d for executor in executors.values() for d in executor.input_distributions],
            cells={identity:cell.report() for identity,cell in cells.items()},
            communication_events=comm.trace,communication_runtime_stats=comm.stats,
            completion_ticks={c.request_id:str(c.completion_time_s.ticks) for c in completed},
            scope='compute_comm_integration',physical_cell_ownership_preserved=True)
        result['audit']=audit(result)
        return result
    finally:
        for cell in cells.values():cell.close()
        network.close();host.close()
