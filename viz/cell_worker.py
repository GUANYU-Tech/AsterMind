"""OS subprocess entry point for an actual V1 Cell run."""
import json
import sys
import traceback
from datetime import datetime,timezone


def main():
    from runtime_environment import activate
    activate()
    from cell_jobs import folder,status,write,project
    from astermind.builder.config import SimulationConfig
    from astermind.engine.simulation import Simulation
    from astermind.__main__ import json_default
    job=sys.argv[1]; output=folder(job)
    def update(**kw): write(output/'status.json',{**status(job),**kw})
    try:
        config=SimulationConfig.load(output/'effective_configuration.json')
        update(status='running',phase='configuration',progress=5,started_at=datetime.now(timezone.utc).isoformat())
        with Simulation(config,physical_output_dir=output/'physical',startup_progress=lambda stage:
                        update(phase=stage['phase'],progress=20,startup_stage=stage)) as simulation:
            update(phase='serving',progress=40)
            result=simulation.run()
        result=json.loads(json.dumps(result,default=json_default))
        write(output/'native_result.json',result)
        write(output/'result.json',project(result))
        write(output/'requests.json',[dict(id=k,**v) for k,v in result['requests'].items()])
        write(output/'communication_events.json',result.get('communication_events',[]))
        write(output/'placement.json',config.cells[0].model_dump(mode='json')['placements'])
        write(output/'batches.json',result.get('batches',[]))
        update(status='success',phase='completed',progress=100,simulation_status=result['status'],finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        (output/'error.log').write_text(traceback.format_exc())
        update(status='failed',error=str(error),finished_at=datetime.now(timezone.utc).isoformat())
        raise


if __name__=='__main__': main()
