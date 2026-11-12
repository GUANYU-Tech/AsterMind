"""Generated/imported requests reach the actual new compute/Comm stack."""
import unittest
from astermind.builder.config import SimulationConfig
from astermind.builder.request_batch import prepare_batch
from astermind.engine.simulation import Simulation
from astermind.engine.audit import audit
from examples.native.compute_comm_matrix import scene


class NativeRequestBatchTests(unittest.TestCase):
    def test_generated_and_imported_requests_execute(self):
        raw=scene('DS01')
        generated=prepare_batch(dict(experiment=raw,kind='generate',mode='replace',
            options=dict(count=3,window_s=.1,input_tokens=2,output_tokens=1)))
        raw['requests']=generated['requests']
        device=raw['requests'][0]['user_device']
        imported=prepare_batch(dict(experiment=raw,mode='append',format='csv',
            content=f'id,arrival_s,user_device,input_tokens,output_tokens\nfrom-file,0.2,{device},2,1\n'))
        raw['requests']=imported['requests']
        with Simulation(SimulationConfig.model_validate(raw),cell_init_workers=1) as simulation:
            result=simulation.run()
        self.assertEqual(result['status'],'completed')
        self.assertEqual(set(result['requests']),{request['id'] for request in raw['requests']})
        self.assertEqual(audit(result)['requests'],4)
