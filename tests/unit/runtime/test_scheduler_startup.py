"""Cold scheduler startup and normal RPCs have different time budgets."""
import unittest
from unittest.mock import Mock, call, patch

from astermind.engine.compute.latest_vllm_backend import LatestSchedulerProcess


class SchedulerStartupTests(unittest.TestCase):
    def context(self):
        context=Mock();parent=Mock();child=Mock();process=Mock()
        context.Pipe.return_value=(parent,child)
        context.Process.return_value=process
        process.is_alive.return_value=False
        return context,parent,child,process

    def test_cold_handshake_gets_startup_budget_while_rpc_retains_deadline(self):
        context,parent,child,process=self.context()
        parent.poll.return_value=True
        parent.recv.side_effect=[('ok','runtime-ready'),('ok','step-ready')]
        with patch('astermind.engine.compute.latest_vllm_backend.multiprocessing.get_context',return_value=context):
            scheduler=LatestSchedulerProcess({},None,storage_options={})
            self.assertEqual(scheduler.runtime_info,'runtime-ready')
            self.assertEqual(scheduler.plan_step(1),'step-ready')
            self.assertEqual(parent.poll.call_args_list,[call(180),call(60)])
            parent.send.assert_called_once_with(('plan_step',(1,),{}))
            scheduler.close()
            process.join.assert_called_once()
            child.close.assert_called_once()

    def test_failed_handshake_still_closes_the_child_and_parent_pipes(self):
        context,parent,child,process=self.context()
        parent.poll.return_value=False
        with patch('astermind.engine.compute.latest_vllm_backend.multiprocessing.get_context',return_value=context):
            with self.assertRaisesRegex(TimeoutError,'180 seconds'):
                LatestSchedulerProcess({},None,storage_options={})
        parent.close.assert_called_once();child.close.assert_called_once()
        process.join.assert_called_once()

    def test_normal_rpc_timeout_is_not_extended_to_hide_hung_steps(self):
        scheduler=LatestSchedulerProcess.__new__(LatestSchedulerProcess)
        scheduler.pipe=Mock();scheduler.pipe.poll.return_value=False
        with self.assertRaisesRegex(TimeoutError,'60 seconds'):scheduler.receive()
        scheduler.pipe.poll.assert_called_once_with(60)


if __name__=='__main__':unittest.main()
