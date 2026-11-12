"""Retain native kernels while exposing each step's input distribution size."""


class NativeStepExecution:
    input_distribution_in_graph = False

    def __init__(self, native):
        self.native = native
        self.last_step_token_count = None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def execute_step(self, schedule, **kwargs):
        self.last_step_token_count = schedule.token_count
        return self.native.execute_step(schedule, **kwargs)
