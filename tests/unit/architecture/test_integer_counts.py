"""Physical hardware counts cannot silently become fractional/boolean devices."""
import unittest

from orbitserve.architecture.hardware import (
    CommunicationSpec, ComputePayloadSpec, IntraInterconnectSpec, StorageSpec,
)


class HardwareCountTests(unittest.TestCase):
    def test_counts_reject_fractional_and_boolean_values_through_both_inputs(self):
        for kind, field in ((ComputePayloadSpec, 'accelerator_count'), (StorageSpec, 'device_count'),
                            (IntraInterconnectSpec, 'switch_count'), (CommunicationSpec, 'isl_terminal_count')):
            for value in (1.9, True, '2', float('nan')):
                with self.subTest(kind=kind.__name__, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        kind(**{field: value})
                    with self.assertRaisesRegex(ValueError, field):
                        kind.from_dict({field: value})
