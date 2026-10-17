"""Contract for the serialized bank-column stream of near-memory compute.

These are physical invariants of the mechanism, not fitted values: the costs
come from the hardware's own published DDR command composites, so every
assertion below is derived from `configs/hardware/samsung_hbm_pim.yaml`
rather than from a measured duration.
"""

import math

import pytest

from src.kernel.bank_stream_model import (
    BankStream,
    bank_stream_ns,
    matmul_stream,
    normalization_stream,
    pointwise_stream,
    resolve_geometry,
    softmax_stream,
    stream_ns,
)
from src.kernel.hardware_spec import HWSpec

PIM_CONFIG = "configs/hardware/samsung_hbm_pim.yaml"


@pytest.fixture(scope="module")
def pim():
    return HWSpec(PIM_CONFIG)


@pytest.fixture(scope="module")
def geometry(pim):
    return resolve_geometry(pim)


class TestGeometry:
    def test_geometry_is_derived_from_the_published_array(self, geometry):
        memory = HWSpec(PIM_CONFIG).memory_subsystem
        column_bytes = (
            memory["pim_row_bytes"] / memory["pim_columns_per_row"])
        assert geometry.lanes_per_column == (
            column_bytes / memory["dtype_bytes"])
        assert geometry.columns_per_buffer == (
            memory["pim_pu_inbuf_bytes"] / column_bytes)
        assert geometry.elements_per_column_step == (
            geometry.lanes_per_column * geometry.units)

    def test_partial_column_still_costs_a_whole_column(self, geometry):
        one = pointwise_stream(geometry.elements_per_column_step)
        part = pointwise_stream(geometry.elements_per_column_step // 4)
        assert stream_ns(part, geometry) == stream_ns(one, geometry)

    def test_a_fractional_column_of_elements_is_rejected(self, pim):
        bad = dict(pim.memory_subsystem)
        bad["pim_pu_inbuf_bytes"] = bad["pim_pu_inbuf_bytes"] + 1

        class _Hardware:
            memory_subsystem = bad
            num_compute_units = pim.num_compute_units

        with pytest.raises(ValueError, match="whole number of columns"):
            resolve_geometry(_Hardware())


class TestBoundaryStructure:
    """Each boundary the stream crosses costs strictly more than the last."""

    @staticmethod
    def _step(stage, geometry, columns):
        step = geometry.elements_per_column_step
        return (stream_ns(type(stage)(**{**stage.__dict__,
                                         "work_elements": columns * step}),
                          geometry)
                - stream_ns(type(stage)(**{**stage.__dict__,
                                           "work_elements":
                                           (columns - 1) * step}), geometry))

    def test_buffer_and_row_boundaries_dominate_a_plain_column(self, geometry):
        stage = pointwise_stream(1.0)
        buffer_columns = geometry.columns_per_buffer
        row_columns = geometry.columns_per_row
        plain = self._step(stage, geometry, 2)
        at_buffer = self._step(stage, geometry, buffer_columns + 1)
        at_row = self._step(stage, geometry, row_columns + 1)
        assert plain < at_buffer < at_row

    def test_a_plain_column_costs_its_own_bank_accesses(self, geometry):
        for stage in (pointwise_stream(1.0), normalization_stream(1.0),
                      softmax_stream(1.0), matmul_stream(1, 1, 1)):
            expected = max(
                stage.accesses_per_column * geometry.column_access_ns,
                geometry.unit_latency_ns)
            assert self._step(stage, geometry, 2) == expected

    def test_a_buffer_boundary_pays_the_bus_turnaround(self, geometry):
        stage = pointwise_stream(1.0)
        at_buffer = self._step(stage, geometry, geometry.columns_per_buffer + 1)
        plain = self._step(stage, geometry, 2)
        assert at_buffer - plain == pytest.approx(
            (geometry.columns_per_buffer - 1) * geometry.column_access_ns
            + geometry.bus_turnaround_ns)

    def test_a_host_fed_operand_has_no_buffer_boundary(self, geometry):
        """A matrix product's buffered operand arrives from the host, so its
        stream never turns the bank bus around mid-row."""

        stage = matmul_stream(1, 1, 1)
        assert stage.refills_per_chunk == 0
        assert self._step(stage, geometry, geometry.columns_per_buffer + 1) == (
            self._step(stage, geometry, 2))

    def test_the_first_column_skips_closing_a_previous_row(self, geometry):
        stage = pointwise_stream(geometry.elements_per_column_step)
        at_row = self._step(
            pointwise_stream(1.0), geometry, geometry.columns_per_row + 1)
        assert stream_ns(stage, geometry) == (
            at_row - geometry.row_close_ns)


class TestOperatorProfiles:
    def test_more_passes_over_the_data_cost_more(self, geometry):
        elements = 64 * geometry.elements_per_column_step
        matmul = stream_ns(matmul_stream(1, 1, elements), geometry)
        pointwise = stream_ns(pointwise_stream(elements), geometry)
        softmax = stream_ns(softmax_stream(elements), geometry)
        normalization = stream_ns(normalization_stream(elements), geometry)
        assert matmul < pointwise < softmax < normalization

    def test_the_stream_grows_monotonically_with_work(self, geometry):
        step = geometry.elements_per_column_step
        times = [stream_ns(pointwise_stream(n * step), geometry)
                 for n in range(1, 130)]
        assert all(b > a for a, b in zip(times, times[1:]))

    def test_a_matrix_product_delivers_its_result_out_of_the_bank(
            self, geometry):
        """The accumulated result is read back out; a wider result costs more
        even when the multiply-accumulate work is unchanged."""

        narrow = stream_ns(matmul_stream(1, 4096, 4096), geometry)
        wide = stream_ns(matmul_stream(4096, 1, 4096), geometry)
        assert wide > narrow

    def test_stages_are_serial(self, pim, geometry):
        stages = (pointwise_stream(8192.0), softmax_stream(8192.0))
        assert bank_stream_ns(stages, pim) == pytest.approx(math.fsum(
            stream_ns(stage, geometry) for stage in stages))


class TestFailClosed:
    def test_empty_operations_are_rejected(self, pim):
        with pytest.raises(ValueError, match="at least one stage"):
            bank_stream_ns((), pim)

    @pytest.mark.parametrize("elements", [0.0, -1.0, float("nan")])
    def test_non_positive_work_is_rejected(self, elements):
        with pytest.raises(ValueError, match="finite and positive"):
            pointwise_stream(elements)

    def test_a_row_change_must_cost_at_least_one_row_cycle(self):
        with pytest.raises(ValueError, match="at least one row cycle"):
            BankStream(
                work_elements=1.0,
                accesses_per_column=1,
                refills_per_chunk=0,
                refills_per_row_change=0,
                row_cycles_per_row_change=0,
                row_change_follows_write=False,
                accumulator_roundtrip=False,
            )

    def test_degenerate_matrix_dimensions_are_rejected(self):
        with pytest.raises(ValueError, match="positive dimensions"):
            matmul_stream(0, 4096, 4096)


class TestPimForwardIntegration:
    """One end-to-end smoke per operator family on the PIM Forward path."""

    CASES = (
        ("gemm", "GEMM_B1_M1_K4096_N4096"),
        ("eager_attn", "EAGERATTN_B16_S1024_HQ32_HKV32_D128_SP2_C0"),
        ("unfused_moe", "UNFUSEDMOE_TOK64_E8_TOPK2_H4096_INT14336"),
        ("vector", "VECTOR_ELEMENTWISE_T256_H2048_B1_ADD"),
    )

    @staticmethod
    def _predict(family, case_id, pim):
        from evaluation.kernel.evaluation_runner import normalize_kernel_spec
        from src.kernel.dataset_storage import load_kernel_dataset_document
        from src.kernel.kernel_base import create_predictor

        document = load_kernel_dataset_document(
            f"kernel_dataset/samsung_hbm_pim/{family}.json.zst")
        entry = document[case_id]
        spec = normalize_kernel_spec(
            entry["op_info"], entry["kernels"][0], "samsung_hbm_pim", pim)
        prediction = create_predictor(spec.kind).midlayer(spec, pim)
        measured = entry["op_info"]["evaluation"]["timing"][
            "total_duration_ns"]
        return prediction, measured

    @pytest.mark.parametrize("family,case_id", CASES)
    def test_the_bank_stream_sets_the_execution_time(
            self, family, case_id, pim):
        prediction, _ = self._predict(family, case_id, pim)
        assert prediction.bottleneck_group == "analytical_execution_floor"

    @pytest.mark.parametrize("family,case_id", CASES)
    def test_prediction_tracks_the_simulated_groundtruth(
            self, family, case_id, pim):
        prediction, measured = self._predict(family, case_id, pim)
        predicted_ns = prediction.t_total_s * 1e9
        assert abs(predicted_ns - measured) / measured < 0.05
