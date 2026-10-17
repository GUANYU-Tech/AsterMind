"""The CANN FIA re-binding only accepts padding that appends dummy slots.

``GraphAttentionShape.from_token_shape`` pads a uniform decode by appending
slots whose Q and KV lengths are both the uniform decode query length, and
pads anything else by extending the largest logical segment.  Only the first
shape leaves every real request in place, so only it may be priced against the
captured slot arrays; the extending shape keeps the existing fail-safe.
"""

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor, predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.predictors.flash_attn import (
    _cann_fia_appended_slot_execution_kernel,
    _has_graph_padding,
)


HARDWARE = HWSpec("configs/hardware/ascend_910b.yaml")

LOGICAL_Q_LENS = (1, 1, 1)
LOGICAL_KV_LENS = (512, 700, 300)
APPENDED_Q_LENS = (1, 1, 1, 1)
APPENDED_KV_LENS = (512, 700, 300, 1)


def _spec(q_lens, kv_lens, model_q_lens, model_kv_lens):
    """Build a decode spec whose launch is taken at the captured slot count."""

    shape = {
        "batch": len(model_q_lens),
        "q_len": max(model_q_lens),
        "kv_len": max(model_kv_lens),
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    selection = {
        "causal": True,
        "kv_layout": "paged",
        "kv_dist": "uniform",
        "q_lens": model_q_lens,
        "kv_lens": model_kv_lens,
        "fa_ksteps": None,
        "fa_eff_mfrac": 1.0,
        "is_decode": True,
        "launch_policy": "injected_launch",
        "split_policy": "serving",
    }
    launch = select_launch_config(
        HARDWARE.hardware_id, KernelKind.FLASH_ATTN, shape, "bf16", "decode",
        selection, HARDWARE)
    attributes = {
        name: selection[name] for name in (
            "causal", "kv_layout", "kv_dist", "fa_ksteps", "fa_eff_mfrac")
    }
    attributes["q_lens"] = q_lens
    attributes["kv_lens"] = kv_lens
    if (q_lens, kv_lens) != (model_q_lens, model_kv_lens):
        attributes["model_input_q_lens"] = model_q_lens
        attributes["model_input_kv_lens"] = model_kv_lens
        attributes["graph_runtime_mode"] = "FULL"
    return KernelSpec(
        KernelKind.FLASH_ATTN, HARDWARE.hardware_id, shape, dtype="bf16",
        variant="decode", launch=launch, attributes=attributes)


def test_appended_dummy_slots_are_priced_from_the_captured_arrays():
    padded = _spec(LOGICAL_Q_LENS, LOGICAL_KV_LENS,
                   APPENDED_Q_LENS, APPENDED_KV_LENS)
    assert _has_graph_padding(padded)

    bound = _cann_fia_appended_slot_execution_kernel(padded)

    assert not _has_graph_padding(bound)
    assert bound.attributes["q_lens"] == APPENDED_Q_LENS
    assert bound.attributes["kv_lens"] == APPENDED_KV_LENS


def test_the_captured_price_stays_above_the_workload_it_supersets():
    padded = _spec(LOGICAL_Q_LENS, LOGICAL_KV_LENS,
                   APPENDED_Q_LENS, APPENDED_KV_LENS)
    exact = _spec(LOGICAL_Q_LENS, LOGICAL_KV_LENS,
                  LOGICAL_Q_LENS, LOGICAL_KV_LENS)

    predictor = create_predictor(KernelKind.FLASH_ATTN)
    assert predictor.resource_input(padded, HARDWARE)[0].pipeline_plan \
        is not None
    assert predictor.resource_input(exact, HARDWARE)[0].pipeline_plan \
        is not None
    assert predict(padded, HARDWARE, "midlayer").t_total_s > predict(
        exact, HARDWARE, "midlayer").t_total_s


def test_a_disturbed_real_slot_keeps_the_fail_safe():
    """One inflated real KV length is the extending shape, not appending."""

    padded = _spec(LOGICAL_Q_LENS, LOGICAL_KV_LENS, APPENDED_Q_LENS,
                   (513,) + APPENDED_KV_LENS[1:])

    assert _cann_fia_appended_slot_execution_kernel(padded) is padded
    assert _has_graph_padding(padded)


def test_a_dummy_slot_carrying_extra_kv_keeps_the_fail_safe():
    padded = _spec(LOGICAL_Q_LENS, LOGICAL_KV_LENS, APPENDED_Q_LENS,
                   APPENDED_KV_LENS[:-1] + (2,))

    assert _cann_fia_appended_slot_execution_kernel(padded) is padded
    assert _has_graph_padding(padded)
