import pytest

from evaluation.kernel.mechanism_validation_report import (
    _nvidia_l2_directional_summary,
)


def _nvidia_l2_table(**overrides):
    names = (
        "lts__t_sectors_srcunit_tex_op_read.sum [sector]",
        "lts__t_sectors_srcunit_tex_op_write.sum [sector]",
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum [sector]",
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum [sector]",
        "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum [sector]",
        "lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum [sector]",
        "lts__t_sectors_srcunit_tex_op_atom_dot_alu.sum [sector]",
        "lts__t_sectors_srcunit_tex_op_atom_dot_cas.sum [sector]",
        "lts__t_sectors_srcunit_tex_op_red.sum [sector]",
    )
    values = {name: 0 for name in names}
    values.update(overrides)
    return {"Memory Workload Analysis Tables": {"L2 Cache": values}}


def test_nvidia_negative_raw_l2_counter_is_explicitly_unavailable():
    field = "lts__t_sectors_srcunit_tex_op_red.sum [sector]"

    result = _nvidia_l2_directional_summary(_nvidia_l2_table(**{field: -1}))

    assert result["l2_source_reduction_bytes"] is None
    assert result["l2_directional_read_write_comparable"] is False
    assert result["l2_directional_counter_status"] == "invalid_raw_counter"
    assert result["l2_directional_invalid_fields"] == [field]


def test_nvidia_l2_counter_must_be_finite():
    field = "lts__t_sectors_srcunit_tex_op_read.sum [sector]"

    with pytest.raises(ValueError, match="must be finite"):
        _nvidia_l2_directional_summary(_nvidia_l2_table(**{field: "nan"}))
