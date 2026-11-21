"""Hardware identities admitted by the Fast Mode Cell path."""

PORTABLE_BATCH_HARDWARE = frozenset({"nvidia_a800", "ascend_910b", "metax_c500"})
FAST_MODE_HARDWARE = PORTABLE_BATCH_HARDWARE | {"nvidia_h200"}
