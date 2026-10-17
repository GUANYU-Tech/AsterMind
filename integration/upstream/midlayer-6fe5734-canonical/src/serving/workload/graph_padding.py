"""Serving graph dispatch and token-shape domains."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Tuple

from .workload_graph import WorkloadKind


_GRAPH_RUNTIME_MODES = ("NONE", "FULL", "PIECEWISE")
_CONFIGURED_MODE_DISPATCH = {
    "NONE": ("NONE", "NONE"),
    "PIECEWISE": ("PIECEWISE", "PIECEWISE"),
    "FULL": ("FULL", "FULL"),
    "FULL_DECODE_ONLY": ("FULL", "NONE"),
    "FULL_AND_PIECEWISE": ("FULL", "PIECEWISE"),
}


@dataclass(frozen=True)
class GraphTokenShape:
    """Token domains resolved from one runtime graph dispatch.

    ``logical_tokens`` is Scheduler work.  It alone advances KV, contributes
    effective FLOP, and represents valid mask/routing rows.  In contrast,
    ``model_input_tokens`` is the captured model-input tensor extent.  It is
    the only token count allowed to size replayed tensors, grids, workspaces,
    and temporary buffers.
    """

    runtime_mode: str
    logical_tokens: int
    model_input_tokens: int
    attention_active_tokens: int
    kv_write_tokens: int

    def __post_init__(self) -> None:
        if self.runtime_mode not in _GRAPH_RUNTIME_MODES:
            raise ValueError(
                f"unknown graph runtime mode: {self.runtime_mode!r}")
        for name in (
                "logical_tokens", "model_input_tokens",
                "attention_active_tokens", "kv_write_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"graph {name} must be an integer")
            if value <= 0:
                raise ValueError(f"graph {name} must be positive")
        if self.model_input_tokens < self.logical_tokens:
            raise ValueError("graph model input cannot shrink logical tokens")
        if self.attention_active_tokens != self.logical_tokens:
            raise ValueError(
                "graph attention metadata must preserve logical rows")
        if self.kv_write_tokens != self.logical_tokens:
            raise ValueError(
                "graph KV metadata must preserve logical writes")
        if self.runtime_mode == "NONE" and (
                self.model_input_tokens != self.logical_tokens):
            raise ValueError("eager execution cannot carry graph padding")

    @property
    def padding_tokens(self) -> int:
        return self.model_input_tokens - self.logical_tokens

    def attention_shape(
        self,
        logical_q_lens: tuple[int, ...],
        logical_kv_lens: tuple[int, ...],
        *,
        uniform_decode_query_len: int,
    ) -> "GraphAttentionShape":
        """Derive the separate active and model-input FA ABI domains.

        Graph replay pads a uniform decode by appending masked one-query
        slots.  For non-uniform work it extends the largest query segment so
        the captured tensor has the selected total extent.  The latter is a
        tensor-layout fact only: logical request lengths remain the sole
        source of valid attention and KV work.
        """

        return GraphAttentionShape.from_token_shape(
            self,
            logical_q_lens,
            logical_kv_lens,
            uniform_decode_query_len=uniform_decode_query_len,
        )


@dataclass(frozen=True)
class GraphAttentionShape:
    """Logical and captured FlashAttention request-length domains."""

    runtime_mode: str
    logical_q_lens: tuple[int, ...]
    logical_kv_lens: tuple[int, ...]
    model_input_q_lens: tuple[int, ...]
    model_input_kv_lens: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.runtime_mode not in _GRAPH_RUNTIME_MODES:
            raise ValueError(
                f"unknown graph runtime mode: {self.runtime_mode!r}")
        for name in (
                "logical_q_lens", "logical_kv_lens",
                "model_input_q_lens", "model_input_kv_lens"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"graph {name} must be a tuple")
            if not values:
                raise ValueError(f"graph {name} cannot be empty")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in values):
                raise ValueError(
                    f"graph {name} must contain positive integers")
        if len(self.logical_q_lens) != len(self.logical_kv_lens):
            raise ValueError("logical attention request lengths must align")
        if len(self.model_input_q_lens) != len(self.model_input_kv_lens):
            raise ValueError("model-input attention request lengths must align")
        if any(query > key_value for query, key_value in zip(
                self.logical_q_lens, self.logical_kv_lens)):
            raise ValueError("logical attention query length exceeds KV length")
        if any(query > key_value for query, key_value in zip(
                self.model_input_q_lens, self.model_input_kv_lens)):
            raise ValueError(
                "model-input attention query length exceeds KV length")
        if self.runtime_mode == "NONE" and (
                self.logical_q_lens != self.model_input_q_lens
                or self.logical_kv_lens != self.model_input_kv_lens):
            raise ValueError("eager attention cannot carry graph padding")

    @property
    def logical_tokens(self) -> int:
        return sum(self.logical_q_lens)

    @property
    def model_input_tokens(self) -> int:
        return sum(self.model_input_q_lens)

    @classmethod
    def from_token_shape(
        cls,
        token_shape: GraphTokenShape,
        logical_q_lens: tuple[int, ...],
        logical_kv_lens: tuple[int, ...],
        *,
        uniform_decode_query_len: int,
    ) -> "GraphAttentionShape":
        for name, values in (
                ("logical_q_lens", logical_q_lens),
                ("logical_kv_lens", logical_kv_lens)):
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"graph {name} must be a sequence")
            if not values:
                raise ValueError(f"graph {name} cannot be empty")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in values):
                raise ValueError(
                    f"graph {name} must contain positive integers")
        q_lens = tuple(logical_q_lens)
        kv_lens = tuple(logical_kv_lens)
        if len(q_lens) != len(kv_lens):
            raise ValueError("logical attention request lengths must align")
        if sum(q_lens) != token_shape.attention_active_tokens:
            raise ValueError(
                "logical attention lengths disagree with graph active tokens")
        if sum(q_lens) != token_shape.kv_write_tokens:
            raise ValueError(
                "logical attention lengths disagree with graph KV writes")
        if (isinstance(uniform_decode_query_len, bool)
                or not isinstance(uniform_decode_query_len, int)
                or uniform_decode_query_len <= 0):
            raise ValueError("uniform decode query length must be positive")
        if token_shape.padding_tokens == 0:
            return cls(token_shape.runtime_mode, q_lens, kv_lens, q_lens, kv_lens)
        if token_shape.runtime_mode == "NONE":
            raise ValueError("eager graph shape cannot carry padding")
        if all(length == uniform_decode_query_len for length in q_lens):
            padding = token_shape.padding_tokens
            if padding % uniform_decode_query_len:
                raise ValueError(
                    "graph bucket cannot represent uniform decode padding")
            padded_requests = padding // uniform_decode_query_len
            model_q_lens = q_lens + (
                uniform_decode_query_len,
            ) * padded_requests
            model_kv_lens = kv_lens + (
                uniform_decode_query_len,
            ) * padded_requests
        else:
            padded_index = max(range(len(q_lens)), key=q_lens.__getitem__)
            model_q_lens = list(q_lens)
            model_kv_lens = list(kv_lens)
            model_q_lens[padded_index] += token_shape.padding_tokens
            model_kv_lens[padded_index] = max(
                model_kv_lens[padded_index], model_q_lens[padded_index])
            model_q_lens = tuple(model_q_lens)
            model_kv_lens = tuple(model_kv_lens)
        result = cls(
            token_shape.runtime_mode,
            q_lens,
            kv_lens,
            model_q_lens,
            model_kv_lens,
        )
        if result.model_input_tokens != token_shape.model_input_tokens:
            raise RuntimeError(
                "model-input attention lengths disagree with graph token shape")
        return result


@dataclass(frozen=True)
class GraphExecutionConfig:
    """Resolved graph-runtime inputs used to derive physical token rows."""

    configured_mode: str
    capture_sizes: Tuple[int, ...]
    piecewise_eager_workload_kinds: Tuple[str, ...]
    uniform_decode_query_len: int = 1

    def __post_init__(self) -> None:
        if self.configured_mode not in _CONFIGURED_MODE_DISPATCH:
            raise ValueError(
                f"unknown configured graph mode: {self.configured_mode!r}")
        if not isinstance(self.piecewise_eager_workload_kinds, (list, tuple)):
            raise TypeError(
                "piecewise eager workload kinds must be a sequence")
        piecewise_kinds = tuple(self.piecewise_eager_workload_kinds)
        if any(not isinstance(kind, str) or not kind for kind in piecewise_kinds):
            raise TypeError(
                "piecewise eager workload kinds must be non-empty strings")
        if len(piecewise_kinds) != len(set(piecewise_kinds)):
            raise ValueError("piecewise eager workload kinds must be unique")
        try:
            normalized_piecewise_kinds = tuple(
                WorkloadKind(kind).value for kind in piecewise_kinds)
        except ValueError as error:
            raise ValueError(
                "piecewise eager workload kind is unsupported") from error
        uses_piecewise = "PIECEWISE" in (
            _CONFIGURED_MODE_DISPATCH[self.configured_mode])
        if uses_piecewise != bool(normalized_piecewise_kinds):
            raise ValueError(
                "piecewise Graph modes require explicit eager workload kinds, "
                "and non-piecewise modes forbid them")
        object.__setattr__(
            self, "piecewise_eager_workload_kinds",
            normalized_piecewise_kinds)
        if (isinstance(self.uniform_decode_query_len, bool)
                or not isinstance(self.uniform_decode_query_len, int)):
            raise TypeError("uniform decode query length must be an integer")
        if self.uniform_decode_query_len <= 0:
            raise ValueError("uniform decode query length must be positive")
        if not isinstance(self.capture_sizes, (list, tuple)):
            raise TypeError("graph capture_sizes must be a sequence")
        sizes = tuple(self.capture_sizes)
        if any(isinstance(size, bool) or not isinstance(size, int)
               for size in sizes):
            raise TypeError("graph capture sizes must be integers")
        if any(size <= 0 for size in sizes):
            raise ValueError("graph capture sizes must be positive")
        if len(sizes) != len(set(sizes)):
            raise ValueError("graph capture sizes must be unique")
        sizes = tuple(sorted(sizes))
        if not self.enabled and sizes:
            raise ValueError("disabled graph execution cannot carry buckets")
        if self.enabled and not sizes:
            raise ValueError("enabled graph execution requires capture buckets")
        object.__setattr__(self, "capture_sizes", sizes)

    @classmethod
    def disabled(cls) -> "GraphExecutionConfig":
        return cls("NONE", (), ())

    @property
    def uniform_decode_mode(self) -> str:
        return _CONFIGURED_MODE_DISPATCH[self.configured_mode][0]

    @property
    def non_uniform_mode(self) -> str:
        return _CONFIGURED_MODE_DISPATCH[self.configured_mode][1]

    @property
    def enabled(self) -> bool:
        return self.configured_mode != "NONE"

    def resolve(
        self,
        logical_tokens: int,
        *,
        uniform_decode: bool,
    ) -> GraphTokenShape:
        if isinstance(logical_tokens, bool) or not isinstance(logical_tokens, int):
            raise TypeError("logical graph tokens must be an integer")
        if logical_tokens <= 0:
            raise ValueError("logical graph tokens must be positive")
        if not isinstance(uniform_decode, bool):
            raise TypeError("uniform_decode must be a boolean")
        runtime_mode = (self.uniform_decode_mode if uniform_decode else
                        self.non_uniform_mode)
        if runtime_mode == "NONE":
            return GraphTokenShape(
                "NONE", logical_tokens, logical_tokens, logical_tokens,
                logical_tokens)
        bucket_index = bisect_left(self.capture_sizes, logical_tokens)
        if bucket_index == len(self.capture_sizes):
            return GraphTokenShape(
                "NONE", logical_tokens, logical_tokens, logical_tokens,
                logical_tokens)
        physical_tokens = self.capture_sizes[bucket_index]
        return GraphTokenShape(
            runtime_mode,
            logical_tokens,
            physical_tokens,
            logical_tokens,
            logical_tokens,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 5,
            "configured_mode": self.configured_mode,
            "uniform_decode_query_len": self.uniform_decode_query_len,
            "capture_sizes": list(self.capture_sizes),
            "piecewise_eager_workload_kinds": list(
                self.piecewise_eager_workload_kinds),
        }
