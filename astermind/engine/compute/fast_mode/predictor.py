"""Online model-family Roofline formulas. This module has no ArchPrism dependency."""

from collections import Counter
from dataclasses import dataclass
from math import ceil, comb, expm1, log1p

from . import mla_gqa
from .hardware import Hardware


BF16 = 2
FP8_BLOCK = 1 + 4 / (128 * 128)
MXFP4 = .5 + 1 / 32
MODEL_FAMILIES = {
    **mla_gqa.MODEL_FAMILIES,
    "deepseek-v4-flash": "v4",
    "deepseek-v4-pro": "v4",
    "glm-5.2": "glm",
    "glm-5.3": "glm",
    "kimi-k3": "k3",
}


@dataclass(frozen=True, slots=True)
class Row:
    phase: str
    query: int
    history: int
    replica: int = 0
    logits: int = 1


# Compatibility with archived formula probes.
K3_NATIVE_READ_H200 = Hardware()


@dataclass(frozen=True, slots=True)
class BatchEstimate:
    replica_tokens: tuple
    replica_logits: tuple
    stage_components: tuple
    # Stage-local time across all replicas. Routed experts are synchronized.
    stage_compute_s: tuple
    rows: tuple = ()


def causal_pairs(query, history, cap=None):
    rising = query if cap is None else min(query, max(0, cap - history))
    return (rising * (2 * history + rising + 1) // 2
            + (0 if cap is None else (query - rising) * cap))


def _compressed_prefix(n, interval, cap):
    end = n if cap is None else min(n, interval * cap - 1)
    whole, tail = divmod(end, interval)
    return interval * whole * (whole - 1) // 2 + whole * (tail + 1) + (
        0 if cap is None else (n - end) * cap)


def compressed_pairs(query, history, interval, cap=None):
    return (_compressed_prefix(history + query, interval, cap)
            - _compressed_prefix(history, interval, cap))


def expected_active(tokens, population, top_k):
    if not tokens:
        return 0.
    if top_k == population:
        return float(population)
    return -population * expm1(tokens * log1p(-top_k / population))


def single_token_peak(experts, top_k, ranks):
    """Mean largest owner load when one token selects distinct uniform experts."""
    owned = experts // ranks
    ways = comb(experts, top_k)
    mean = 0.
    for limit in range(min(owned, top_k)):
        owner = [comb(owned, j) for j in range(limit + 1)]
        counts = [1] + [0] * top_k
        for _ in range(ranks):
            counts = [sum(counts[total - j] * owner[j]
                          for j in range(min(limit, total) + 1))
                      for total in range(top_k + 1)]
        mean += 1 - counts[top_k] / ways
    return mean


class Predictor:
    """Share identical layer formulas within each batch, preserving layer types."""

    def __init__(self, name, model, tp, dp, ep, ranges, *, hardware=Hardware(), force_sparse_glm=None,
                 glm_dsa_bf16_timing=False, small_fp8_gemm=None, grouped_glm_ffn=False):
        self.name = name
        self.family = MODEL_FAMILIES.get(name)
        if self.family is None:
            raise ValueError("unreviewed model formula")
        self.model = model.get("text_config", model)
        self.tp, self.dp, self.ep = tp, dp, ep
        self.hardware = hardware
        self.small_fp8_gemm = small_fp8_gemm
        self.glm_dsa_bf16_timing = glm_dsa_bf16_timing
        if grouped_glm_ffn and (name != "glm-5.2" or dp != 1 or ep != 1):
            raise ValueError("grouped GLM FFN timing requires GLM-5.2 with DP1 and EP1")
        self.grouped_glm_ffn = grouped_glm_ffn
        # Use the sparse-prefill timing policy for both pinned GLM models.
        self.force_sparse_glm = self.family == "glm" if force_sparse_glm is None else force_sparse_glm
        self.extra_layout = None
        m = self.model
        layer_count = m["num_hidden_layers"]
        if (not ranges or ranges[0][0] != 0 or ranges[-1][1] != layer_count
                or any(b <= a or (i and a != ranges[i-1][1]) for i, (a, b) in enumerate(ranges))
                or any(type(x) is not int or x <= 0 for x in (tp, dp, ep))
                or m["num_attention_heads"] % tp
                or ep not in (1, tp * dp) or dp > 1 and ep == 1):
            raise ValueError("invalid or unsupported formula layout")
        if self.family == "v4":
            # The checkpoint's final compression entry is the appended MTP layer.
            compression = tuple(m["compress_ratios"][:layer_count])
            if (len(compression) != layer_count or not set(compression) <= {0, 4, 128} or m["o_groups"] % tp
                    or m["expert_dtype"] != "fp4"):
                raise ValueError("unsupported V4 output sharding or compression layout")
            kinds = tuple((ratio, layer < m["num_hash_layers"]) for layer, ratio in enumerate(compression))
            # TP can exceed output groups only under a separate source-backed partition.
            self.routed = m["n_routed_experts"]
            self.top_k = m["num_experts_per_tok"]
        elif self.family == "glm":
            kinds = tuple(zip(m["mlp_layer_types"], m["indexer_types"], strict=True))
            if len(kinds) != layer_count or any(m["indexer_types"][a] != "full" for a, _ in ranges):
                raise ValueError("each GLM pipeline stage must start at a full indexer")
            if name == "glm-5.2" and m.get("quantization_config"):
                raise ValueError("the GLM-5.2 profile requires BF16 checkpoint weights")
            self.routed = m["n_routed_experts"]
            self.top_k = m["num_experts_per_tok"]
        elif self.family == "k3":
            linear = m["linear_attn_config"]
            kda, nope = set(linear["kda_layers"]), set(linear["full_attn_layers"])
            if kda & nope or kda | nope != set(range(1, layer_count + 1)) or linear["num_heads"] % tp:
                raise ValueError("expected K3's one-based KDA layer IDs")
            kinds = tuple(("kda" if layer + 1 in kda else "nope",
                           "dense" if layer < m["first_k_dense_replace"] else "moe")
                          for layer in range(layer_count))
            self.routed = m["num_experts"]
            self.top_k = m["num_experts_per_token"]
        else:
            self.extra_layout = mla_gqa.family_layout(name, m, tp, dp, ep, len(ranges))
            kinds = tuple((self.family, kind) for kind in self.extra_layout.layer_types)
            self.routed, self.top_k = self.extra_layout.routed, self.extra_layout.top_k
        if self.routed % ep:
            raise ValueError("expert ownership must divide evenly")
        self.dp1_peak = single_token_peak(self.routed, self.top_k, ep) if dp == 1 and ep > 1 else 0.
        self.layers = kinds
        self.stage_kinds = tuple(Counter(kinds[a:b]) for a, b in ranges)
        self.unique_kinds = tuple(dict.fromkeys(kinds))
        if self.family == "glm":
            self.attention_keys = dict.fromkeys(self.unique_kinds)
            self.dense_kinds = {kind for kind in self.unique_kinds if kind[0] == "dense"}
        elif self.family == "k3" or self.extra_layout is not None:
            self.attention_keys = {kind: kind[0] for kind in self.unique_kinds}
            self.dense_kinds = {kind for kind in self.unique_kinds if kind[1] == "dense"}
        else:
            self.attention_keys = {kind: kind[0] for kind in self.unique_kinds}
            self.dense_kinds = set()
        self.unique_attention = tuple(dict.fromkeys(self.attention_keys.values()))
        self.layer_formula = {"v4": self._v4, "glm": self._glm, "k3": self._k3}.get(
            self.family, self._source_family_attention)
        self.local_heads = m["num_attention_heads"] // tp
        self.hidden = m["hidden_size"]
        self.linear_dtype = BF16 if name in ("glm-5.2", "kimi-k3") else FP8_BLOCK
        self.routed_dtype = self.linear_dtype if self.family == "glm" else MXFP4
        if self.extra_layout is not None:
            self.linear_dtype = self.extra_layout.linear_weight_bytes
            self.routed_dtype = self.extra_layout.expert_weight_bytes

    def _source_family_attention(self, n, rows, family):
        return mla_gqa.attention(family, self.model, self.tp, self.hardware, n, rows)

    def _linear(self, weights, n, *, dtype=None, operands=0, active=None):
        dtype = self.linear_dtype if dtype is None else dtype
        return self.hardware.roofline(2 * n * (weights if active is None else active),
                                      weights * dtype + operands, fp8=dtype == FP8_BLOCK)

    def _ffn(self, n, total, dense=False, *, split=False):
        if self.extra_layout is not None:
            return mla_gqa.feed_forward(self.extra_layout, self.model, self.tp, self.dp, self.ep,
                                        len(self.stage_kinds), self.hardware, n, total, dense, self.dp1_peak,
                                        split=split)
        m, h = self.model, self.hidden
        if self.grouped_glm_ffn:
            if dense:
                estimate = self._linear(3 * h * m["intermediate_size"] // self.tp, n)
                return (estimate, 0., 0.) if split else estimate
            # The archived Case Study III prices shared and routed GEMMs as
            # one Roofline group, using the expected distinct routed weights.
            gate = self._linear(h * self.routed, n)
            expert = 3 * h * m["moe_intermediate_size"] // self.tp
            shared = m["n_shared_experts"]
            active = expected_active(n, self.routed, self.top_k)
            estimate = gate + self._linear(expert * (shared + active), n,
                                           active=expert * (shared + self.top_k))
            return (estimate, gate, 0.) if split else estimate
        model_sequence_sharded = self.dp > 1 and self.tp > 1 and len(self.stage_kinds) == 1
        if dense:
            local_n = ceil(n / self.tp) if model_sequence_sharded else n
            weight_shards = 1 if model_sequence_sharded else self.tp
            estimate = (self._linear(3 * h * m["intermediate_size"] // weight_shards, local_n,
                                     operands=4 * local_n * h * BF16) if n else 0.)
            return (estimate, 0., 0.) if split else estimate
        if self.family == "k3":
            width, shared = m["routed_expert_hidden_size"], m["num_shared_experts"]
        else:
            width, shared = h, m["n_shared_experts"]
        intermediate = m["moe_intermediate_size"]
        owners = self.routed // self.ep
        local_intermediate = intermediate // (self.tp if self.ep == 1 else 1)
        if self.family == "k3" and self.ep == 1:
            local_intermediate = max(256, local_intermediate)
        ranked_expert = 3 * width * local_intermediate
        active = expected_active(total, self.routed, self.top_k) * owners / self.routed
        expert_tokens = total / self.ep
        if total == 1 and self.dp1_peak:
            active = self.dp1_peak
            expert_tokens = self.dp1_peak / self.top_k
        routed = self._linear(ranked_expert * active, expert_tokens, dtype=self.routed_dtype,
                              active=ranked_expert * self.top_k,
                              operands=4 * expert_tokens * self.top_k * width * BF16)
        if not n:
            return (routed, 0., 0.) if split else routed
        sequence_sharded = self.dp > 1 and (self.family == "glm" or len(self.stage_kinds) == 1)
        local_n = ceil(n / self.tp) if sequence_sharded else n
        shared_tp = 1 if sequence_sharded else self.tp
        gate = self._linear(h * self.routed, local_n, dtype=BF16, operands=local_n * self.routed * 4)
        if self.small_fp8_gemm is None:
            shared_time = self._linear(3 * h * intermediate * shared // shared_tp, local_n)
        else:
            # The provider shards only complete FP8 scale blocks.
            columns = ceil(intermediate * shared / (128 * shared_tp)) * 128
            shared_time = (self.small_fp8_gemm(local_n, 2 * columns, h)
                           + self.small_fp8_gemm(local_n, h, columns))
        latent = self._linear(2 * h * width, local_n, dtype=BF16) if width != h else 0.
        estimate = routed + gate + shared_time + latent
        return (estimate, gate + latent / 2, latent / 2) if split else estimate

    def _glm(self, n, rows, _):
        m, h, heads = self.model, self.hidden, self.local_heads
        latent, rope = m["kv_lora_rank"], m["qk_rope_head_dim"]
        nope, val, qrank = m["qk_nope_head_dim"], m["v_head_dim"], m["q_lora_rank"]
        fresh = (not self.force_sparse_glm and all(row.phase == "prefill" and not row.history
                 and row.query <= m["index_topk"] for row in rows))
        visible = sum(causal_pairs(r.query, r.history) for r in rows)
        projection = self._linear(h * (qrank + latent + rope)
                                  + qrank * heads * (nope + rope) + heads * val * h,
                                  n, operands=4 * n * h * BF16)
        projection += self._linear(heads * latent * nope, n, dtype=BF16)
        if fresh:
            projection += self._linear(heads * latent * (nope + val), n)
            attn = self.hardware.roofline(2 * visible * heads * (nope + rope + val),
                                          BF16 * n * heads * (nope + rope + val))
        else:
            projection += self._linear(heads * latent * val, n, dtype=BF16)
            selected = sum(causal_pairs(r.query, r.history, m["index_topk"]) for r in rows)
            attn = self.hardware.roofline(2 * selected * heads * (2 * latent + rope),
                                          BF16 * selected * (latent + rope))
        d, ih = m["index_head_dim"], m["index_n_heads"]
        index_time = self._linear(h * (d + ih) + qrank * ih * d, n)
        # An explicitly selected timing proxy changes only the score compute
        # rate. The source's packed index-cache layout and byte count stay intact.
        index_time += self.hardware.roofline(2 * visible * ih * d, visible * (d + 4),
                                            fp8=not self.glm_dsa_bf16_timing)
        attn += self.hardware.roofline(0, n * (latent + rope) * BF16)
        return projection, attn, index_time

    def _v4(self, n, rows, interval):
        m, h, heads = self.model, self.hidden, self.local_heads
        d, qrank, orank = m["head_dim"], m["q_lora_rank"], m["o_lora_rank"]
        groups = m["o_groups"] // self.tp
        mhc_bytes = 4 * m["hc_mult"] * n * h * BF16
        use_small_gemm = self.small_fp8_gemm is not None
        if use_small_gemm:
            projection = (self.small_fp8_gemm(n, qrank + d, h)
                          + self.small_fp8_gemm(n, heads * d, qrank)
                          + self.small_fp8_gemm(n, orank, heads * d // groups, groups=groups, prequantized=True)
                          + self.small_fp8_gemm(n, h, groups * orank)
                          + self.hardware.roofline(0, mhc_bytes))
        else:
            projection = self._linear(h * (qrank + d) + qrank * heads * d + heads * d * orank
                                      + groups * orank * h, n, operands=mhc_bytes)
        window = m["sliding_window"]
        prefill = tuple(r for r in rows if r.phase == "prefill")
        decode = tuple(r for r in rows if r.phase == "decode")
        index_time = 0.
        if interval:
            cap = m["index_topk"] if interval == 4 else None
            # The overlapping C4 compressor projects two streams of twice the head width.
            projection += self._linear(2 * h * d * (2 if interval == 4 else 1), n, dtype=BF16)
            if interval == 4:
                ih, d_index = m["index_n_heads"], m["index_head_dim"]
                index_time = self._linear(h * (ih + 4 * d_index), n, dtype=BF16)
                if any((r.history + r.query) // interval > cap for r in rows):
                    scan = sum(compressed_pairs(r.query, r.history, interval) for r in rows)
                    index_time += (self.small_fp8_gemm(n, ih * d_index, qrank) if use_small_gemm
                                   else self._linear(qrank * ih * d_index, n))
                    index_time += self.hardware.roofline(2 * scan * ih * d_index, scan * 132, fp8=True)
        else:
            cap = None
        prefill_pairs = sum(causal_pairs(r.query, r.history, window) +
                            (compressed_pairs(r.query, r.history, interval, cap) if interval else 0)
                            for r in prefill)
        decode_pairs = sum(causal_pairs(r.query, r.history, window) +
                           (compressed_pairs(r.query, r.history, interval, cap) if interval else 0)
                           for r in decode)
        gathered = sum(r.query + min(r.history, window - 1) +
                       ((r.query + r.history) // interval if interval else 0) for r in prefill)
        padded_heads = max(64, heads)
        attn = self.hardware.roofline(4 * prefill_pairs * padded_heads * d,
                                      gathered * (584 + 2 * d) + prefill_pairs * 4 * d
                                      + sum(r.query for r in prefill) * padded_heads * 4 * d)
        attn += self.hardware.roofline(4 * decode_pairs * padded_heads * d,
                                       decode_pairs * 584 + sum(r.query for r in decode) * padded_heads * 4 * d)
        return projection, attn, index_time

    def _k3(self, n, rows, attention):
        m, h, heads = self.model, self.hidden, self.local_heads
        if attention == "kda":
            linear = m["linear_attn_config"]
            d = linear["head_dim"]
            heads = linear["num_heads"] // self.tp
            projection = self._linear(h * ceil((4 * heads * d + d + heads) / 16) * 16
                                      + d * heads * d + h * heads * d, n, dtype=BF16)
            decoded = sum(r.query for r in rows if r.phase == "decode")
            chunks = sum(ceil(r.query / 64) for r in rows if r.phase == "prefill")
            initial = sum(r.phase == "prefill" and r.history > 0 for r in rows)
            state_traffic = 4 * heads * d * d * (2 * decoded + chunks + initial)
            ops = heads * (6 * n * d * d + 9 * n * d + n + 4 * chunks * 64 * 64 * d)
            context = self.hardware.roofline(ops, state_traffic + 12 * heads * n * d * BF16)
        else:
            latent, rope = m["kv_lora_rank"], m["qk_rope_head_dim"]
            nope, value, rank = m["qk_nope_head_dim"], m["v_head_dim"], m["q_lora_rank"]
            projection = self._linear(h * (rank + latent + rope)
                                      + rank * heads * (nope + rope) + 2 * heads * value * h, n)
            decode_visible = sum(causal_pairs(r.query, r.history) for r in rows if r.phase == "decode")
            prefill_visible = sum(causal_pairs(r.query, r.history) for r in rows if r.phase == "prefill")
            context = self.hardware.roofline(2 * decode_visible * heads * (2 * latent + rope),
                                             BF16 * decode_visible * (latent + rope))
            prefill_context = sum(r.query + r.history for r in rows if r.phase == "prefill")
            context += self.hardware.roofline(2 * prefill_visible * heads * (nope + rope + value),
                                              BF16 * prefill_context * (latent + rope))
            if decode_visible:
                projection += self._linear(heads * latent * (nope + value),
                                           sum(r.query for r in rows if r.phase == "decode"))
            if prefill_visible:
                projection += self._linear(heads * latent * (nope + value), prefill_context)
        return projection, context, 0.

    def estimate(self, rows, *, runtime_detail=False):
        rows = tuple(rows)
        replicas = [[] for _ in range(self.dp)]
        for r in rows:
            if (r.phase not in ("prefill", "decode") or r.query <= 0 or r.history < 0
                    or not 0 <= r.replica < self.dp or not 0 <= r.logits <= r.query):
                raise ValueError("invalid scheduled rows")
            replicas[r.replica].append(r)
        sizes = tuple(sum(r.query for r in local) for local in replicas)
        logits = tuple(sum(r.logits for r in local) for local in replicas)
        total = sum(sizes)
        if not total:
            raise ValueError("expected an active batch")
        expert_rows = (sum(ceil(max(1, size) / self.tp) * self.tp for size in sizes)
                       if self.dp > 1 and (self.family in ("glm", "v3", "k2", "gqa") or len(self.stage_kinds) == 1)
                       else total)
        # Equivalent layer families are evaluated once per replica.
        per_replica = []
        for local, n in zip(replicas, sizes, strict=True):
            attention = {key: self.layer_formula(n, local, key) for key in self.unique_attention} if n else {}
            ffn = {}
            values = {}
            for kind in self.unique_kinds:
                routed_rows = (total if self.family == "v4" and kind[1]
                               else expert_rows)
                dense = kind in self.dense_kinds
                ffn_key = (dense, routed_rows)
                if ffn_key not in ffn:
                    if n or self.dp > 1:
                        ffn[ffn_key] = self._ffn(n, routed_rows, dense, split=runtime_detail)
                    else:
                        ffn[ffn_key] = (0., 0., 0.) if runtime_detail else 0.
                ff = ffn[ffn_key]
                ff = ff if runtime_detail else (ff,)
                if n:
                    proj, attn, index = attention[self.attention_keys[kind]]
                    if self.family == "glm" and kind[1] != "full":
                        index = 0.
                    values[kind] = (proj, attn, index, *ff)
                else:
                    values[kind] = (0., 0., 0., *ff)
            per_replica.append(values)
        vocab = ceil(self.model["vocab_size"] / 64) * 64 // self.tp
        output = self._linear(self.hidden * vocab, max(logits), dtype=BF16) if max(logits) else 0.
        if runtime_detail:
            # Relative placement of compute within the original stage budget.
            layers = []
            for kind in self.layers:
                attention = max(sum(replica[kind][:3]) for replica in per_replica)
                _, _, _, ffn, route, restore = max(
                    (replica[kind] for replica in per_replica), key=lambda values: values[3])
                layers.append((attention, route, max(0., ffn - route - restore), restore))
            return tuple(layers), output
        stage_components = []
        for counts in self.stage_kinds:
            stage_components.append(tuple(max(sum(count * values[k][i] for k, count in counts.items())
                                              for values in per_replica) for i in range(4)))
        times = [sum(row) for row in stage_components]
        times[-1] += output
        return BatchEstimate(sizes, logits, tuple(stage_components), tuple(times), rows)
