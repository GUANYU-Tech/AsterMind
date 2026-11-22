"""Model-aware rank traffic for a fixed H200 formation, without an online graph."""

from dataclasses import dataclass, field
from math import ceil, comb


@dataclass(frozen=True, slots=True)
class Transfer:
    operation: str
    role: str
    ranks: tuple
    per_row_bytes: int
    replica: int
    logits: bool = False
    split_rows: bool = False
    requires_multirow: bool = False
    # Timing reuse ignores the repeated layer; runtime releases each position.
    layer: int | None = field(default=None, compare=False)

    def payload(self, batch, tp):
        if self.logits:
            return self.per_row_bytes * batch.replica_logits[self.replica]
        n = batch.replica_tokens[self.replica]
        if self.requires_multirow and n <= 1:
            return 0
        return self.per_row_bytes * (ceil(n / tp) if self.split_rows else n)


@dataclass(frozen=True, slots=True)
class ExpertGroup:
    ranks: tuple
    sources: tuple
    repetitions: int


class RankProgram:
    """Write each model's communication rules from its source-level forward phases."""

    def __init__(self, predictor, ranges, cards):
        self.p, self.ranges, self.cards = predictor, tuple(ranges), cards
        self.tp, self.dp, self.pp = predictor.tp, predictor.dp, len(ranges)
        if self.tp * self.dp * self.pp != cards:
            raise ValueError("rank placement does not cover this deployment")
        self.width = predictor.hidden * 2
        self.routed_width = (predictor.model["routed_expert_hidden_size"] * 2
                             if predictor.family == "k3" else self.width)
        self.combine_width = self.routed_width
        meta = predictor.top_k * 4
        if predictor.family == "v3":
            self.dispatch_roles = ("dispatch-input", "dispatch-weights", "dispatch-ids", "dispatch-scales")
            self.dispatch_widths = (predictor.hidden, meta, meta, predictor.hidden // 128 * 4)
        elif predictor.family == "gqa":
            # The native Direct adapter packs selected FP32 weights and IDs
            # into the source position called dispatch-logits.
            self.dispatch_roles = ("dispatch-hidden", "dispatch-logits")
            self.dispatch_widths = (self.routed_width, 2 * meta)
        else:
            payload_role = {"k2": "dispatch-input", "k3": "dispatch-latent"}.get(
                predictor.family, "dispatch-hidden")
            self.dispatch_roles = (payload_role, "dispatch-weights", "dispatch-ids")
            self.dispatch_widths = (self.routed_width, meta, meta)
        self.expert_phase_order = ("route", *self.dispatch_roles, "experts", "combine")
        self.vocabulary = (ceil(predictor.model["vocab_size"] / 64) * 64 // self.tp) * 2
        self.stages, self.boundaries, self.experts = zip(*(
            self._stage(stage, first, last) for stage, (first, last) in enumerate(self.ranges)), strict=True)
        self.input_transfers = tuple(
            Transfer("broadcast", "input-ids", self.tp_ranks(0, replica), 8, replica)
            for replica in range(self.dp) if self.tp > 1
        )
        if self.dp > 1:
            owners = predictor.routed // predictor.ep
            self.hit_probability = 1 - comb(predictor.routed - owners, predictor.top_k) / comb(
                predictor.routed, predictor.top_k)

    def tp_ranks(self, stage, replica):
        first = (replica * self.pp + stage) * self.tp
        return tuple(range(first, first + self.tp))

    def _stage(self, stage, first, last):
        p = self.p
        tasks, boundaries, groups = [], [], []
        glm, v4, k3 = (p.family == "glm", p.family == "v4", p.family == "k3")
        family_mla = p.family in ("v3", "k2")
        across_layer_sp = self.dp > 1 and self.tp > 1 and self.pp == 1 and p.family != "gqa"
        if self.dp > 1:
            count = sum(self._routed(i) for i in range(first, last))
            if count:
                ranks = tuple(self.tp_ranks(stage, r) for r in range(self.dp))
                if (v4 or k3) and self.pp > 1:
                    groups.extend(ExpertGroup(tuple(items), tuple((r, -1) for r in range(self.dp)), count)
                                  for items in zip(*ranks, strict=True))
                else:
                    groups.append(ExpertGroup(sum(ranks, ()), tuple(
                        (r, lane) for r in range(self.dp) for lane in range(self.tp)), count))
        pp_phases = {}
        for r in range(self.dp):
            ranks = self.tp_ranks(stage, r)
            layer = None

            def tp_task(operation, role, width, *, logits=False, split=False, requires_multirow=False):
                if self.tp > 1:
                    tasks.append(Transfer(operation, role, ranks, width, r, logits, split, requires_multirow, layer))

            if stage == 0:
                tp_task("all_reduce", "embedding", self.width)
            for i in range(first, last):
                layer = i
                routed = self._routed(i)
                if across_layer_sp and (not family_mla or routed):
                    if not family_mla or (i > first and self._routed(i - 1)):
                        tp_task("all_gather", "sp-attention-gather", self.width, split=True,
                                requires_multirow=family_mla)
                    tp_task("reduce_scatterv", "sp-attention-scatter", self.width * self.tp, split=True)
                    continue
                if glm and i:
                    tp_task("all_reduce", "input-all-reduce", self.width)
                tp_task("all_reduce", "attention", self.width)
                if glm:
                    if routed and self.dp > 1:
                        tp_task("all_gather", "moe-sp-restore", self.width, split=True)
                elif routed and self.dp > 1 and (family_mla or p.family == "gqa"):
                    tp_task("all_gather", "moe-sp-restore", self.width, split=True)
                elif k3 and routed:
                    tp_task("all_reduce", "latent-all-reduce", self.routed_width)
                    tp_task("all_reduce", "shared-all-reduce", self.width)
                else:
                    tp_task("all_reduce", "ffn", self.width)
            layer = None
            if last == len(p.layers):
                if across_layer_sp:
                    if family_mla:
                        if self._routed(last - 1):
                            tp_task("all_gather", "final-pair", self.width * 2, split=True, requires_multirow=True)
                    else:
                        tp_task("all_gather", "final-hidden", self.width * p.model["hc_mult"] if v4
                                else self.width, split=True)
                elif glm:
                    tp_task("all_reduce", "final-all-reduce", self.width)
                tp_task("all_gather", "logits", self.vocabulary, logits=True)
            if stage + 1 < self.pp:
                payloads = [("hidden_states", self.width * p.model["hc_mult"] if v4 else self.width)]
                if not v4:
                    count = ceil(last / p.model["attn_res_block_size"]) if k3 else 1
                    payloads.append(("residual", self.width * count))
                targets = self.tp_ranks(stage + 1, r)
                for index, (role, width) in enumerate(payloads):
                    sends = pp_phases.setdefault(2 * index, [])
                    sends.extend(Transfer("send_recv", role, (a, b), width // self.tp, r)
                                 for a, b in zip(ranks, targets, strict=True))
                    if self.tp > 1:
                        pp_phases.setdefault(2 * index + 1, []).append(
                            Transfer("all_gather", role, targets, width // self.tp, r))
        boundaries.extend(phase for _, phase in sorted(pp_phases.items()))
        return tuple(tasks), tuple(boundaries), tuple(groups)

    def _routed(self, layer):
        family = self.p.family
        return (family == "v4" or (self.p.layers[layer][0] == "sparse" if family == "glm"
                                  else self.p.layers[layer][1] == "moe"))

    def _expert_rows(self, sources, tokens):
        return tuple(tokens[r] if lane == -1 else
                     max(0, min(ceil(tokens[r] / self.tp), tokens[r] - lane * ceil(tokens[r] / self.tp)))
                     for r, lane in sources)
