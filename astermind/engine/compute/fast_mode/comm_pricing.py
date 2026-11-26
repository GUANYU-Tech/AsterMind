"""Evaluate formula-derived physical tasks with COMM_Backend's no-contention planner."""

from collections import Counter, defaultdict
from dataclasses import replace
from itertools import chain, repeat
from math import floor
from random import Random

from astermind.engine.clock import Time
from astermind.engine.comm.native_intra import native_enabled
from astermind.engine.comm.operator import CommunicationOperator
from .frame_quote import FrameQuoteMemo


def integer_dispatch_rows(rows, probability, destination_order):
    """Round the off-diagonal mean row while keeping a bounded integer token count."""
    if len(rows) < 2 or not 0 <= probability <= 1:
        raise ValueError("invalid Direct expert group")
    peers = len(rows) - 1
    matrix = []
    for source, tokens in enumerate(rows):
        mean = tokens * probability
        base = floor(mean)
        extra = floor((mean - base) * peers + .5)
        counts = [base] * len(rows)
        counts[source] = 0
        for target in destination_order[source][:extra]:
            counts[target] += 1
        matrix.append(tuple(counts))
    return tuple(matrix)


class RankCommunicationPricing:
    """Preserve ordered physical participants and let policy select algorithms."""

    def __init__(self, runtime, program, devices, cell_id):
        if len(devices) != program.cards or len(set(devices)) != len(devices):
            raise ValueError("COMM_Backend requires one physical device per rank")
        if ((program.dp > 1 and getattr(runtime.policy, "ep_strategy", None) != "direct")
                or native_enabled(runtime.policy, cell_id)):
            raise ValueError("this rank program requires Direct EP and COMM_Backend surrogate pricing")
        self.runtime, self.program, self.devices, self.cell_id = runtime, program, devices, cell_id
        self.frame_quotes = FrameQuoteMemo()
        self.stages = tuple(tuple(Counter(stage).items()) for stage in program.stages)
        random = Random(42)
        self.destination_order = {
            group.ranks: tuple(tuple(random.sample(
                [target for target in range(len(group.ranks)) if target != source],
                len(group.ranks) - 1)) for source in range(len(group.ranks)))
            for groups in program.experts for group in groups
        }

    def _operator(self, operation, ranks, payload):
        semantics = "payload" if operation in ("send_recv", "broadcast") else "per_rank_input"
        reduce = operation in ("all_reduce", "reduce_scatterv")
        if operation == "reduce_scatterv" and payload % len(ranks):
            raise ValueError("TP sequence reduction requires uniform rank shards")
        shards = (payload // len(ranks),) * len(ranks) if operation == "reduce_scatterv" else ()
        return CommunicationOperator(operation, None, tuple(self.devices[r] for r in ranks),
                                     payload, semantics, reduction="sum" if reduce else None,
                                     partition_bytes=shards, cell_id=self.cell_id)

    def _expert_operations(self, group, batch):
        program = self.program
        source_tokens = program._expert_rows(group.sources, batch.replica_tokens)
        tokens = integer_dispatch_rows(source_tokens, program.hit_probability,
                                       self.destination_order[group.ranks])
        if not any(map(any, tokens)):
            return ()
        op = self._operator("all_to_all", group.ranks, 0)
        combined = tuple(zip(*tokens, strict=True))
        dispatch_widths = getattr(program, "dispatch_widths", (
            program.routed_width, program.p.top_k * 4, program.p.top_k * 4))
        combine_width = getattr(program, "combine_width", program.routed_width)

        def operator(rows, width):
            matrix = tuple(tuple(n * width for n in row) for row in rows)
            return replace(op, bytes=sum(map(sum, matrix)),
                           bytes_semantics="aggregate_input", traffic_matrix_bytes=matrix)

        return (*((operator(tokens, width), count) for width, count in Counter(dispatch_widths).items()),
                (operator(combined, combine_width), 1))

    def _expert_seconds(self, quote, group, batch):
        # Identical dispatch widths share one quote and still charge each role.
        operations = self._expert_operations(group, batch)
        if not operations:
            return 0.
        dispatch = sum(count * float(quote(op)) for op, count in operations[:-1])
        return group.repetitions * (dispatch + float(quote(operations[-1][0])))

    def runtime_phases(self, batch):
        """Release operations in forward order within the original compute budget."""
        program = self.program
        layers, output = program.p.estimate(batch.rows, runtime_detail=True)

        def operators(transfers):
            for transfer in transfers:
                size = transfer.payload(batch, program.tp)
                if size:
                    yield self._operator(transfer.operation, transfer.ranks, size)

        def group_operations(group):
            entries = self._expert_operations(group, batch)
            return tuple(chain.from_iterable(repeat(op, count) for op, count in entries))

        def replica_phase(tasks):
            replicas = [[] for _ in range(program.dp)]
            for transfer in tasks:
                replicas[transfer.replica].append(transfer)
            return tuple(operators(items) for items in replicas)

        yield tuple((op,) for op in operators(program.input_transfers))
        for stage_id, ((first, last), computation, stage, groups, boundary) in enumerate(zip(
                program.ranges, batch.stage_compute_s, program.stages, program.experts,
                program.boundaries, strict=True)):
            by_position = defaultdict(list)
            for transfer in stage:
                by_position[transfer.layer, transfer.role].append(transfer)

            final = stage_id == len(program.stages) - 1
            raw = [Time(value).ticks for row in layers[first:last] for value in row]
            raw.append(Time(output if final else 0.).ticks)
            original = Time(computation).ticks
            budget, offset, emitted = sum(raw), 0, 0

            def duration(weight):
                nonlocal offset, emitted
                offset += weight
                boundary = original * offset // budget if budget else 0
                delta = boundary - emitted
                emitted = boundary
                return Time.from_ticks(delta)

            def at(layer, *roles):
                return replica_phase(chain.from_iterable(by_position[layer, role] for role in roles))

            yield at(None, "embedding")
            expert_ops = tuple(ops for group in groups if (ops := group_operations(group)))
            for layer in range(first, last):
                i = (layer - first) * 4
                attention, router, ffn, restore = raw[i:i+4]
                yield at(layer, "input-all-reduce", "sp-attention-gather")
                yield duration(attention)
                yield at(layer, "attention", "sp-attention-scatter")
                if expert_ops and program._routed(layer):
                    yield duration(router)
                    yield tuple(ops[:-1] for ops in expert_ops)
                    yield duration(ffn)
                    yield tuple((ops[-1],) for ops in expert_ops)
                    yield duration(restore)
                else:
                    yield duration(router + ffn + restore)
                yield at(layer, "moe-sp-restore", "latent-all-reduce", "shared-all-reduce", "ffn")
            yield at(None, "final-pair", "final-hidden", "final-all-reduce")
            yield duration(raw[-1])
            yield at(None, "logits")
            if emitted != original:
                raise ValueError("runtime phases did not conserve the predicted stage compute time")
            for phase in boundary:
                yield tuple((op,) for op in operators(phase))

    def price(self, batch):
        from .quote import BatchCommunication

        quote = BatchCommunication(self.runtime)
        deadline = quote.routes.frame.valid_until_s
        key = batch.replica_tokens, batch.replica_logits
        return self.frame_quotes.price(quote, key, lambda selected: self.quote_batch(batch, selected)), deadline

    def quote_batch(self, batch, quote):
        """Keep the task inventory while swapping a certified communication estimator for calibration."""
        program = self.program
        ordinary = {}

        def transfer_seconds(transfer):
            payload = transfer.payload(batch, program.tp)
            if not payload:
                return 0.
            key = transfer.operation, transfer.ranks, payload
            if key not in ordinary:
                ordinary[key] = float(quote(self._operator(*key)))
            return ordinary[key]

        total = max(map(transfer_seconds, program.input_transfers), default=0.)
        for stage, boundary, groups in zip(self.stages, program.boundaries, program.experts, strict=True):
            replica_seconds = [0.] * program.dp
            for transfer, repetitions in stage:
                replica_seconds[transfer.replica] += repetitions * transfer_seconds(transfer)
            total += max(replica_seconds)
            total += max((self._expert_seconds(quote, group, batch) for group in groups), default=0.)
            total += sum(max(map(transfer_seconds, phase), default=0.) for phase in boundary)
        return total
