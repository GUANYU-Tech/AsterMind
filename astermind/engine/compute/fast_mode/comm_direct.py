"""Evaluate native Direct waves with separate intra- and inter-satellite paths."""

from collections import Counter
from dataclasses import asdict
from math import floor

from astermind.engine.comm import intra_topology
from astermind.engine.comm.policy import resolve_task_policy
from astermind.engine.comm.task import CommunicationTask
from .comm_reuse import UnsupportedFormation


def mask_prefix(peers):
    """Index the paths selected by the existing deterministic rounding order."""
    prefixes = [{}]
    complete = {}
    for lane, wave in peers:
        bit = 1 << wave
        complete[lane] = complete.get(lane, 0) | bit
        prefixes.append(dict(complete))
    return tuple(tuple((lane, mask & ~head.get(lane, 0), head.get(lane, 0))
                       for lane, mask in complete.items()) for head in prefixes)


class DirectLegacyIntra:
    """The intra-lane branch that COMM_Backend uses inside a cross-satellite Direct plan."""

    def __init__(self, local):
        bandwidth = min(177000, local.bandwidth)
        self.model = intra_topology.theory.make_h200_sendrecv_theory(
            ring_bw_bytes_per_us=bandwidth, packet_bw_bytes_per_us=bandwidth,
            link_latency_us=local.latency)

    def ticks(self, payload, frames):
        return intra_topology._ns(self.model.predict(payload, 2, 'ring', 128, collective='sendrecv')) * 10**9


class DirectGroups:
    def __init__(self, fixed, quote):
        self.pricing = pricing = fixed.pricing
        program = pricing.program
        self.probability = program.hit_probability if pricing.program.dp > 1 else 0.
        self.intra = {}
        self.stages = tuple(tuple(self._bind(fixed, quote, g) for g in stage) for stage in program.experts)
        self.dispatch_widths = tuple(Counter(program.dispatch_widths).items())
        self.combine_width = program.combine_width

    def _bind(self, fixed, quote, group):
        pricing = fixed.pricing
        prototype = pricing._operator('all_to_all', group.ranks, 0)
        task = CommunicationTask(**asdict(prototype))
        _, _, selection = resolve_task_policy(fixed.runtime.policy, task, quote.routes)
        if selection['scope'] != 'inter_satellite' or selection['algorithm'] != 'pairwise':
            raise UnsupportedFormation('bound Direct requires a cross-satellite pairwise group')
        ranks = prototype.participants
        size = len(ranks)

        def lane(source, target):
            physical, cross = fixed.lanes(quote, source, target)
            if cross:
                return physical
            if physical not in self.intra:
                self.intra[physical] = DirectLegacyIntra(physical)
            return self.intra[physical]

        forward, reverse = [], []
        for row, targets in enumerate(pricing.destination_order[group.ranks]):
            forward.append(mask_prefix((lane(ranks[row], ranks[col]), (col - row) % size) for col in targets))
            reverse.append(mask_prefix((lane(ranks[col], ranks[row]), (row - col) % size) for col in targets))
        return group, tuple(forward), tuple(reverse)

    def _histograms(self, bound, batch):
        group, forward, reverse = bound
        counts = self.pricing.program._expert_rows(group.sources, batch.replica_tokens)
        dispatch, combine = {}, {}
        peers = len(group.ranks) - 1
        for index, tokens in enumerate(counts):
            mean = tokens * self.probability
            base = floor(mean)
            extra = floor((mean - base) * peers + .5)
            if base == 0 and extra == 0:
                continue
            for bins, prefixes in ((dispatch, forward), (combine, reverse)):
                for lane, regular, additional in prefixes[index][extra]:
                    if regular and base:
                        key = lane, base
                        bins[key] = bins.get(key, 0) | regular
                    if additional:
                        key = lane, base + 1
                        bins[key] = bins.get(key, 0) | additional
        return dispatch, combine

    @staticmethod
    def _waves(histogram, width, prices, frames):
        by_ticks = {}
        for (lane, tokens), mask in histogram.items():
            key = lane, tokens * width
            value = prices.get(key)
            if value is None:
                value = prices[key] = lane.ticks(key[1], frames)
            by_ticks[value] = by_ticks.get(value, 0) | mask
        claimed = total = 0
        for ticks in sorted(by_ticks, reverse=True):
            mask = by_ticks[ticks]
            total += (mask & ~claimed).bit_count() * ticks
            claimed |= mask
        return total

    def phase_ticks(self, bound, batch, prices, frames):
        dispatch, combine = self._histograms(bound, batch)
        return (tuple(self._waves(dispatch, width, prices, frames) for width, _ in self.dispatch_widths),
                self._waves(combine, self.combine_width, prices, frames))

    def price(self, stage, batch, prices, frames):
        def group_seconds(bound):
            dispatch, combine = self.phase_ticks(bound, batch, prices, frames)
            value = sum(n * (ticks / 10**18) for ticks, (_, n) in zip(dispatch, self.dispatch_widths, strict=True))
            return bound[0].repetitions * (value + combine / 10**18)

        return max(map(group_seconds, self.stages[stage]), default=0.)
