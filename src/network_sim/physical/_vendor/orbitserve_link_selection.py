"""Unmodified selection methods from OrbitServe communication.py.

Both generic degree limits and Walker neighbor selection are preserved.
Acquisition timing is managed by the new explicit-terminal controller.
"""
from typing import Any


def _edge_sort_key(
    edge: tuple[str, str, dict[str, Any], float],
) -> tuple[float, str, str, str]:
    endpoint_a, endpoint_b, row, distance_km = edge
    return (float(distance_km), endpoint_a, endpoint_b, str(row["link_id"]))


def _degree_limited_edges(
    candidates: tuple[tuple[str, str, dict[str, Any], float], ...],
    terminal_count: int,
) -> tuple[tuple[str, str, dict[str, Any], float], ...]:
    degree: dict[str, int] = {}
    selected: list[tuple[str, str, dict[str, Any], float]] = []
    for edge in candidates:
        endpoint_a, endpoint_b, _, _ = edge
        if degree.get(endpoint_a, 0) >= terminal_count:
            continue
        if degree.get(endpoint_b, 0) >= terminal_count:
            continue
        selected.append(edge)
        degree[endpoint_a] = degree.get(endpoint_a, 0) + 1
        degree[endpoint_b] = degree.get(endpoint_b, 0) + 1
    return tuple(selected)


class OrbitServeSelector:
    def __init__(self, topology, terminal_count):
        self.topology = topology
        self.isl_terminal_count = terminal_count

    def _select_terminal_edges(
        self,
        candidates: tuple[tuple[str, str, dict[str, Any], float], ...],
    ) -> tuple[tuple[str, str, dict[str, Any], float], ...]:
        """Select active links without exceeding the physical terminal layout."""

        topology = self.topology
        if topology is None:
            return _degree_limited_edges(
                tuple(sorted(candidates, key=_edge_sort_key)),
                self.isl_terminal_count,
            )

        same_plane: list[tuple[str, str, dict[str, Any], float]] = []
        cross_plane: dict[
            tuple[int, int], list[tuple[str, str, dict[str, Any], float]]
        ] = {}
        for edge in candidates:
            endpoint_a, endpoint_b, _, _ = edge
            plane_a = topology.plane_by_node.get(endpoint_a)
            plane_b = topology.plane_by_node.get(endpoint_b)
            if plane_a is None or plane_b is None:
                continue
            if plane_a == plane_b:
                if (endpoint_a, endpoint_b) in topology.same_plane_neighbor_pairs:
                    same_plane.append(edge)
                continue
            plane_pair = (
                (plane_a, plane_b)
                if plane_a < plane_b
                else (plane_b, plane_a)
            )
            if plane_pair in topology.adjacent_plane_pairs:
                cross_plane.setdefault(plane_pair, []).append(edge)

        selected_cross: list[tuple[str, str, dict[str, Any], float]] = []
        for plane_pair in sorted(cross_plane):
            used_in_boundary: set[str] = set()
            for edge in sorted(cross_plane[plane_pair], key=_edge_sort_key):
                endpoint_a, endpoint_b, _, _ = edge
                if endpoint_a in used_in_boundary or endpoint_b in used_in_boundary:
                    continue
                used_in_boundary.update((endpoint_a, endpoint_b))
                selected_cross.append(edge)

        # Four terminals mean two fixed ring links and at most one dynamically
        # selected neighbor across each of the two adjacent plane boundaries.
        prioritized = tuple(
            sorted(same_plane, key=_edge_sort_key)
            + sorted(selected_cross, key=_edge_sort_key)
        )
        return _degree_limited_edges(prioritized, self.isl_terminal_count)
