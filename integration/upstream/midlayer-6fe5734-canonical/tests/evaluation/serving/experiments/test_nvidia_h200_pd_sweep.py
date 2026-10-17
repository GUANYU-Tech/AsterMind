from argparse import Namespace

from evaluation.serving.experiments.nvidia_h200_pd_sweep import (
    average_scheduler_concurrency, cluster_observed_concurrency,
    point_timeout_seconds,
)


def test_low_capacity_timeout_is_an_explicit_override():
    args = Namespace(timeout_seconds=600.0, low_capacity_timeout_seconds=2700.0,
                     unbounded_low_capacities=False, unbounded_all_capacities=False)

    assert point_timeout_seconds(args, 1) == 2700.0
    assert point_timeout_seconds(args, 2) == 2700.0
    assert point_timeout_seconds(args, 4) == 600.0


def test_regular_timeout_remains_the_default_for_every_capacity():
    args = Namespace(timeout_seconds=600.0, low_capacity_timeout_seconds=None,
                     unbounded_low_capacities=False, unbounded_all_capacities=False)

    assert point_timeout_seconds(args, 1) == 600.0
    assert point_timeout_seconds(args, 128) == 600.0


def test_unbounded_low_capacities_keep_the_standard_high_capacity_guard():
    args = Namespace(timeout_seconds=600.0, low_capacity_timeout_seconds=2700.0,
                     unbounded_low_capacities=True, unbounded_all_capacities=False)

    assert point_timeout_seconds(args, 1) is None
    assert point_timeout_seconds(args, 2) is None
    assert point_timeout_seconds(args, 4) == 600.0


def test_unbounded_all_capacities_removes_every_wall_clock_guard():
    args = Namespace(timeout_seconds=600.0, low_capacity_timeout_seconds=None,
                     unbounded_low_capacities=False, unbounded_all_capacities=True)

    assert point_timeout_seconds(args, 1) is None
    assert point_timeout_seconds(args, 128) is None


def test_average_scheduler_concurrency_uses_scheduler_counters():
    payload = {'scheduler_progress': {'summary': {
        'scheduled_request_rows': 354, 'step_count': 100,
    }}}

    assert average_scheduler_concurrency(payload) == 3.54


def test_observed_concurrency_clusters_only_indistinguishable_points():
    points = [
        {'average_scheduler_concurrency': 1.99},
        {'average_scheduler_concurrency': 3.853},
        {'average_scheduler_concurrency': 3.9132},
        {'average_scheduler_concurrency': 3.9136},
    ]

    clusters = cluster_observed_concurrency(points)

    assert [[point['average_scheduler_concurrency'] for point in cluster]
            for cluster in clusters] == [[1.99], [3.853], [3.9132, 3.9136]]
