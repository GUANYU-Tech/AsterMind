"""Host resource limits for initialization; never affect simulated capacity."""

import os
from pathlib import Path

GIB = 1024**3
# Conservative startup allowance, not a measured per-Cell memory footprint.
WORKER_MEMORY_ALLOWANCE = GIB


def available_cpu_count():
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return max(1, count)


def available_memory_bytes():
    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            remaining = max(0, int(limit) - used)
            available = remaining if available is None else min(available, remaining)
    except (OSError, ValueError):
        pass
    return available


def startup_policy(cell_count, requested=None):
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError("cell_init_workers must be a positive integer or None")
    cpus, memory = available_cpu_count(), available_memory_bytes()
    memory_slots = None
    if memory is not None:
        reserve = max(GIB, memory // 5)
        memory_slots = max(1, (memory - reserve) // WORKER_MEMORY_ALLOWANCE)
    limit = requested if requested is not None else min(cpus, memory_slots or cpus)
    return dict(
        workers=min(cell_count, limit),
        source="explicit" if requested is not None else "automatic",
        cpu_slots=cpus,
        memory_slots=memory_slots,
        available_memory_bytes=memory,
        worker_memory_allowance_bytes=WORKER_MEMORY_ALLOWANCE,
    )
