"""Bounded startup of independent native Cells before the event loop runs."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .startup_resources import startup_policy


def initialize_cells(configs, create, *, workers=None, progress=None):
    """Keep construction isolated and publish in configuration order.

    ``create`` must not submit work to the shared simulation timeline. Native
    Cell constructors only prepare local state and start isolated schedulers.
    Successful objects remain owned here until the whole group is ready.
    """
    configs = tuple(configs)
    workers = startup_policy(len(configs), workers)["workers"]
    if not configs:
        return {}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cell-startup")
    futures, pending, results = {}, set(), {}
    remaining = iter(configs)

    def refill():
        while len(pending) < workers:
            config = next(remaining, None)
            if config is None:
                break
            future = pool.submit(create, config)
            futures[future] = config.id
            pending.add(future)

    def report():
        if progress is not None:
            active = {futures[future] for future in pending}
            progress(
                completed=len(results),
                total=len(configs),
                workers=workers,
                active=[config.id for config in configs if config.id in active],
            )

    try:
        refill()
        report()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Resolve the whole completed group before launching more workers.
            for future in done:
                results[futures[future]] = future.result()
            refill()
            report()
    except BaseException as error:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        # Include workers which succeeded after another worker failed.
        for future in futures:
            if not future.cancelled() and future.exception() is None:
                try:
                    future.result().close()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"Cell {futures[future]} cleanup failed: {cleanup_error}"
                    )
        raise
    else:
        pool.shutdown(wait=True)
    return {config.id: results[config.id] for config in configs}
