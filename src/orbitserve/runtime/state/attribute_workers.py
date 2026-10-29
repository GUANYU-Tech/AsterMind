"""Two-phase workers retain compact columns until the parent releases encoding."""

from queue import Empty

_prepared_queue = None
_encode_gate = None


def initialize(queue, gate):
    global _prepared_queue, _encode_gate
    _prepared_queue, _encode_gate = queue, gate
    _prepared_queue.cancel_join_thread()


def calculate_then_encode(item):
    from .attribute_cache import AttributeBlock
    from .physical_dataset import _calculate_chunk, _encode_prepared

    identity, payload = item
    try:
        raw = _calculate_chunk(payload)
        blocks = [
            (link, time_s, AttributeBlock(values), len(values))
            for link, time_s, values in raw
        ]
        del raw
        _prepared_queue.put((identity, blocks, None))
    except BaseException as error:
        _prepared_queue.put((identity, None, error))
        raise
    _encode_gate.wait()
    return _encode_prepared(blocks)


def receive_prepared(queue, futures):
    prepared = {}
    while len(prepared) < len(futures):
        try:
            identity, blocks, error = queue.get(timeout=1.0)
        except Empty:
            for future in futures:
                if future.done() and future.exception() is not None:
                    raise future.exception()
            continue
        if error is not None:
            raise error
        if identity in prepared:
            raise RuntimeError("duplicate physical attribute worker response")
        prepared[identity] = blocks
    return [prepared[i] for i in range(len(futures))]
