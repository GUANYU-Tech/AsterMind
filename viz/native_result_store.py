"""Persist executed evidence even when the final causal audit rejects it."""
from pathlib import Path


def save_audited_result(path, result, *, auditor, default, progress=None):
    path = Path(path)
    from astermind.common.result_storage import write_result

    def save():
        if progress:
            progress('保存计算结果与共享容量快照')
        write_result(path, result, default=default)

    try:
        if progress:
            progress('校验请求因果与通信资源约束')
        result['audit'] = (auditor(result) if result['status'] == 'completed'
                           else dict(passed=False, error='Execution contains failed requests'))
    except Exception as error:
        result['audit'] = dict(passed=False, error_type=type(error).__name__, error=str(error))
        save()
        raise
    save()
