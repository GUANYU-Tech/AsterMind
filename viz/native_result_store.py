"""Persist executed evidence even when the final causal audit rejects it."""
import json
from pathlib import Path


def save_audited_result(path, result, *, auditor, default):
    path = Path(path)

    def save():
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8', buffering=1024 * 1024) as stream:
            json.dump(result, stream, default=default, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)

    try:
        result['audit'] = (auditor(result) if result['status'] == 'completed'
                           else dict(passed=False, error='Execution contains failed requests'))
    except Exception as error:
        result['audit'] = dict(passed=False, error_type=type(error).__name__, error=str(error))
        save()
        raise
    save()
