"""Runtime-only fault inputs; physical samples never contain random outcomes."""

import math
import random
from bisect import bisect_left
from dataclasses import replace

RUNTIME_NETWORK_FIELDS = frozenset({
    'transport_model', 'faults', 'failure_policy', 'max_transfer_retries',
    'fault_timing', 'random_faults',
})


def physical_network_settings(value):
    return {k: v for k, v in value.items() if k not in RUNTIME_NETWORK_FIELDS}


def random_fault_configs(settings):
    """Independent node/link processes, accepting the original flat input too."""
    if not isinstance(settings, dict):
        raise ValueError('random_faults must be an object')
    if not {'nodes', 'links'}.intersection(settings):
        return (settings,) if settings else ()
    if set(settings) - {'nodes', 'links'}:
        raise ValueError('分组故障配置只允许 nodes 和 links，不可混用旧单组参数')
    configs = []
    for key, kind in (('nodes', 'node'), ('links', 'link')):
        config = settings.get(key, {})
        if not isinstance(config, dict) or config.get('target_type', kind) != kind:
            raise ValueError(f'{key} 必须是独立的 {kind} 故障配置')
        configs.append({**config, 'target_type': kind})
    return tuple(configs)


def random_faults_enabled(settings):
    return any(config.get('enabled') for config in random_fault_configs(settings))


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} 必须为有限正数')


def _duration_config(config):
    if config.get('model') == 'exponential':
        return dict(distribution='exponential', mean_steps=config.get('mean_down_steps'))
    return config.get('duration', dict(distribution='fixed', steps=config.get('duration_steps')))


def _validate_duration(config):
    duration = _duration_config(config)
    if not isinstance(duration, dict):
        raise ValueError('duration 必须包含持续时间分布和参数')
    distribution = duration.get('distribution')
    if distribution == 'fixed':
        if type(duration.get('steps')) is not int or duration['steps'] <= 0:
            raise ValueError('需要正整数 duration.steps 或 duration_steps；请将旧修复概率改为持续时间分布')
    elif distribution == 'uniform':
        low, high = duration.get('min_steps'), duration.get('max_steps')
        if any(type(v) is not int for v in (low, high)) or not 1 <= low <= high:
            raise ValueError('均匀持续时间需要整数 1 <= min_steps <= max_steps')
    elif distribution in {'exponential', 'lognormal'}:
        _positive(duration.get('mean_steps'), 'duration.mean_steps')
        if distribution == 'lognormal':
            _positive(duration.get('sigma'), 'duration.sigma')
    else:
        raise ValueError('持续时间分布应为 fixed、uniform、exponential 或 lognormal')


def validate_random_faults(settings):
    for config in random_fault_configs(settings):
        if config.get('enabled'):
            _validate_config(config)


def _validate_config(settings):
    if settings.get('target_type') not in {'node', 'link'} or type(settings.get('seed')) is not int:
        raise ValueError('random faults require node/link target_type and integer seed')
    if settings.get('time_basis') == 'seconds':
        _validate_seconds(settings)
        return
    if settings.get('model', 'fixed_ratio') not in {'fixed_ratio', 'per_step', 'exponential', 'weibull'}:
        raise ValueError('unsupported random fault model')
    model = settings.get('model', 'fixed_ratio')
    keys = ('ratio',) if model == 'fixed_ratio' else ('failure_probability',) if model == 'per_step' else ()
    for key in keys:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key} must be a finite probability in [0,1]')
    if 'start_step' in settings or 'end_step' in settings:
        if 'start_s' in settings or 'end_s' in settings:
            raise ValueError('随机故障不能同时使用时间片编号和秒数')
        start, end = settings.get('start_step'), settings.get('end_step')
        if any(type(v) is not int for v in (start, end)) or not 0 <= start < end:
            raise ValueError('随机故障需要整数时间片编号 0 <= start_step < end_step')
    else:
        # Older fixed-ratio documents used seconds. Keep those inputs readable.
        start, end = settings.get('start_s'), settings.get('end_s')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError('随机故障需要 start_step 和 end_step（旧输入可使用 start_s/end_s）')
    if model != 'fixed_ratio':
        _validate_duration(settings)
    if model == 'exponential':
        _positive(settings.get('mean_up_steps'), 'mean_up_steps')
    elif model == 'weibull':
        _positive(settings.get('up_scale_steps'), 'up_scale_steps')
        _positive(settings.get('up_shape'), 'up_shape')


def _boundary_index(time, boundaries):
    index = bisect_left(boundaries, time)
    for candidate in (index, index - 1):
        if 0 <= candidate < len(boundaries) and math.isclose(time, boundaries[candidate], rel_tol=0, abs_tol=1e-8):
            return candidate
    raise ValueError('故障与恢复时刻必须位于物理时间片边界')


def random_fault_window(settings, boundaries):
    """Resolve zero-based [start_step, end_step) and validate full repair times."""
    validate_random_faults(settings)
    if {'nodes', 'links'}.intersection(settings):
        return tuple(random_fault_window(config, boundaries) for config in random_fault_configs(settings))
    if not settings.get('enabled'):
        return None
    if settings.get('time_basis') == 'seconds':
        start, end = settings['start_s'], settings['end_s']
        if not boundaries[0] <= start < end <= boundaries[-1]:
            raise ValueError('连续随机故障注入区间必须位于仿真时域内')
        return start, end
    if 'start_step' in settings:
        start, end = settings['start_step'], settings['end_step']
    else:
        start, end = (_boundary_index(settings[k], boundaries) for k in ('start_s', 'end_s'))
    last = len(boundaries) - 1
    if end > last:
        raise ValueError(f'随机故障结束片超出仿真范围；最大边界编号为 {last}')
    if settings.get('model', 'fixed_ratio') != 'fixed_ratio':
        duration = _duration_config(settings)
        upper = duration.get('steps') if duration['distribution'] == 'fixed' else duration.get('max_steps') if duration['distribution'] == 'uniform' else None
        if upper is not None and end - 1 + upper > last:
            raise ValueError(f'最晚故障的恢复片超出仿真范围；end_step - 1 + 最大持续片数 必须 <= {last}')
    return start, end


def _step_time(index, boundaries):
    if index < len(boundaries):
        return boundaries[index]
    # Unbounded repair distributions can extend past the observed horizon.
    # Preserve the sampled duration rather than truncating or resampling it.
    step = boundaries[1] - boundaries[0]
    if not all(math.isclose(b - a, step, rel_tol=1e-9, abs_tol=1e-8) for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('超出仿真时域的随机恢复需要等间隔时间片')
    result = boundaries[0] + index * step
    if not math.isfinite(result):
        raise ValueError('故障分布参数生成了不可表示的恢复时间')
    return result


def _sample_duration(duration, rng):
    distribution = duration['distribution']
    if distribution == 'fixed':
        return duration['steps']
    if distribution == 'uniform':
        return rng.randint(duration['min_steps'], duration['max_steps'])
    mean = duration['mean_steps']
    try:
        value = (rng.expovariate(1 / mean) if distribution == 'exponential' else
                 rng.lognormvariate(math.log(mean) - duration['sigma'] ** 2 / 2, duration['sigma']))
        return max(1, math.ceil(value))
    except (OverflowError, ValueError) as error:
        raise ValueError('持续时间分布参数生成了不可表示的样本') from error


def fault_step_details(fault, boundaries):
    """Annotate actual fault logs without changing the canonical fault schema."""
    if not math.isfinite(fault['end_s']):
        return {}
    try:
        start = _boundary_index(fault['start_s'], boundaries)
        if fault['end_s'] <= boundaries[-1]:
            end = _boundary_index(fault['end_s'], boundaries)
        else:
            step = boundaries[1] - boundaries[0]
            end = round((fault['end_s'] - boundaries[0]) / step)
            if not math.isclose(_step_time(end, boundaries), fault['end_s'], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError('continuous repair')
    except ValueError:
        from bisect import bisect_right
        index = max(0, bisect_right(boundaries, fault['start_s']) - 1)
        return dict(duration_s=fault['end_s'] - fault['start_s'],
                    start_sample_index=index, start_offset_s=fault['start_s'] - boundaries[index],
                    right_censored=fault['end_s'] > boundaries[-1])
    return dict(start_step=start, end_step=end, duration_steps=end - start,
                duration_s=fault['end_s'] - fault['start_s'], right_censored=end >= len(boundaries))


def validate_fault_boundaries(spec, boundaries):
    if spec.fault_timing == 'continuous' and any(
            not boundaries[0] <= fault['start_s'] < boundaries[-1] for fault in spec.faults):
        raise ValueError('故障发生时刻必须位于仿真时域内')
    if spec.fault_timing == 'boundary' and any(
            c.get('enabled') and c.get('time_basis') == 'seconds'
            for c in random_fault_configs(spec.random_faults)):
        raise ValueError('秒级连续随机故障需要 fault_timing=continuous')
    if spec.fault_timing == 'boundary':
        for fault in spec.faults:
            _boundary_index(fault['start_s'], boundaries)
            if fault.get('source') == 'random' and fault['end_s'] > boundaries[-1]:
                if not fault_step_details(fault, boundaries):
                    raise ValueError('随机恢复时刻必须对齐延伸后的物理时间片边界')
            else:
                _boundary_index(fault['end_s'], boundaries)
    random_fault_window(spec.random_faults, boundaries)


def realize_runtime_faults(spec, boundaries, edges):
    """Expand reproducible input scenarios at Runtime, hidden from route forecasts."""
    validate_fault_boundaries(spec, boundaries)
    if not random_faults_enabled(spec.random_faults):
        return spec
    faults = [dict(f) for f in spec.faults]
    for settings in random_fault_configs(spec.random_faults):
        if not settings.get('enabled'):
            continue
        targets = sorted(n.node_id for n in spec.nodes if n.role != 'ground') if settings['target_type'] == 'node' else sorted(e['link_id'] for e in edges)
        rng = random.Random(settings['seed'])
        if settings.get('time_basis') == 'seconds':
            faults.extend(_realize_seconds(settings, targets, rng))
            continue
        def append(target, start, end):
            faults.append(dict(target_type=settings['target_type'], target_id=target,
                               start_s=_step_time(start, boundaries), end_s=_step_time(end, boundaries), source='random'))
        start, end = random_fault_window(settings, boundaries)
        model = settings.get('model', 'fixed_ratio')
        if model == 'fixed_ratio':
            for target in rng.sample(targets, math.ceil(len(targets) * settings['ratio'])):
                append(target, start, end)
        elif model == 'per_step':
            repaired_at = {}
            for step in range(start, end):
                for target in targets:
                    if step <= repaired_at.get(target, -1):
                        continue
                    if rng.random() < settings['failure_probability']:
                        repaired_at[target] = step + _sample_duration(_duration_config(settings), rng)
                        append(target, step, repaired_at[target])
        else:
            for target in targets:
                step = start
                while step < end:
                    try:
                        healthy = (rng.expovariate(1 / settings['mean_up_steps']) if model == 'exponential' else
                                   rng.weibullvariate(settings['up_scale_steps'], settings['up_shape']))
                        step += max(1, math.ceil(healthy))
                    except (OverflowError, ValueError) as error:
                        raise ValueError('健康期分布参数生成了不可表示的样本') from error
                    if step >= end:
                        break
                    repaired = step + _sample_duration(_duration_config(settings), rng)
                    append(target, step, repaired)
                    step = repaired
    return replace(spec, faults=tuple(faults), random_faults={})


def _validate_seconds(config):
    if any(key in config for key in ('start_step', 'end_step')):
        raise ValueError('连续随机故障使用 start_s/end_s，不可混用时间片编号')
    start, end = config.get('start_s'), config.get('end_s')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (start, end)) or not 0 <= start < end:
        raise ValueError('连续随机故障需要 0 <= start_s < end_s')
    model = config.get('model')
    if model == 'fixed_ratio':
        ratio = config.get('ratio')
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError('ratio must be a finite probability in [0,1]')
        return
    if model not in {'exponential', 'weibull'}:
        raise ValueError('连续模型支持 exponential、weibull、fixed_ratio')
    _positive(config.get('mean_up_s') if model == 'exponential' else config.get('up_scale_s'), '健康期秒数')
    if model == 'weibull':
        _positive(config.get('up_shape'), 'up_shape')
    duration = config.get('duration', {})
    if not isinstance(duration, dict):
        raise ValueError('duration 必须包含持续时间分布和参数')
    distribution = duration.get('distribution')
    if distribution == 'fixed':
        _positive(duration.get('seconds'), 'duration.seconds')
    elif distribution == 'uniform':
        _positive(duration.get('min_s'), 'duration.min_s')
        _positive(duration.get('max_s'), 'duration.max_s')
        if duration['min_s'] > duration['max_s']:
            raise ValueError('duration.min_s must be <= max_s')
    elif distribution in {'exponential', 'lognormal'}:
        _positive(duration.get('mean_s'), 'duration.mean_s')
        if distribution == 'lognormal':
            _positive(duration.get('sigma'), 'duration.sigma')
    else:
        raise ValueError('持续时间分布应为 fixed、uniform、exponential 或 lognormal')


def _realize_seconds(config, targets, rng):
    """Renewal processes in seconds; samples never depend on geometry cadence."""
    start, stop = config['start_s'], config['end_s']
    model, duration = config['model'], config.get('duration', {})
    if model == 'fixed_ratio':
        intervals = ((target, start, stop) for target in rng.sample(targets, math.ceil(len(targets)*config['ratio'])))
        return [dict(target_type=config['target_type'], target_id=t, start_s=a, end_s=b, source='random')
                for t, a, b in intervals]
    result = []
    for target in targets:
        time = start
        while time < stop:
            healthy = (rng.expovariate(1/config['mean_up_s']) if model == 'exponential'
                       else rng.weibullvariate(config['up_scale_s'], config['up_shape']))
            onset = time + healthy
            if onset >= stop:
                break
            dist = duration['distribution']
            down = (duration['seconds'] if dist == 'fixed' else
                    rng.uniform(duration['min_s'], duration['max_s']) if dist == 'uniform' else
                    rng.expovariate(1/duration['mean_s']) if dist == 'exponential' else
                    rng.lognormvariate(math.log(duration['mean_s'])-duration['sigma']**2/2, duration['sigma']))
            time = onset + down
            if not math.isfinite(time) or onset <= start and healthy == 0 or time <= onset:
                raise ValueError('故障分布参数生成了不可表示的时间')
            result.append(dict(target_type=config['target_type'], target_id=target,
                               start_s=onset, end_s=time, source='random'))
    return result
