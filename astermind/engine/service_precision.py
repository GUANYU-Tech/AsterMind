"""Check byte service against capacity with an explicit clock-rounding budget."""

from __future__ import annotations

import math
from astermind.engine.clock import TICKS_PER_SECOND


def bandwidth_check(row, *, native_ns_clock=False):
    required = (
        "interval_start_s",
        "interval_end_s",
        "allocated_bandwidth_bps",
        "link_capacity_bps",
        "bytes_sent",
    )
    if any(name not in row for name in required):
        return dict(state="fail", detail="缺少带宽服务区间或服务量证据")
    start, finish = row["interval_start_s"], row["interval_end_s"]
    rate, capacity, sent = (
        row["allocated_bandwidth_bps"],
        row["link_capacity_bps"],
        row["bytes_sent"],
    )
    values = (start, finish, rate, capacity, sent)
    if (
        not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        )
        or min(start, finish, rate, capacity, sent) < 0
        or finish < start
    ):
        return dict(state="fail", detail="服务区间、速率、容量或字节量无效")
    duration = finish - start
    if row.get('service_clock') == 'simgrid_float64_seconds':
        first,last=row.get('native_interval_start_s'),row.get('native_interval_end_s')
        error=row.get('native_service_error_bytes')
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (first,last,error)) or error<0 or last<first:
            return dict(state='fail',detail='SimGrid 原生区间或数值误差证据无效')
        if abs(first-start)>math.ulp(first) or abs(last-finish)>math.ulp(last):
            return dict(state='fail',detail='SimGrid 原生时钟与事件记录不一致')
        elapsed=last-first
        if elapsed==0:
            return dict(state='pass') if sent==0 and rate==0 else dict(state='fail',detail='SimGrid 零时长区间不能提供服务')
        rounding=max(math.ulp(sent)*8,abs(rate*elapsed/8)*1e-14,1e-18)
        if sent>capacity*elapsed/8+error+rounding or rate>capacity+(error+rounding)*8/elapsed or abs(rate*elapsed/8-sent)>rounding:
            return dict(state='fail',detail='SimGrid 服务量或速率超过容量与原生浮点误差边界')
        return dict(state='pass',native_error_bound_bytes=error)
    if 'interval_start_ticks' in row and 'interval_end_ticks' in row:
        try:
            first, last = int(row['interval_start_ticks']), int(row['interval_end_ticks'])
        except (ValueError, TypeError):
            return dict(state='fail', detail='整数时钟字段无效')
        if first < 0 or last < first:
            return dict(state='fail', detail='服务时钟倒退')
        duration = (last - first) / TICKS_PER_SECOND
        if abs(start - first / TICKS_PER_SECOND) > math.ulp(start) or abs(finish - last / TICKS_PER_SECOND) > math.ulp(finish):
            return dict(state='fail', detail='秒显示值与整数时钟不一致')
        # Only arithmetic roundoff remains; never grant an absolute-time ULP budget.
        tolerance = max(math.ulp(sent) * 8, abs(rate * duration / 8) * 1e-14, 1e-18)
        if rate > capacity * (1 + 1e-12) or sent > rate * duration / 8 + tolerance or (duration == 0 and sent):
            return dict(state='fail', detail='分配速率超过容量或服务字节超过实际时长可提供的服务量')
        return dict(state='pass')
    clock_error = (math.ulp(start) + math.ulp(finish)) / 2
    if native_ns_clock:
        native_start = row.get("native_interval_start_ns")
        native_finish = row.get("native_interval_end_ns")
        if native_start is not None and native_finish is not None:
            if (
                not all(
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(v)
                    for v in (native_start, native_finish)
                )
                or native_finish < native_start
            ):
                return dict(state="fail", detail="原生服务时钟无效")
            native_duration = (native_finish - native_start) / 1e9
            native_error = (math.ulp(native_start) + math.ulp(native_finish)) / 2e9
            if abs(native_duration - duration) > clock_error + native_error:
                return dict(state="fail", detail="原生服务区间与全局时间区间不一致")
            if native_duration > 0:
                duration = native_duration
                clock_error = native_error
            else:
                clock_error += native_error
        else:
            clock_error += (math.ulp(start * 1e9) + math.ulp(finish * 1e9)) / 2e9
    payload = row.get("payload_bytes", sent)
    if (
        not isinstance(payload, (int, float))
        or isinstance(payload, bool)
        or not math.isfinite(payload)
        or payload < 0
    ):
        return dict(state="fail", detail="服务负载字节量无效")
    byte_error = (
        capacity * clock_error / 8
        + 2 * math.ulp(payload)
        + (1e-6 if native_ns_clock else 0)
    )
    limit = capacity * duration / 8
    consistent = abs(rate * duration / 8 - sent) <= byte_error + max(
        1e-12, sent * 1e-12
    )
    strict = rate <= capacity * (1 + 1e-5) + 1e-3
    if duration == 0 and sent > 0:
        return dict(state="fail", detail="零时长区间不能提供正服务量")
    if strict and consistent:
        return dict(state="pass")
    facts = dict(
        duration_s=duration,
        bytes_sent=sent,
        capacity_bps=capacity,
        recorded_rate_bps=rate,
        excess_bytes=sent - limit,
        precision_bound_bytes=byte_error,
    )
    if consistent and capacity > 0 and sent - limit <= byte_error:
        return dict(
            state="precision",
            **facts,
            detail=f"时间精度提示：服务量超出反算容量 {sent - limit:.6g} B；该区间的时钟与字节精度界为 {byte_error:.6g} B",
        )
    return dict(
        state="fail",
        **facts,
        detail=f"带宽或服务字节超过容量：记录速率 {rate:.9g} bit/s，容量 {capacity:.9g} bit/s；服务超额 {sent - limit:.6g} B，精度界 {byte_error:.6g} B",
    )
