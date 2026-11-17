"""Presentation categories; raw engine logs remain available for diagnostics."""
from collections import Counter, deque


BUSINESS = {
    'v1_admit': 'V1 整条路径传输开始', 'v1_service': 'V1 共享资源服务区间',
    'v1_p2p_complete': 'V1 点到点传输完成', 'v1_retry': 'V1 整次传输重试',
    'v1_failed': 'V1 通信失败', 'v1_comm_plan': 'V1 通信计划',
    'v1_request_state': 'V1 请求状态更新',
    'request_arrival': '请求到达', 'cell_input_ready': '输入就绪，等待 Cell 调度',
    'compute_start': '计算开始', 'prefill_complete': 'Prefill 完成',
    'compute_complete': '请求计算完成', 'request_complete': '请求完成',
    'cell_compute_complete': 'Cell 计算完成', 'compute_wait_for_cell': '等待 Cell 计算资源',
    'first_token_ready': '首 token 生成', 'kv_publish': 'KV 缓存就绪',
    'request_failed': '请求失败', 'first_delivered': '首 token 交付',
    'output_delivered': '全部输出交付', 'route_plan': '传输选路',
    'route_recovery': '恢复选路', 'route_wait': '等待可行路线',
    'retry_transfer': '重试传输', 'transfer_queued': '逐跳排队',
    'transmit_start': '逐跳开始发送', 'transmit_complete': '逐跳发送完成',
    'hop_arrival': '逐跳到达', 'transfer_cancelled': '传输中断',
    'bandwidth_update': '逐跳带宽分配', 'external_flow_done': '端到端传输完成',
    'surrogate_complete': 'ProtoBridge 通信完成',
}
NETWORK = {
    'link_state': '链路状态更新', 'topology_sample': '物理拓扑采样',
    'network_update': '网络状态更新', 'fault_injected': '故障生效',
    'fault_restored': '故障恢复', 'topology_command': '人工拓扑操作',
}
DIAGNOSTIC = {
    'external_callback': '内部回调', 'control_boundary': '在线控制检查点',
    'fault_batch': '故障批处理回调', 'cell_transfer_terminal': '消息终态通知',
}


def is_playback_event(event):
    """User-visible execution, excluding periodic snapshots and callbacks."""
    return event['kind'] in BUSINESS or event['kind'] in {
        'fault_injected', 'fault_restored', 'topology_command',
    }


def annotate(event):
    row = dict(event)
    kind = row['kind']
    if kind == 'v1_allocation':
        # A diagnostic table can contain every pool in the constellation.
        # HTTP/replay panels need counts, while the lossless native trace and
        # runtime_events archive retain the complete allocation evidence.
        capacities, flows = row.pop('capacities', None), row.pop('flows', None)
        row.update(capacity_pool_count=len(capacities) if capacities is not None else row.get('capacity_pool_count', row.get('capacity_snapshot', {}).get('pool_count', 0)),
                   flow_count=len(flows) if flows is not None else row.get('flow_count', 0))
    row['category'] = 'business' if kind in BUSINESS else 'network' if kind in NETWORK else 'diagnostic'
    row['event_label'] = (BUSINESS | NETWORK | DIAGNOSTIC).get(kind, '内部记录：' + kind)
    return row


class EventWindow:
    """Retain each category independently, so network churn cannot hide requests."""
    def __init__(self, limit=300):
        self.cursor = 0
        self.counts = Counter()
        self.rows = {key: deque(maxlen=limit) for key in ('business', 'network', 'diagnostic')}

    def capture(self, log):
        for i in range(self.cursor, len(log)):
            row = annotate(dict(log[i], event_order=i))
            self.counts[row['category']] += 1
            self.rows[row['category']].append(row)
        self.cursor = len(log)
        return dict(counts={key: self.counts[key] for key in self.rows},
                    recent={key: list(rows) for key, rows in self.rows.items()})


def request_milestones(result):
    """Recover documented milestones for results saved before semantic logging."""
    for request, state in result['requests'].items():
        for kind, field in (
            ('request_arrival', 'arrival_ns'), ('cell_input_ready', 'cell_ready_ns'),
            ('compute_complete', 'compute_complete_ns'), ('first_delivered', 'first_token_received_ns'),
            ('output_delivered', 'last_token_received_ns'),
        ):
            at = state.get(field)
            if at is not None and (kind != 'output_delivered' or state.get('status') == 'completed'):
                yield dict(kind=kind, time_s=at/1e9, request_id=request, cell_id=state.get('cell'),
                           record_source='saved_request_summary')
