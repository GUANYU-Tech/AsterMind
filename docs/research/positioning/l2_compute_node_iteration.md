# L2 compute-node iteration contract

This iteration treats one Orbit/Runtime scheduling object as one logical
`compute_node`. A compute node contains `serving.formation.member_count`
physical satellites. Orbit does not expand those members or model their
internal links, and Runtime schedules, queues, accounts energy, and manages KV
cache only at compute-node granularity.

## Serving input

```json
{
  "serving": {
    "formation": {
      "member_count": 4,
      "topology": "ring",
      "link_bandwidth_gbps": 100.0,
      "link_propagation_delay_s": 0.00001
    }
  }
}
```

`topology` accepts `ring`, `switch`, or `fully_connected`.
`link_propagation_delay_s` is one-way, single-hop propagation delay. Bandwidth
is a direct configured input; this layer does not constrain it from the
Architecture terminal profile.

Serving first estimates one physical member with the unchanged Architecture
profile. It then uses the provisional analytical wrapper

`formation_stage_time = single_member_stage_time / member_count + internal_communication_time`

to produce the logical node's prefill/decode capacity and KV capacity. The
communication term is topology-aware but intentionally remains a first-order
model pending calibration and deeper Serving-model discussion.

## Cost aggregation

Experiments aggregate the closed Architecture cost as

`total_cost = compute_node_count * formation_member_count * single_satellite_cost`.

The single-satellite cost already includes the configured laser terminals,
so Experiments add no separate topology/link/terminal surcharge. Architecture
itself remains a single-physical-satellite model.

## Runtime KV cache

`runtime_strategy.kv_cache_copy_enabled` is a boolean. When enabled and the
selected compute node lacks sufficient reusable context, Runtime copies from
the reachable compute node holding the longest reusable prefix. The source is
retained; there is no migration ratio and no formation-internal KV policy.

Topology refresh and fixed-time state updates are outside this iteration.
