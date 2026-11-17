"""Personal viewer exports; never imported by the distributed simulation package."""

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

def export_runtime_views(root):
    root = Path(root)
    result = root / "experiment_result.json"
    if not result.exists():
        return
    runtime = SimpleNamespace(**json.loads(result.read_text(encoding="utf-8"))["runtime"])
    _write_runtime_playback(root, runtime)
    preview_records = []
    for row in runtime.request_records[:20]:
        rid = row["request"]["request_id"]
        path = [
            {
                k: leg.get(k)
                for k in (
                    "link_id",
                    "source_node_id",
                    "target_node_id",
                    "start_s",
                    "finish_s",
                    "arrival_s",
                    "stage",
                )
            }
            for leg in row["routing"]["selected_path"][:256]
        ]
        events = [
            e for e in runtime.state_summary["event_log"] if e.get("request_id") == rid
        ][:200]
        preview_records.append(
            {
                "request": row["request"],
                "timing": row["timing"],
                "outcome": row["outcome"],
                "decision": row["decision"],
                "path": path,
                "events": events,
            }
        )
    (root / "runtime_viewer.json").write_text(
        json.dumps(
            {
                "summary": runtime.request_summary,
                "metrics": runtime.inference_metrics,
                "records": preview_records,
                "record_limit": 20,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _write_runtime_playback(root, runtime):
    folder = root / "runtime_playback"
    folder.mkdir(exist_ok=True)
    provenance = getattr(runtime, "provenance", {})
    transport = provenance.get("network_transport", {"backend": "queue"})
    events = defaultdict(list)
    all_events = []
    topology_events = []
    flow_states = {}
    active_transfers = set()
    queued_transfers = set()
    propagating_transfers = set()
    active_compute = set()
    completed = set()
    failed = set()
    sources=list(runtime.state_summary['event_log'])
    sources=sorted(sources,key=lambda e:e['time_s'])
    link_reasons = {}
    for source in sources:
        rid = source.get("request_id")
        solar = False
        if source['kind'] == 'link_state':
            link_id = source['link_id']
            previous = link_reasons.get(link_id)
            link_reasons[link_id] = source.get('reason')
            solar = source.get('reason') == 'sun_exclusion' or previous == 'sun_exclusion'
            if solar:
                source = dict(source, source='sun', sun_transition='blocked' if source.get('reason') == 'sun_exclusion' else 'cleared')
        if source['kind']=='link_state' and not solar:
            topology_events.append(dict(source,event_order=len(all_events)-.5))
            continue
        if rid is None and not solar and source['kind'] not in {'fault_injected', 'fault_restored', 'topology_command'}:
            continue
        event = dict(source, event_order=len(all_events))
        if solar or source['kind'] in {'fault_injected', 'fault_restored', 'topology_command'}:
            topology_events.append(event)
        key = (rid, event["transfer_id"]) if event.get("transfer_id") else (
            rid, event.get("link_id"), event.get("source_node_id"),
            event.get("start_s"), event.get("stage"),
        )
        if len(all_events) % 200 == 0:
            context_name = f"context-{len(all_events) // 200:05d}.json.gz"
            with (
                (folder / context_name).open("wb") as raw,
                gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream,
            ):
                stream.write(
                    json.dumps(
                        list(flow_states.values()),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                )
        kind = event["kind"]
        flow_status = {
            "transfer_queued": "queued",
            "transmit_start": "transmitting",
            "transmit_complete": "propagating",
        }.get(kind)
        if kind == "bandwidth_update" and key in flow_states:
            flow_states[key]['sent_bytes'] = flow_states[key].get('sent_bytes', 0) + event.get('bytes_sent', 0)
            flow_states[key].update(allocated_bandwidth_bps=event.get("allocated_bandwidth_bps"),
                                    interval_start_s=event.get("interval_start_s"),
                                    interval_end_s=event.get("interval_end_s"))
        if flow_status:
            flow_states[key] = {
                **{k: v for k, v in event.items() if k != "kind"},
                "status": flow_status,
            }
        elif kind in {"hop_arrival", "transfer_cancelled"}:
            flow_states.pop(key, None)
        elif kind == "request_failed":
            flow_states = {k: v for k, v in flow_states.items() if k[0] != rid}
        if kind == "transfer_queued":
            queued_transfers.add(key)
        elif kind == "transmit_start":
            queued_transfers.discard(key)
            active_transfers.add(key)
        elif kind == "transmit_complete":
            active_transfers.discard(key)
            propagating_transfers.add(key)
        elif kind == "hop_arrival":
            propagating_transfers.discard(key)
        elif kind == "transfer_cancelled":
            queued_transfers.discard(key)
            active_transfers.discard(key)
            propagating_transfers.discard(key)
        elif kind == "compute_start":
            active_compute.add(rid)
        elif kind == "compute_complete":
            active_compute.discard(rid)
        elif kind == "request_complete":
            completed.add(rid)
        elif kind == "request_failed":
            failed.add(rid)
            active_compute.discard(rid)
            active_transfers = {k for k in active_transfers if k[0] != rid}
            queued_transfers = {k for k in queued_transfers if k[0] != rid}
            propagating_transfers = {k for k in propagating_transfers if k[0] != rid}
        event["traffic_counts"] = {
            "queued": len(queued_transfers),
            "transmitting": len(active_transfers),
            "propagating": len(propagating_transfers),
            "computing": len(active_compute),
            "completed": len(completed),
            "failed": len(failed),
        }
        event["request_artifact"] = (
            hashlib.sha256((rid or runtime.request_records[0]['request']['request_id']).encode()).hexdigest() + ".json.gz"
        )
        all_events.append(event)
        events[rid].append(event)
    event_pages = []
    with gzip.open(folder / 'topology.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(topology_events, stream, ensure_ascii=False, allow_nan=False)
    # Bounded topology pages include the last observed link states and active
    # causes at their start, so seeking never downloads a day of event history.
    topology_pages = []
    link_context, command_context, fault_context = {}, {}, {}
    for start in range(0, len(topology_events), 2000):
        batch = topology_events[start:start+2000]
        initial = sorted((*link_context.values(), *command_context.values(), *fault_context.values()),
                         key=lambda e: (e['time_s'], e['event_order']))
        name = f'topology-{start//2000:05d}.json.gz'
        with gzip.open(folder/name, 'wt', encoding='utf-8') as stream:
            json.dump(dict(initial=initial, events=batch), stream, ensure_ascii=False, allow_nan=False)
        topology_pages.append(dict(artifact=name, first_time_s=batch[0]['time_s'], last_time_s=batch[-1]['time_s'],
                                   first_event_order=batch[0]['event_order']))
        for event in batch:
            if event['kind'] == 'link_state':
                link_context[event['link_id']] = event
            elif event['kind'] == 'topology_command':
                command_context[event['target_id']] = event
            elif event['kind'] in {'fault_injected', 'fault_restored'}:
                key = tuple(event.get(k) for k in ('target_type', 'target_id', 'source', 'start_s', 'end_s'))
                if event['kind'] == 'fault_injected':
                    fault_context[key] = event
                else:
                    fault_context.pop(key, None)
    for start in range(0, len(all_events), 200):
        name = f"events-{start // 200:05d}.json.gz"
        rows = all_events[start : start + 200]
        with (
            (folder / name).open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream,
        ):
            stream.write(
                json.dumps(
                    rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
            )
        event_pages.append(
            {
                "artifact": name,
                "count": len(rows),
                "first_time_s": rows[0]["time_s"],
                "last_time_s": rows[-1]["time_s"],
                "initial_transfers_artifact": f"context-{start // 200:05d}.json.gz",
            }
        )
    catalog = []
    for record in runtime.request_records:
        request = record["request"]
        rid = request["request_id"]
        name = hashlib.sha256(rid.encode()).hexdigest() + ".json.gz"
        payload = {"record": record, "events": events.get(rid, [])}
        with (
            (folder / name).open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream,
        ):
            stream.write(
                json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
            )
        catalog.append(
            {
                "request_id": rid,
                "arrival_time_s": request["arrival_time_s"],
                "completed": record["outcome"]["completed"],
                "artifact": name,
            }
        )
    pages = []
    for start in range(0, len(catalog), 200):
        name = f"requests-{start // 200:05d}.json"
        rows = catalog[start : start + 200]
        (folder / name).write_text(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        pages.append({"artifact": name, "count": len(rows)})
    (folder / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "astermind.engine_playback.v1",
                "summary": runtime.request_summary,
                "metrics": runtime.inference_metrics,
                "requests": pages,
                "request_count": len(catalog),
                "event_pages": event_pages,
                "event_count": len(all_events),
                "event_semantics": "actual_runtime_events_independent_of_visual_interpolation",
                "has_scene_preview": False,
                "network_view": "../viewer",
                "environment": {
                    "index": "../environment-index",
                    "frames": "../frames/",
                    "position_semantics": "visual_itrs_hermite_physics_sample_hold",
                    "topology_pages": topology_pages,
                    "topology_events_artifact": "topology.json.gz",
                },
                "transport": {
                    "schema_version": "orbitserve.transfer_lifecycle.v1",
                    "unit": "whole_message_store_and_forward",
                    "identity": "transfer_id",
                    "backend": transport,
                    "bandwidth_sharing": transport.get("backend") in {"flowsim", "fluid", "simgrid", "ns3"},
                    "queue_model": provenance.get("communication_queue_model", "directed_link_reservation_and_contact_wait"),
                    "implemented_events": [
                        "transfer_queued",
                        "transmit_start",
                        "transmit_complete",
                        "hop_arrival",
                        "request_failed",
                        "bandwidth_update",
                        "transfer_cancelled",
                    ],
                    "extension_events": {
                        "transfer_dropped": {
                            "implemented": False,
                            "required_fields": [
                                "transfer_id",
                                "time_s",
                                "reason",
                                "dropped_bytes",
                            ],
                        },
                        "transfer_retried": {
                            "implemented": False,
                            "required_fields": [
                                "transfer_id",
                                "time_s",
                                "retry_of",
                                "attempt",
                            ],
                        },
                    },
                    "packet_loss_model": None,
                    "retransmission_model": None,
                    "failure_semantics": "request_failure_is_not_packet_loss",
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
