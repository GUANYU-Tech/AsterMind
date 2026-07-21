"""Deployment catalog helpers for the runtime evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from ...common import to_jsonable
from .deployment_contracts import RuntimeDeployment, RuntimeDeploymentCatalog


PD_PLACEMENT_SCOPES = frozenset(
    {
        "intra_cluster",
        "same_plane_cross_cluster",
        "cross_plane",
        "hybrid",
        "hybrid_explicit_pairs",
        "hybrid_template_mix",
        "unscoped_disaggregated",
    }
)






def build_runtime_deployment_catalog(scenario: Any) -> RuntimeDeploymentCatalog:
    """Build a deterministic runtime deployment catalog.

    Colocated serving keeps the legacy aggregate logical pool.  Disaggregated PD
    serving gets explicit prefill/decode deployment groups when the serving
    throughput estimate provides canonical group counts and group size fields.
    """

    serving_deployments = tuple(getattr(scenario, "serving_deployments", ()) or ())
    constellation = _mapping(getattr(scenario, "constellation", None))
    total_satellites = _constellation_satellite_count(constellation)
    if not serving_deployments:
        return RuntimeDeploymentCatalog(
            warnings=("runtime deployment catalog is empty because serving_deployments is empty",),
            provenance={"source": "runtime_deployments.build_runtime_deployment_catalog"},
        )

    primary = serving_deployments[0]
    payload = _mapping(primary)
    throughput_payload = _mapping(payload.get("throughput_estimate"))
    serving_payload = _mapping(payload.get("serving_spec"))
    pd_mode = (
        _optional_str(payload.get("pd_mode"))
        or _optional_str(throughput_payload.get("pd_mode"))
        or _optional_str(serving_payload.get("pd_mode"))
    )
    if pd_mode == "disaggregated":
        return _build_disaggregated_catalog(
            primary=primary,
            payload=payload,
            throughput_payload=throughput_payload,
            serving_payload=serving_payload,
            constellation=constellation,
            total_satellites=total_satellites,
        )
    return _build_aggregate_catalog(
        payload=payload,
        throughput_payload=throughput_payload,
        constellation=constellation,
        total_satellites=total_satellites,
    )


def _build_aggregate_catalog(
    *,
    payload: dict[str, Any],
    throughput_payload: dict[str, Any],
    constellation: dict[str, Any],
    total_satellites: int | None,
) -> RuntimeDeploymentCatalog:
    deployment_id = str(payload.get("deployment_id") or "deployment_0")
    satellites_per_deployment = _optional_int(
        _first_present(payload.get("satellites_per_deployment"), throughput_payload.get("satellites_per_deployment"))
    )
    member_satellites = _explicit_member_satellites(payload)
    if not member_satellites:
        member_count = (
            _serving_satellite_budget_count(payload, throughput_payload)
            or total_satellites
            or satellites_per_deployment
            or 1
        )
        member_satellites = _synthetic_satellite_ids(max(1, member_count))
    budget_payload = _serving_satellite_budget_payload(payload, throughput_payload)
    warnings = list(_string_tuple(payload.get("warnings")))
    warnings.append(
        "Phase 1a uses an aggregate logical deployment pool; topology-aware placement is not implemented"
    )
    prefill_tokens_per_s = _optional_float(
        _first_present(payload.get("prefill_tokens_per_s"), throughput_payload.get("prefill_tokens_per_s"))
    )
    decode_tokens_per_s = _optional_float(
        _first_present(payload.get("decode_tokens_per_s"), throughput_payload.get("decode_tokens_per_s"))
    )
    deployment = RuntimeDeployment(
        deployment_id=deployment_id,
        role=str(payload.get("role") or "colocated"),
        member_satellites=member_satellites,
        entry_satellite_ids=_entry_satellites_for(member_satellites),
        placement_scope="aggregate_logical_pool",
        prefill_tokens_per_s=prefill_tokens_per_s,
        decode_tokens_per_s=decode_tokens_per_s,
        request_throughput_per_s=_optional_float(
            _first_present(payload.get("request_throughput_per_s"), throughput_payload.get("request_throughput_per_s"))
        ),
        per_group_prefill_tokens_per_s=prefill_tokens_per_s,
        per_group_decode_tokens_per_s=decode_tokens_per_s,
        kv_cache_capacity_tokens=_optional_int(
            _first_present(payload.get("kv_cache_capacity_tokens"), throughput_payload.get("kv_cache_capacity_tokens"))
        ),
        kv_cache_bytes_per_token=_optional_float(
            _first_present(
                payload.get("kv_cache_bytes_per_token"),
                throughput_payload.get("kv_cache_bytes_per_token"),
            )
        ),
        satellites_per_deployment=satellites_per_deployment,
        stage_satellite_count=_optional_int(
            _first_present(payload.get("stage_satellite_count"), throughput_payload.get("stage_satellite_count"))
        ),
        internal_comm_accounting=_optional_str(payload.get("internal_comm_accounting")) or "embedded",
        pd_mode=_optional_str(payload.get("pd_mode")) or _optional_str(throughput_payload.get("pd_mode")),
        placement_model="aggregate_logical_pool",
        selection_policy="fixed_default_deployment",
        provenance={
            "source": "ServingDeploymentView",
            "input_deployment_id": deployment_id,
            "member_satellite_assignment": _member_assignment_source(payload, constellation),
            **budget_payload,
            "physical_nearest_semantics": "not_available_without_access_trace",
        },
        warnings=tuple(dict.fromkeys(warnings)),
    )
    return RuntimeDeploymentCatalog(
        deployments=(deployment,),
        placement_model="aggregate_logical_pool",
        selection_policy="fixed_default_deployment",
        warnings=deployment.warnings,
        provenance={
            "source": "runtime_deployments.build_runtime_deployment_catalog",
            "constellation_satellite_count": total_satellites,
            **budget_payload,
            "phase": "1a_compute_only",
        },
    )


def _build_disaggregated_catalog(
    *,
    primary: Any,
    payload: dict[str, Any],
    throughput_payload: dict[str, Any],
    serving_payload: dict[str, Any],
    constellation: dict[str, Any],
    total_satellites: int | None,
) -> RuntimeDeploymentCatalog:
    deployment_id = str(payload.get("deployment_id") or "deployment_0")
    base_warnings = list(_string_tuple(payload.get("warnings")))
    base_warnings.extend(_string_tuple(throughput_payload.get("warnings")))
    evaluated = _optional_bool(_first_present(payload.get("evaluated"), throughput_payload.get("evaluated")))
    feasible = _optional_bool(_first_present(payload.get("feasible"), throughput_payload.get("feasible")))
    if evaluated is False:
        return _fail_closed_catalog(
            "serving_not_feasible:throughput_estimate_not_evaluated",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )
    if feasible is False:
        return _fail_closed_catalog(
            "serving_not_feasible:throughput_estimate_infeasible",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )

    satellites_per_deployment = _optional_int(
        _first_present(payload.get("satellites_per_deployment"), throughput_payload.get("satellites_per_deployment"))
    )
    stage_satellite_count = _optional_int(
        _first_present(payload.get("stage_satellite_count"), throughput_payload.get("stage_satellite_count"))
    )
    if satellites_per_deployment is None or satellites_per_deployment <= 0:
        return _fail_closed_catalog(
            "serving_not_feasible:missing_satellites_per_deployment",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )
    if stage_satellite_count is None or stage_satellite_count <= 0:
        return _fail_closed_catalog(
            "serving_not_feasible:missing_stage_satellite_count",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )

    prefill_group_count = _optional_int(
        _first_present(
            payload.get("prefill_deployment_group_count"),
            throughput_payload.get("prefill_deployment_group_count"),
        )
    )
    decode_group_count = _optional_int(
        _first_present(
            payload.get("decode_deployment_group_count"),
            throughput_payload.get("decode_deployment_group_count"),
        )
    )
    if prefill_group_count is None or decode_group_count is None:
        return _fail_closed_catalog(
            "serving_not_feasible:missing_pd_deployment_group_count",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )
    if prefill_group_count <= 0 or decode_group_count <= 0:
        return _fail_closed_catalog(
            "serving_not_feasible:invalid_pd_deployment_group_count",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )

    weight_memory_per_satellite_gb = _optional_float(
        _first_present(
            payload.get("weight_memory_per_satellite_gb"),
            throughput_payload.get("weight_memory_per_satellite_gb"),
        )
    )
    usable_memory_per_satellite_gb = _optional_float(
        _first_present(
            payload.get("usable_memory_per_satellite_gb"),
            throughput_payload.get("usable_memory_per_satellite_gb"),
        )
    )
    if (
        weight_memory_per_satellite_gb is not None
        and usable_memory_per_satellite_gb is not None
        and weight_memory_per_satellite_gb > usable_memory_per_satellite_gb
    ):
        return _fail_closed_catalog(
            "serving_not_feasible:weight_memory_exceeds_usable_memory",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )

    aggregate_prefill_tokens_per_s = _optional_float(
        _first_present(payload.get("prefill_tokens_per_s"), throughput_payload.get("prefill_tokens_per_s"))
    )
    aggregate_decode_tokens_per_s = _optional_float(
        _first_present(payload.get("decode_tokens_per_s"), throughput_payload.get("decode_tokens_per_s"))
    )
    if aggregate_prefill_tokens_per_s is None or aggregate_prefill_tokens_per_s <= 0:
        return _fail_closed_catalog(
            "serving_not_feasible:missing_prefill_throughput",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )
    if aggregate_decode_tokens_per_s is None or aggregate_decode_tokens_per_s <= 0:
        return _fail_closed_catalog(
            "serving_not_feasible:missing_decode_throughput",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )

    required_satellites = (prefill_group_count + decode_group_count) * satellites_per_deployment
    budget_payload = _serving_satellite_budget_payload(payload, throughput_payload)
    satellite_catalog = _satellite_catalog_from_constellation(constellation)
    member_satellites = _resolve_member_satellites(
        payload=payload,
        satellite_catalog=satellite_catalog,
        total_satellites=total_satellites,
        required_satellites=required_satellites,
        satellites_per_deployment=satellites_per_deployment,
    )
    if len(member_satellites) < required_satellites:
        return _fail_closed_catalog(
            "serving_not_feasible:insufficient_satellites_for_pd_deployment",
            (
                *base_warnings,
                (
                    "PD deployment groups require "
                    f"{required_satellites} satellites but only {len(member_satellites)} are available"
                ),
            ),
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )

    pd_placement = _requested_pd_placement(payload, throughput_payload, serving_payload)
    requested_scope = _normalise_placement_scope(_requested_placement_scope(payload, throughput_payload, serving_payload))
    topology_available = _topology_metadata_available(satellite_catalog, member_satellites)
    placement_scope = requested_scope or "unscoped_disaggregated"
    if placement_scope in PD_PLACEMENT_SCOPES - {"unscoped_disaggregated"} and not topology_available:
        reason = "topology_scope_unavailable:missing_cluster_or_cell_identity"
        if _fail_closed_on_unsupported_scope(pd_placement):
            return _fail_closed_catalog(
                reason,
                (*base_warnings, f"requested PD placement_scope {placement_scope} requires topology metadata"),
                total_satellites=total_satellites,
                deployment_id=deployment_id,
            )
        base_warnings.append(
            f"{reason}; requested {placement_scope} downgraded to unscoped_disaggregated"
        )
        placement_scope = "unscoped_disaggregated"
    elif placement_scope not in PD_PLACEMENT_SCOPES:
        return _fail_closed_catalog(
            "deployment_unavailable:unknown_pd_placement_scope",
            (*base_warnings, f"unknown PD placement_scope {placement_scope}"),
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )
    if placement_scope == "unscoped_disaggregated":
        base_warnings.append(
            "PD deployment catalog is unscoped because topology metadata is unavailable or not requested; "
            "scope-specific intra/same-plane/cross-plane claims are disabled"
        )

    internal_comm_accounting = _resolve_internal_comm_accounting(
        primary=primary,
        payload=payload,
        throughput_payload=throughput_payload,
        pd_mode="disaggregated",
    )
    prefill_deployment_ids = tuple(f"{deployment_id}:prefill:{index}" for index in range(prefill_group_count))
    decode_deployment_ids = tuple(f"{deployment_id}:decode:{index}" for index in range(decode_group_count))
    satellite_metadata = _satellite_metadata_by_id(satellite_catalog)
    kv_cache_capacity_tokens = _optional_int(
        _first_present(payload.get("kv_cache_capacity_tokens"), throughput_payload.get("kv_cache_capacity_tokens"))
    )
    kv_cache_bytes_per_token = _optional_float(
        _first_present(
            payload.get("kv_cache_bytes_per_token"),
            throughput_payload.get("kv_cache_bytes_per_token"),
        )
    )
    aggregate_request_throughput = _optional_float(
        _first_present(payload.get("request_throughput_per_s"), throughput_payload.get("request_throughput_per_s"))
    )
    per_group_prefill = aggregate_prefill_tokens_per_s / prefill_group_count
    per_group_decode = aggregate_decode_tokens_per_s / decode_group_count
    common_provenance = {
        "source": "ServingDeploymentView",
        "input_deployment_id": deployment_id,
        "throughput_source": "aggregate_role_throughput_split_by_pd_group_count",
        "aggregate_prefill_tokens_per_s": aggregate_prefill_tokens_per_s,
        "aggregate_decode_tokens_per_s": aggregate_decode_tokens_per_s,
        "prefill_deployment_group_count": prefill_group_count,
        "decode_deployment_group_count": decode_group_count,
        **budget_payload,
        "member_satellite_assignment": _member_assignment_source(payload, constellation),
        "placement_scope_requested": requested_scope,
        "placement_scope_effective": placement_scope,
        "valid_for_paper_main_results": False,
    }
    deployments: list[RuntimeDeployment] = [
        RuntimeDeployment(
            deployment_id=deployment_id,
            role="colocated",
            member_satellites=member_satellites,
            entry_satellite_ids=_entry_satellites_for(member_satellites),
            placement_scope="aggregate_logical_pool_compat",
            prefill_tokens_per_s=aggregate_prefill_tokens_per_s,
            decode_tokens_per_s=aggregate_decode_tokens_per_s,
            request_throughput_per_s=aggregate_request_throughput,
            kv_cache_capacity_tokens=kv_cache_capacity_tokens,
            kv_cache_bytes_per_token=kv_cache_bytes_per_token,
            satellites_per_deployment=satellites_per_deployment,
            stage_satellite_count=stage_satellite_count,
            internal_comm_accounting=internal_comm_accounting,
            pd_mode="disaggregated",
            placement_model="aggregate_logical_pool_compat",
            selection_policy="fixed_default_deployment",
            provenance={
                **common_provenance,
                "compatibility_default_deployment": True,
                "physical_pd_placement_semantics": "use prefill_deployments/decode_deployments helpers",
            },
            warnings=tuple(
                dict.fromkeys(
                    (
                        *base_warnings,
                        "aggregate compatibility deployment is retained for legacy default_deployment consumers",
                    )
                )
            ),
        )
    ]

    offset = 0
    for index, deployment_id_item in enumerate(prefill_deployment_ids):
        members = member_satellites[offset : offset + satellites_per_deployment]
        offset += satellites_per_deployment
        deployments.append(
            _pd_role_deployment(
                deployment_id=deployment_id_item,
                role="prefill",
                members=members,
                satellite_metadata=satellite_metadata,
                placement_scope=placement_scope,
                prefill_tokens_per_s=per_group_prefill,
                decode_tokens_per_s=None,
                request_throughput_per_s=None,
                per_group_prefill_tokens_per_s=per_group_prefill,
                per_group_decode_tokens_per_s=None,
                kv_cache_capacity_tokens=kv_cache_capacity_tokens,
                kv_cache_bytes_per_token=kv_cache_bytes_per_token,
                satellites_per_deployment=satellites_per_deployment,
                stage_satellite_count=stage_satellite_count,
                internal_comm_accounting=internal_comm_accounting,
                paired_deployment_ids=decode_deployment_ids,
                provenance={**common_provenance, "role_group_index": index},
                warnings=tuple(dict.fromkeys(base_warnings)),
            )
        )
    for index, deployment_id_item in enumerate(decode_deployment_ids):
        members = member_satellites[offset : offset + satellites_per_deployment]
        offset += satellites_per_deployment
        deployments.append(
            _pd_role_deployment(
                deployment_id=deployment_id_item,
                role="decode",
                members=members,
                satellite_metadata=satellite_metadata,
                placement_scope=placement_scope,
                prefill_tokens_per_s=None,
                decode_tokens_per_s=per_group_decode,
                request_throughput_per_s=None,
                per_group_prefill_tokens_per_s=None,
                per_group_decode_tokens_per_s=per_group_decode,
                kv_cache_capacity_tokens=kv_cache_capacity_tokens,
                kv_cache_bytes_per_token=kv_cache_bytes_per_token,
                satellites_per_deployment=satellites_per_deployment,
                stage_satellite_count=stage_satellite_count,
                internal_comm_accounting=internal_comm_accounting,
                paired_deployment_ids=prefill_deployment_ids,
                provenance={**common_provenance, "role_group_index": index},
                warnings=tuple(dict.fromkeys(base_warnings)),
            )
        )
    deployments, pairing_warnings = _with_scope_filtered_pairings(
        tuple(deployments),
        placement_scope=placement_scope,
        pd_placement=pd_placement,
    )
    base_warnings.extend(pairing_warnings)
    if not _has_valid_pd_pair(deployments):
        return _fail_closed_catalog(
            "deployment_unavailable:no_valid_pd_pair_for_scope",
            base_warnings,
            total_satellites=total_satellites,
            deployment_id=deployment_id,
        )
    catalog_warnings = tuple(
        dict.fromkeys((*base_warnings, *(warning for deployment in deployments for warning in deployment.warnings)))
    )
    return RuntimeDeploymentCatalog(
        deployments=tuple(deployments),
        placement_model="explicit_pd_deployment_groups",
        selection_policy="fixed_default_deployment_with_explicit_pd_groups",
        warnings=catalog_warnings,
        provenance={
            "source": "runtime_deployments.build_runtime_deployment_catalog",
            "phase": "M1_pd_deployment_catalog",
            "constellation_satellite_count": total_satellites,
            "pd_mode": "disaggregated",
            "placement_scope": placement_scope,
            "pd_placement": pd_placement,
            **budget_payload,
            "prefill_deployment_group_count": prefill_group_count,
            "decode_deployment_group_count": decode_group_count,
            "satellites_per_deployment": satellites_per_deployment,
            "unused_satellite_count": max(0, len(member_satellites) - required_satellites),
            "valid_for_paper_main_results": False,
        },
    )


def _pd_role_deployment(
    *,
    deployment_id: str,
    role: str,
    members: tuple[str, ...],
    satellite_metadata: dict[str, dict[str, Any]],
    placement_scope: str,
    prefill_tokens_per_s: float | None,
    decode_tokens_per_s: float | None,
    request_throughput_per_s: float | None,
    per_group_prefill_tokens_per_s: float | None,
    per_group_decode_tokens_per_s: float | None,
    kv_cache_capacity_tokens: int | None,
    kv_cache_bytes_per_token: float | None,
    satellites_per_deployment: int,
    stage_satellite_count: int,
    internal_comm_accounting: str,
    paired_deployment_ids: tuple[str, ...],
    provenance: dict[str, Any],
    warnings: tuple[str, ...],
) -> RuntimeDeployment:
    topology = _group_topology_fields(members, satellite_metadata)
    return RuntimeDeployment(
        deployment_id=deployment_id,
        role=role,
        member_satellites=members,
        entry_satellite_ids=_entry_satellites_for(members),
        member_energy_weights={member: 1.0 / len(members) for member in members},
        placement_scope=placement_scope,
        prefill_tokens_per_s=prefill_tokens_per_s,
        decode_tokens_per_s=decode_tokens_per_s,
        request_throughput_per_s=request_throughput_per_s,
        per_group_prefill_tokens_per_s=per_group_prefill_tokens_per_s,
        per_group_decode_tokens_per_s=per_group_decode_tokens_per_s,
        kv_cache_capacity_tokens=kv_cache_capacity_tokens,
        kv_cache_bytes_per_token=kv_cache_bytes_per_token,
        satellites_per_deployment=satellites_per_deployment,
        stage_satellite_count=stage_satellite_count,
        internal_comm_accounting=internal_comm_accounting,
        pd_mode="disaggregated",
        placement_model="explicit_pd_deployment_group",
        selection_policy="runtime_scheduler_selects_pd_pair",
        plane_index=topology["plane_index"],
        cell_index=topology["cell_index"],
        cluster_id=topology["cluster_id"],
        compute_cell_id=topology["compute_cell_id"],
        cluster_member_indices=topology["cluster_member_indices"],
        paired_deployment_ids=paired_deployment_ids,
        provenance={
            **provenance,
            "compute_energy_distribution": "explicit_uniform_member_weights",
        },
        warnings=warnings,
    )


def _with_scope_filtered_pairings(
    deployments: tuple[RuntimeDeployment, ...],
    *,
    placement_scope: str | None = None,
    pd_placement: dict[str, Any] | None = None,
) -> tuple[tuple[RuntimeDeployment, ...], tuple[str, ...]]:
    prefill_groups = tuple(deployment for deployment in deployments if deployment.role == "prefill")
    decode_groups = tuple(deployment for deployment in deployments if deployment.role == "decode")
    if not prefill_groups or not decode_groups:
        return deployments, ()
    pd_placement = dict(pd_placement or {})
    placement_scope = _normalise_placement_scope(placement_scope) or "unscoped_disaggregated"
    explicit_pairs, explicit_warnings = _explicit_pair_scope_map(
        pd_placement,
        prefill_groups=prefill_groups,
        decode_groups=decode_groups,
    )
    prefill_pairs: dict[str, tuple[str, ...]] = {}
    prefill_pair_scopes: dict[str, dict[str, str]] = {}
    prefill_pair_rules: dict[str, dict[str, str]] = {}
    warnings = list(explicit_warnings)
    template_name = _hybrid_template_name(pd_placement)
    if placement_scope == "hybrid_template_mix" and _hybrid_template_allowed_scopes(template_name) is None:
        warnings.append(f"unsupported_pd_hybrid_template:{template_name}")
    for prefill in prefill_groups:
        raw_pairs: list[tuple[str, str, str]] = []
        for decode in decode_groups:
            pair_scope = _pd_pair_scope_for(
                prefill,
                decode,
                placement_scope=placement_scope,
                pd_placement=pd_placement,
                explicit_pairs=explicit_pairs,
            )
            if pair_scope is not None:
                raw_pairs.append(
                    (
                        decode.deployment_id,
                        pair_scope,
                        _pd_pairing_rule_for(
                            placement_scope=placement_scope,
                            pd_placement=pd_placement,
                            explicit_pairs=explicit_pairs,
                        ),
                    )
                )
        limited_pairs = _limit_template_pairs(
            prefill,
            raw_pairs,
            placement_scope=placement_scope,
            pd_placement=pd_placement,
        )
        prefill_pairs[prefill.deployment_id] = tuple(pair_id for pair_id, _scope, _rule in limited_pairs)
        prefill_pair_scopes[prefill.deployment_id] = {
            pair_id: pair_scope for pair_id, pair_scope, _rule in limited_pairs
        }
        prefill_pair_rules[prefill.deployment_id] = {
            pair_id: pair_rule for pair_id, _pair_scope, pair_rule in limited_pairs
        }
    decode_pairs: dict[str, list[str]] = {decode.deployment_id: [] for decode in decode_groups}
    decode_pair_scopes: dict[str, dict[str, str]] = {decode.deployment_id: {} for decode in decode_groups}
    decode_pair_rules: dict[str, dict[str, str]] = {decode.deployment_id: {} for decode in decode_groups}
    for prefill_id, pair_ids in prefill_pairs.items():
        for decode_id in pair_ids:
            decode_pairs.setdefault(decode_id, []).append(prefill_id)
            decode_pair_scopes.setdefault(decode_id, {})[prefill_id] = prefill_pair_scopes[prefill_id][decode_id]
            decode_pair_rules.setdefault(decode_id, {})[prefill_id] = prefill_pair_rules[prefill_id][decode_id]
    filtered: list[RuntimeDeployment] = []
    for deployment in deployments:
        if deployment.role == "prefill":
            filtered.append(
                replace(
                    deployment,
                    paired_deployment_ids=prefill_pairs.get(deployment.deployment_id, ()),
                    paired_deployment_scopes=prefill_pair_scopes.get(deployment.deployment_id, {}),
                    paired_deployment_rules=prefill_pair_rules.get(deployment.deployment_id, {}),
                )
            )
        elif deployment.role == "decode":
            filtered.append(
                replace(
                    deployment,
                    paired_deployment_ids=tuple(decode_pairs.get(deployment.deployment_id, ())),
                    paired_deployment_scopes=decode_pair_scopes.get(deployment.deployment_id, {}),
                    paired_deployment_rules=decode_pair_rules.get(deployment.deployment_id, {}),
                )
            )
        else:
            filtered.append(deployment)
    return tuple(filtered), tuple(dict.fromkeys(warnings))


def _has_valid_pd_pair(deployments: tuple[RuntimeDeployment, ...] | list[RuntimeDeployment]) -> bool:
    return any(deployment.role == "prefill" and deployment.paired_deployment_ids for deployment in deployments)


def _pd_pair_scope_allowed(prefill: RuntimeDeployment, decode: RuntimeDeployment) -> bool:
    return _pd_pair_scope_for(
        prefill,
        decode,
        placement_scope=_optional_str(prefill.placement_scope) or _optional_str(decode.placement_scope),
        pd_placement={},
        explicit_pairs={},
    ) is not None


def _pd_pair_scope_for(
    prefill: RuntimeDeployment,
    decode: RuntimeDeployment,
    *,
    placement_scope: str | None,
    pd_placement: dict[str, Any],
    explicit_pairs: dict[tuple[str, str], str],
) -> str | None:
    scope = _optional_str(prefill.placement_scope) or _optional_str(decode.placement_scope)
    scope = _normalise_placement_scope(placement_scope or scope)
    if scope == "hybrid_explicit_pairs":
        explicit_scope = explicit_pairs.get((prefill.deployment_id, decode.deployment_id))
        if explicit_scope is None:
            return None
        if _scope_rule_allowed(explicit_scope, prefill, decode):
            return explicit_scope
        return None
    if scope in {"hybrid", "hybrid_template_mix"}:
        return _hybrid_template_pair_scope(prefill, decode, pd_placement)
    if scope in {None, "unscoped_disaggregated", "aggregate_logical_pool"}:
        return "unscoped_disaggregated"
    if _scope_rule_allowed(scope, prefill, decode):
        return scope
    return None


def _scope_rule_allowed(scope: str | None, prefill: RuntimeDeployment, decode: RuntimeDeployment) -> bool:
    scope = _normalise_placement_scope(scope)
    if scope in {None, "unscoped_disaggregated", "aggregate_logical_pool"}:
        return True
    if scope == "intra_cluster":
        return _same_cluster(prefill, decode)
    if scope == "same_plane_cross_cluster":
        return _same_plane(prefill, decode) and _different_cluster_or_cell(prefill, decode)
    if scope == "cross_plane":
        return prefill.plane_index is not None and decode.plane_index is not None and prefill.plane_index != decode.plane_index
    return False


def _hybrid_template_pair_scope(
    prefill: RuntimeDeployment,
    decode: RuntimeDeployment,
    pd_placement: dict[str, Any],
) -> str | None:
    inferred = _inferred_pair_scope(prefill, decode)
    if inferred is None:
        return None
    template = _hybrid_template_name(pd_placement)
    allowed = _hybrid_template_allowed_scopes(template)
    if allowed is None:
        return None
    return inferred if inferred in allowed else None


def _inferred_pair_scope(prefill: RuntimeDeployment, decode: RuntimeDeployment) -> str | None:
    if _same_cluster(prefill, decode):
        return "intra_cluster"
    if _same_plane(prefill, decode) and _different_cluster_or_cell(prefill, decode):
        return "same_plane_cross_cluster"
    if prefill.plane_index is not None and decode.plane_index is not None and prefill.plane_index != decode.plane_index:
        return "cross_plane"
    return None


def _explicit_pair_scope_map(
    pd_placement: dict[str, Any],
    *,
    prefill_groups: tuple[RuntimeDeployment, ...],
    decode_groups: tuple[RuntimeDeployment, ...],
) -> tuple[dict[tuple[str, str], str], tuple[str, ...]]:
    raw_pairs = pd_placement.get("explicit_pairs")
    if raw_pairs is None:
        raw_pairs = pd_placement.get("pairs")
    if not raw_pairs:
        return {}, ()
    prefill_by_index = {
        _role_group_index(deployment): deployment
        for deployment in prefill_groups
        if _role_group_index(deployment) is not None
    }
    decode_by_index = {
        _role_group_index(deployment): deployment
        for deployment in decode_groups
        if _role_group_index(deployment) is not None
    }
    prefill_by_id = {deployment.deployment_id: deployment for deployment in prefill_groups}
    decode_by_id = {deployment.deployment_id: deployment for deployment in decode_groups}
    mapping: dict[tuple[str, str], str] = {}
    warnings: list[str] = []
    for index, raw_pair in enumerate(raw_pairs if isinstance(raw_pairs, (list, tuple)) else ()):
        pair = _mapping(raw_pair)
        prefill = _pair_deployment_ref(
            pair,
            role="prefill",
            by_id=prefill_by_id,
            by_index=prefill_by_index,
        )
        decode = _pair_deployment_ref(
            pair,
            role="decode",
            by_id=decode_by_id,
            by_index=decode_by_index,
        )
        if prefill is None or decode is None:
            warnings.append(f"pd_pair_deployment_missing:{index}")
            continue
        resolved_scope = _inferred_pair_scope(prefill, decode)
        declared_scope = _normalise_placement_scope(
            _optional_str(pair.get("scope"))
            or _optional_str(pair.get("placement_scope"))
            or _optional_str(pair.get("declared_scope"))
        )
        if declared_scope in {"hybrid", "hybrid_explicit_pairs", "hybrid_template_mix"}:
            declared_scope = None
        if resolved_scope is None:
            warnings.append(f"pd_placement explicit pair {index} lacks an inferable concrete scope")
            continue
        if declared_scope is not None and declared_scope != resolved_scope:
            warnings.append(
                "pd_pair_scope_mismatch:"
                f"{prefill.deployment_id}->{decode.deployment_id}:declared={declared_scope}:resolved={resolved_scope}"
            )
            continue
        mapping[(prefill.deployment_id, decode.deployment_id)] = resolved_scope
    return mapping, tuple(warnings)


def _pair_deployment_ref(
    pair: dict[str, Any],
    *,
    role: str,
    by_id: dict[str, RuntimeDeployment],
    by_index: dict[int, RuntimeDeployment],
) -> RuntimeDeployment | None:
    raw_id = _first_present(
        pair.get(f"{role}_deployment_id"),
        pair.get(f"{role}_id"),
        pair.get(role),
    )
    if isinstance(raw_id, str) and raw_id in by_id:
        return by_id[raw_id]
    raw_index = _optional_int(
        _first_present(
            pair.get(f"{role}_group_index"),
            pair.get(f"{role}_index"),
            raw_id if not isinstance(raw_id, str) else None,
        )
    )
    if raw_index is not None:
        return by_index.get(raw_index)
    return None


def _role_group_index(deployment: RuntimeDeployment) -> int | None:
    return _optional_int((deployment.provenance or {}).get("role_group_index"))


def _limit_template_pairs(
    prefill: RuntimeDeployment,
    raw_pairs: list[tuple[str, str, str]],
    *,
    placement_scope: str | None,
    pd_placement: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    del prefill
    scope = _normalise_placement_scope(placement_scope)
    if scope != "hybrid_template_mix" and scope != "hybrid":
        return tuple(raw_pairs)
    template_payload = _mapping(pd_placement.get("hybrid_template"))
    max_pairs = _optional_int(
        _first_present(
            pd_placement.get("max_pairs_per_prefill"),
            pd_placement.get("max_pairs_per_prefill_group"),
            pd_placement.get("max_pair_count"),
            template_payload.get("max_pairs_per_prefill"),
            template_payload.get("max_pairs_per_prefill_group"),
            template_payload.get("max_pair_count"),
        )
    )
    if max_pairs is None:
        max_pairs = 3
    max_cross_plane = _optional_int(
        _first_present(
            pd_placement.get("max_cross_plane_pairs_per_prefill"),
            pd_placement.get("max_cross_plane_pairs_per_prefill_group"),
            template_payload.get("max_cross_plane_pairs_per_prefill"),
            template_payload.get("max_cross_plane_pairs_per_prefill_group"),
        )
    )
    if max_cross_plane is None:
        max_cross_plane = 1
    max_pairs = max(0, max_pairs)
    max_cross_plane = max(0, max_cross_plane)
    cross_pairs = [pair for pair in raw_pairs if pair[1] == "cross_plane"]
    primary_pairs = [pair for pair in raw_pairs if pair[1] != "cross_plane"]
    cross_keep = cross_pairs[:max_cross_plane]
    primary_limit = max(0, max_pairs - len(cross_keep))
    kept = [*primary_pairs[:primary_limit], *cross_keep]
    return tuple(kept[:max_pairs])


def _hybrid_template_name(pd_placement: dict[str, Any]) -> str:
    template_payload = _mapping(pd_placement.get("hybrid_template"))
    return str(
        _first_present(
            template_payload.get("template_id"),
            template_payload.get("id"),
            template_payload.get("name"),
            pd_placement.get("hybrid_template"),
            pd_placement.get("template"),
            pd_placement.get("hybrid_template_name"),
            "same_plane_plus_cross_plane_overflow",
        )
    )


def _hybrid_template_allowed_scopes(template: str) -> frozenset[str] | None:
    aliases = {
        "same_plane_primary_cross_plane_overflow": "same_plane_plus_cross_plane_overflow",
        "cluster_hotspot_spillover": "same_plane_plus_cross_plane_overflow",
        "local_prefill_remote_decode": "same_plane_plus_cross_plane_overflow",
        "local_decode_remote_prefill": "same_plane_plus_cross_plane_overflow",
        "energy_shadow_cross_plane_escape": "intra_cluster_plus_cross_plane_escape",
    }
    template = aliases.get(str(template), str(template))
    if template == "intra_cluster_plus_cross_plane_escape":
        return frozenset({"intra_cluster", "cross_plane"})
    if template == "all_supported":
        return frozenset({"intra_cluster", "same_plane_cross_cluster", "cross_plane"})
    if template == "same_plane_plus_cross_plane_overflow":
        return frozenset({"same_plane_cross_cluster", "cross_plane"})
    return None


def _pd_pairing_rule_for(
    *,
    placement_scope: str | None,
    pd_placement: dict[str, Any],
    explicit_pairs: dict[tuple[str, str], str],
) -> str:
    del explicit_pairs
    scope = _normalise_placement_scope(placement_scope)
    if scope == "hybrid_explicit_pairs":
        return "user_explicit"
    if scope == "hybrid_template_mix":
        return f"hybrid_template:{_hybrid_template_name(pd_placement)}"
    return scope or "unscoped_disaggregated"


def _same_cluster(prefill: RuntimeDeployment, decode: RuntimeDeployment) -> bool:
    if prefill.cluster_id is not None and decode.cluster_id is not None:
        return prefill.cluster_id == decode.cluster_id
    if prefill.compute_cell_id is not None and decode.compute_cell_id is not None:
        return prefill.compute_cell_id == decode.compute_cell_id
    return _same_plane(prefill, decode) and prefill.cell_index is not None and prefill.cell_index == decode.cell_index


def _same_plane(prefill: RuntimeDeployment, decode: RuntimeDeployment) -> bool:
    return prefill.plane_index is not None and decode.plane_index is not None and prefill.plane_index == decode.plane_index


def _different_cluster_or_cell(prefill: RuntimeDeployment, decode: RuntimeDeployment) -> bool:
    if prefill.cluster_id is not None and decode.cluster_id is not None:
        return prefill.cluster_id != decode.cluster_id
    if prefill.compute_cell_id is not None and decode.compute_cell_id is not None:
        return prefill.compute_cell_id != decode.compute_cell_id
    return prefill.cell_index is not None and decode.cell_index is not None and prefill.cell_index != decode.cell_index


def _fail_closed_catalog(
    reason: str,
    warnings: tuple[str, ...] | list[str],
    *,
    total_satellites: int | None,
    deployment_id: str,
) -> RuntimeDeploymentCatalog:
    catalog_warnings = tuple(dict.fromkeys((*tuple(warnings), reason)))
    return RuntimeDeploymentCatalog(
        deployments=(),
        placement_model="explicit_pd_deployment_groups_unavailable",
        selection_policy="no_runtime_deployment",
        warnings=catalog_warnings,
        provenance={
            "source": "runtime_deployments.build_runtime_deployment_catalog",
            "phase": "M1_pd_deployment_catalog",
            "input_deployment_id": deployment_id,
            "constellation_satellite_count": total_satellites,
            "fail_closed": True,
            "reason": reason,
            "valid_for_paper_main_results": False,
        },
    )


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return to_jsonable(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = to_jsonable(value.to_dict())
        return converted if isinstance(converted, dict) else {}
    converted = to_jsonable(value)
    return converted if isinstance(converted, dict) else {}


def _serving_satellite_budget_payload(
    payload: dict[str, Any],
    throughput_payload: dict[str, Any],
) -> dict[str, Any]:
    pd_provenance = _mapping(_mapping(throughput_payload.get("provenance")).get("pd"))
    keys = (
        "serving_satellite_budget_count",
        "constellation_satellite_count",
        "satellite_budget_mode",
        "satellite_budget_source",
        "comparison_group_id",
        "satellite_budget_unused_count",
        "role_budget_unused_count",
        "pd_role_budget_source",
        "prefill_pool_satellites",
        "decode_pool_satellites",
    )
    out: dict[str, Any] = {}
    for key in keys:
        value = _first_present(payload.get(key), throughput_payload.get(key), pd_provenance.get(key))
        if value is not None:
            out[key] = value
    return out


def _serving_satellite_budget_count(
    payload: dict[str, Any],
    throughput_payload: dict[str, Any],
) -> int | None:
    budget_payload = _serving_satellite_budget_payload(payload, throughput_payload)
    count = _optional_int(budget_payload.get("serving_satellite_budget_count"))
    if count is None or count <= 0:
        return None
    return count


def _constellation_satellite_count(constellation: dict[str, Any]) -> int | None:
    total_satellites = _optional_int(constellation.get("satellite_count"))
    if total_satellites is not None:
        return total_satellites
    return _optional_int(_mapping(constellation.get("design")).get("satellite_count"))


def _explicit_member_satellites(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in payload.get("member_satellites", ()) or ())


def _resolve_member_satellites(
    *,
    payload: dict[str, Any],
    satellite_catalog: tuple[dict[str, Any], ...],
    total_satellites: int | None,
    required_satellites: int,
    satellites_per_deployment: int = 1,
) -> tuple[str, ...]:
    explicit = _explicit_member_satellites(payload)
    if explicit:
        return explicit
    catalog_ids = tuple(str(item.get("satellite_id")) for item in satellite_catalog if item.get("satellite_id"))
    if catalog_ids:
        placement = _mapping(_mapping(payload.get("serving_spec")).get("pd_placement"))
        selection = _optional_str(
            _first_present(
                placement.get("member_selection"),
                placement.get("member_assignment"),
                placement.get("member_selection_policy"),
            )
        )
        if selection in {"strided_catalog_groups", "stride_catalog_groups", "even_catalog_groups"}:
            return _strided_catalog_groups(
                catalog_ids,
                required_satellites=required_satellites,
                satellites_per_deployment=satellites_per_deployment,
            )
        if selection in {"paired_strided_catalog_groups", "paired_stride_catalog_groups", "local_paired_strided_groups"}:
            return _paired_strided_catalog_groups(
                catalog_ids,
                payload=payload,
                required_satellites=required_satellites,
                satellites_per_deployment=satellites_per_deployment,
            )
        return catalog_ids
    if total_satellites is not None:
        return _synthetic_satellite_ids(max(0, total_satellites))
    return _synthetic_satellite_ids(required_satellites)


def _strided_catalog_groups(
    catalog_ids: tuple[str, ...],
    *,
    required_satellites: int,
    satellites_per_deployment: int,
) -> tuple[str, ...]:
    group_size = max(1, int(satellites_per_deployment))
    required_groups = max(0, math.ceil(required_satellites / group_size))
    groups = tuple(
        catalog_ids[index : index + group_size]
        for index in range(0, len(catalog_ids), group_size)
        if len(catalog_ids[index : index + group_size]) == group_size
    )
    if required_groups <= 0:
        return ()
    if required_groups >= len(groups):
        return tuple(item for group in groups for item in group)
    selected: list[tuple[str, ...]] = []
    seen: set[int] = set()
    stride = len(groups) / required_groups
    for item_index in range(required_groups):
        group_index = min(len(groups) - 1, int(item_index * stride))
        while group_index in seen and group_index + 1 < len(groups):
            group_index += 1
        while group_index in seen and group_index > 0:
            group_index -= 1
        seen.add(group_index)
        selected.append(groups[group_index])
    return tuple(item for group in selected for item in group)


def _paired_strided_catalog_groups(
    catalog_ids: tuple[str, ...],
    *,
    payload: dict[str, Any],
    required_satellites: int,
    satellites_per_deployment: int,
) -> tuple[str, ...]:
    group_size = max(1, int(satellites_per_deployment))
    groups = tuple(
        catalog_ids[index : index + group_size]
        for index in range(0, len(catalog_ids), group_size)
        if len(catalog_ids[index : index + group_size]) == group_size
    )
    if not groups:
        return ()
    total_required_groups = max(0, math.ceil(required_satellites / group_size))
    throughput = _mapping(payload.get("throughput_estimate"))
    pd_placement = _mapping(_mapping(payload.get("serving_spec")).get("pd_placement"))
    prefill_count = _optional_int(throughput.get("prefill_deployment_group_count"))
    decode_count = _optional_int(throughput.get("decode_deployment_group_count"))
    if prefill_count is None or decode_count is None:
        prefill_count = total_required_groups // 2
        decode_count = total_required_groups - prefill_count
    prefill_count = max(0, min(int(prefill_count), len(groups)))
    decode_count = max(0, min(int(decode_count), len(groups)))
    pair_count = min(prefill_count, decode_count)
    if pair_count <= 0:
        return _strided_catalog_groups(
            catalog_ids,
            required_satellites=required_satellites,
            satellites_per_deployment=satellites_per_deployment,
        )
    decode_offset = _optional_int(
        _first_present(
            pd_placement.get("paired_decode_group_offset"),
            pd_placement.get("decode_group_offset"),
            pd_placement.get("local_decode_group_offset"),
        )
    )
    if decode_offset is None:
        decode_offset = max(1, int(pd_placement.get("groups_per_cluster", 8) or 8))
    selected_prefill: list[int] = []
    selected_decode: list[int] = []
    used: set[int] = set()
    stride = len(groups) / pair_count
    for index in range(pair_count):
        prefill_index = _nearest_unused_group_index(int(index * stride), used, len(groups))
        used.add(prefill_index)
        selected_prefill.append(prefill_index)
        decode_index = _nearest_unused_group_index(prefill_index + decode_offset, used, len(groups))
        used.add(decode_index)
        selected_decode.append(decode_index)
    while len(selected_prefill) < prefill_count:
        index = _nearest_unused_group_index(int(len(selected_prefill) * len(groups) / prefill_count), used, len(groups))
        used.add(index)
        selected_prefill.append(index)
    while len(selected_decode) < decode_count:
        index = _nearest_unused_group_index(int(len(selected_decode) * len(groups) / decode_count) + decode_offset, used, len(groups))
        used.add(index)
        selected_decode.append(index)
    selected = (*selected_prefill[:prefill_count], *selected_decode[:decode_count])
    return tuple(item for group_index in selected for item in groups[group_index])[:required_satellites]


def _nearest_unused_group_index(start: int, used: set[int], group_count: int) -> int:
    if group_count <= 0:
        return 0
    start = start % group_count
    if start not in used:
        return start
    for distance in range(1, group_count):
        forward = (start + distance) % group_count
        if forward not in used:
            return forward
        backward = (start - distance) % group_count
        if backward not in used:
            return backward
    return start


def _synthetic_satellite_ids(count: int) -> tuple[str, ...]:
    return tuple(f"sat-{index:05d}" for index in range(max(0, count)))


def _entry_satellites_for(member_satellites: tuple[str, ...]) -> tuple[str, ...]:
    return member_satellites[:1]


def _member_assignment_source(payload: dict[str, Any], constellation: dict[str, Any]) -> str:
    if _explicit_member_satellites(payload):
        return "payload.member_satellites"
    if _satellite_catalog_from_constellation(constellation):
        return "constellation.satellite_catalog"
    return "deterministic synthetic sat-xxxxx ids"


def _satellite_catalog_from_constellation(constellation: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    sources = (
        constellation.get("satellite_catalog"),
        _mapping(constellation.get("design")).get("satellite_catalog"),
        _mapping(constellation.get("architecture")).get("satellite_catalog"),
        _mapping(_mapping(constellation.get("design")).get("architecture_analysis")).get("satellite_catalog"),
    )
    for source in sources:
        if isinstance(source, list):
            return tuple(_mapping(item) for item in source if isinstance(item, dict))
        if isinstance(source, tuple):
            return tuple(_mapping(item) for item in source if isinstance(item, dict))
    return ()


def _satellite_metadata_by_id(satellite_catalog: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("satellite_id")): item
        for item in satellite_catalog
        if item.get("satellite_id") is not None
    }


def _topology_metadata_available(
    satellite_catalog: tuple[dict[str, Any], ...],
    member_satellites: tuple[str, ...],
) -> bool:
    if not satellite_catalog:
        return False
    metadata = _satellite_metadata_by_id(satellite_catalog)
    for satellite_id in member_satellites:
        item = metadata.get(satellite_id)
        if not item:
            return False
        if item.get("cluster_id") or item.get("compute_cell_id"):
            continue
        if (
            item.get("orbital_set_name") is not None
            and item.get("plane_index") is not None
            and item.get("cell_index") is not None
        ):
            continue
        return False
    return True


def _requested_placement_scope(
    payload: dict[str, Any],
    throughput_payload: dict[str, Any],
    serving_payload: dict[str, Any],
) -> str | None:
    pd_placement = _requested_pd_placement(payload, throughput_payload, serving_payload)
    return (
        _optional_str(payload.get("pd_placement_scope"))
        or _optional_str(throughput_payload.get("pd_placement_scope"))
        or _optional_str(serving_payload.get("pd_placement_scope"))
        or _optional_str(pd_placement.get("placement_scope"))
    )


def _requested_pd_placement(
    payload: dict[str, Any],
    throughput_payload: dict[str, Any],
    serving_payload: dict[str, Any],
) -> dict[str, Any]:
    return _mapping(
        _first_present(
            payload.get("pd_placement"),
            throughput_payload.get("pd_placement"),
            serving_payload.get("pd_placement"),
        )
    )


def _normalise_placement_scope(scope: Any) -> str | None:
    text = _optional_str(scope)
    if text is None:
        return None
    aliases = {
        "same_plane": "same_plane_cross_cluster",
        "hybrid": "hybrid_template_mix",
        "explicit_pairs": "hybrid_explicit_pairs",
        "template_mix": "hybrid_template_mix",
    }
    return aliases.get(text, text)


def _fail_closed_on_unsupported_scope(pd_placement: dict[str, Any]) -> bool:
    value = pd_placement.get("fail_closed_on_unsupported_scope")
    if value is None:
        return True
    return bool(value)


def _group_topology_fields(
    members: tuple[str, ...],
    satellite_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = tuple(satellite_metadata.get(member, {}) for member in members)
    cluster_member_indices = tuple(
        index
        for index in (_optional_int(item.get("cluster_member_index")) for item in metadata)
        if index is not None
    )
    return {
        "plane_index": _shared_optional_int(item.get("plane_index") for item in metadata),
        "cell_index": _shared_optional_int(item.get("cell_index") for item in metadata),
        "cluster_id": _shared_optional_str(item.get("cluster_id") for item in metadata),
        "compute_cell_id": _shared_optional_str(item.get("compute_cell_id") for item in metadata),
        "cluster_member_indices": cluster_member_indices,
    }


def _resolve_internal_comm_accounting(
    *,
    primary: Any,
    payload: dict[str, Any],
    throughput_payload: dict[str, Any],
    pd_mode: str,
) -> str:
    if isinstance(primary, dict) and "internal_comm_accounting" in primary:
        return _normalise_internal_comm_accounting(primary.get("internal_comm_accounting"))
    if "internal_comm_accounting" in throughput_payload:
        return _normalise_internal_comm_accounting(throughput_payload.get("internal_comm_accounting"))
    raw_attr = getattr(primary, "internal_comm_accounting", None)
    if raw_attr and str(raw_attr) != "embedded":
        return _normalise_internal_comm_accounting(raw_attr)
    if pd_mode == "disaggregated":
        return "externalized_pd_kv"
    return _normalise_internal_comm_accounting(payload.get("internal_comm_accounting"))


def _normalise_internal_comm_accounting(value: Any) -> str:
    text = _optional_str(value)
    if text == "externalized":
        return "externalized_pd_kv"
    return text or "embedded"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _shared_optional_int(values: Any) -> int | None:
    present = tuple(value for value in (_optional_int(item) for item in values) if value is not None)
    return present[0] if present and all(item == present[0] for item in present) else None


def _shared_optional_str(values: Any) -> str | None:
    present = tuple(value for value in (_optional_str(item) for item in values) if value is not None)
    return present[0] if present and all(item == present[0] for item in present) else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


__all__ = [
    "RuntimeDeployment",
    "RuntimeDeploymentCatalog",
    "build_runtime_deployment_catalog",
]
