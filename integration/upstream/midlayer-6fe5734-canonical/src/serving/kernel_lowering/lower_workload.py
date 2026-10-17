"""Explicit WorkloadNode-to-KernelSpec lowering rules."""

import dataclasses
from dataclasses import fields
import pickle
from typing import Callable, Mapping

from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.kernel.fused_moe_physical_specs import (
    build_fused_moe_physical_specs,
)
from src.kernel.predictors.fused_moe import (
    ROUTING_ORDER_SENSITIVE_RECIPE_VARIANTS,
    canonical_expert_axis,
)
from src.kernel.launch_config.block_fp8_gemm import (
    BLOCK_FP8_WEIGHT_QUANTIZATION,
    PER_TOKEN_GROUP_QUANT_GROUP_SIZE,
)
from src.kernel.launch_config.select_launch_config import (
    normalize_flash_attn_execution_attributes,
    select_fused_moe_recipe,
    select_flash_attn_copy_launches,
    select_flash_attn_forward_partition,
    select_flash_attn_output_fill_launches,
    select_flash_attn_setup_launches,
    select_block_fp8_input_quant_launch,
    select_fused_moe_input_quant_launch,
    select_gemm_reducer_launch,
    select_launch_config,
)
from src.serving.deployment.deployment_config import (
    worker_role_operator_group,
    worker_role_phase,
)
from src.serving.deployment.deployment_plan import (
    DeploymentPlan,
    pd_kv_transfer_rank_pairs,
)
from src.serving.workload.build_workload import _collective_ring_order
from src.serving.workload.helper_launch import (
    RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED,
)
from src.serving.workload.runtime_inputs import (
    validate_expert_routing_provenance,
)
from src.serving.workload.workload_graph import WorkloadGraph, WorkloadKind, WorkloadNode

from .kernel_plan import (
    KernelBlockRepeat, KernelCall, KernelPlan, require_hardware)


_SINGLE_COMPUTE_KIND = {
    WorkloadKind.LINEAR: KernelKind.GEMM,
    WorkloadKind.LM_HEAD: KernelKind.GEMM,
    WorkloadKind.ROUTER: KernelKind.GEMM,
    WorkloadKind.FLASH_ATTN: KernelKind.FLASH_ATTN,
    WorkloadKind.NORM: KernelKind.VECTOR,
    WorkloadKind.ACTIVATION: KernelKind.VECTOR,
    WorkloadKind.ROTARY: KernelKind.VECTOR,
    WorkloadKind.ELEMENTWISE: KernelKind.VECTOR,
    WorkloadKind.REDUCE: KernelKind.VECTOR,
    WorkloadKind.GATHER: KernelKind.VECTOR,
    WorkloadKind.VECTOR: KernelKind.VECTOR,
    WorkloadKind.KV_UPDATE: KernelKind.KV_CACHE,
}

_COMMUNICATION_KINDS = {
    WorkloadKind.COLLECTIVE,
    WorkloadKind.P2P,
    WorkloadKind.PD_KV_TRANSFER,
    WorkloadKind.AF_ACTIVATION_TRANSFER,
}

_GRAPH_TOKEN_DOMAIN_FIELDS = (
    "graph_logical_tokens",
    "graph_model_input_tokens",
    "graph_runtime_mode",
)


def _graph_token_domain(node: WorkloadNode) -> tuple[int, int, str] | None:
    """Validate an operator's effective-work/captured-tensor row contract."""

    if node.kind is WorkloadKind.FLASH_ATTN:
        # FlashAttention has its more precise per-request contract below.
        return None
    present = tuple(
        name for name in _GRAPH_TOKEN_DOMAIN_FIELDS
        if name in node.attributes)
    if not present:
        return None
    if present != _GRAPH_TOKEN_DOMAIN_FIELDS:
        raise ValueError(
            f"workload node {node.node_id!r} has incomplete graph token "
            f"domains: present={present!r}")
    logical_tokens = node.attributes["graph_logical_tokens"]
    model_input_tokens = node.attributes["graph_model_input_tokens"]
    runtime_mode = node.attributes["graph_runtime_mode"]
    for name, value in (
            ("graph_logical_tokens", logical_tokens),
            ("graph_model_input_tokens", model_input_tokens)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if model_input_tokens < logical_tokens:
        raise ValueError("graph model-input tokens cannot shrink work")
    if runtime_mode not in ("NONE", "FULL", "PIECEWISE"):
        raise ValueError(f"unknown graph runtime mode: {runtime_mode!r}")
    if runtime_mode == "NONE" and logical_tokens != model_input_tokens:
        raise ValueError("eager node cannot declare graph padding")
    return logical_tokens, model_input_tokens, runtime_mode


def _logical_prediction_shape(
    node: WorkloadNode,
    physical_shape: Mapping[str, int],
) -> tuple[dict[str, int], tuple[int, int, str] | None]:
    """Keep semantic work logical while preserving physical launch input."""

    domain = _graph_token_domain(node)
    if domain is None:
        return dict(physical_shape), None
    logical_tokens, model_input_tokens, _ = domain
    result = dict(physical_shape)
    token_dimensions = {
        KernelKind.GEMM: ("m",),
        KernelKind.VECTOR: ("rows", "seq_len", "tok", "element_count"),
        KernelKind.KV_CACHE: ("seq_len", "element_count"),
    }
    kind = _SINGLE_COMPUTE_KIND[node.kind]
    if kind not in token_dimensions:
        raise ValueError(
            f"graph token domains are unsupported for {kind.value}")
    changed = False
    for name in token_dimensions[kind]:
        if name not in result:
            continue
        value = result[name]
        if value % model_input_tokens:
            raise ValueError(
                f"{node.node_id!r} physical {name}={value} is not an "
                "integral graph-token multiple")
        result[name] = value // model_input_tokens * logical_tokens
        changed = True
    if not changed:
        raise ValueError(
            f"{node.node_id!r} graph token domain has no token-shaped "
            f"dimension in {sorted(result)!r}")
    return result, domain


def _kernel_graph_token_attributes(
    domain: tuple[int, int, str] | None,
    physical_shape: Mapping[str, int],
) -> dict[str, object]:
    if domain is None:
        return {}
    logical_tokens, model_input_tokens, runtime_mode = domain
    return {
        "graph_logical_tokens": logical_tokens,
        "graph_model_input_tokens": model_input_tokens,
        "graph_runtime_mode": runtime_mode,
        "model_input_shape": dict(physical_shape),
    }


def _derived_graph_token_attributes(
    parent: KernelSpec,
    physical_shape: Mapping[str, int],
) -> dict[str, object]:
    """Carry a parent's graph domain into a generated physical helper.

    Lowering expands one graph-attributed operation into source-defined
    metadata, copy, and reduction kernels.  Those helpers execute on a
    physical captured buffer, so their own shape remains physical; retaining
    the parent domain prevents the helper from silently becoming a
    single-token-domain KernelSpec.
    """

    if parent.kind is KernelKind.FLASH_ATTN:
        names = (
            "model_input_q_lens", "model_input_kv_lens",
            "graph_runtime_mode",
        )
        present = tuple(name for name in names if name in parent.attributes)
        if not present:
            return {}
        if present != names:
            raise ValueError(
                "FLASH_ATTN helper graph token domains are incomplete: "
                f"present={present!r}")
        if "q_lens" not in parent.attributes:
            raise ValueError("FLASH_ATTN helper has no logical request lengths")
        logical_tokens = sum(parent.attributes["q_lens"])
        model_input_tokens = sum(parent.attributes["model_input_q_lens"])
        runtime_mode = parent.attributes["graph_runtime_mode"]
        if model_input_tokens < logical_tokens:
            raise ValueError(
                "FLASH_ATTN helper model-input tokens cannot shrink work")
        return {
            "graph_logical_tokens": logical_tokens,
            "graph_model_input_tokens": model_input_tokens,
            "graph_runtime_mode": runtime_mode,
            "model_input_shape": dict(physical_shape),
        }
    names = (
        "graph_logical_tokens", "graph_model_input_tokens",
        "graph_runtime_mode", "model_input_shape",
    )
    present = tuple(name for name in names if name in parent.attributes)
    if not present:
        return {}
    if present != names:
        raise ValueError(
            "generated helper graph token domains are incomplete: "
            f"present={present!r}")
    return {
        "graph_logical_tokens": parent.attributes["graph_logical_tokens"],
        "graph_model_input_tokens": parent.attributes[
            "graph_model_input_tokens"],
        "graph_runtime_mode": parent.attributes["graph_runtime_mode"],
        "model_input_shape": dict(physical_shape),
    }


def _validate_graph_domain_boundary(
    node: WorkloadNode,
    graph: WorkloadGraph,
) -> None:
    """Require an activation-transfer boundary to preserve its token domain."""

    domain = _graph_token_domain(node)
    predecessors = graph.predecessors(node.node_id)
    if not predecessors:
        return
    predecessor_domains = tuple(
        _graph_token_domain(graph.node(predecessor))
        for predecessor in predecessors)
    if domain is None:
        if any(predecessor_domain is not None
               for predecessor_domain in predecessor_domains):
            raise ValueError(
                "graph activation boundary omitted its producer token domain")
        return
    if any(predecessor_domain != domain
           for predecessor_domain in predecessor_domains):
        raise ValueError(
            "graph activation boundary token domain disagrees with its "
            "producer")


def _logical_node_contract(node: WorkloadNode, routing_relabelling=None):
    """Freeze every KernelSpec source field, excluding placement identity."""

    attributes = dict(node.attributes)
    if node.kind is WorkloadKind.FUSED_MOE:
        attributes.pop("routing_provenance")
        attributes.pop("moe_invocation_id")
        if routing_relabelling is not None:
            global_counts, local_counts = routing_relabelling
            attributes["tokens_per_expert"] = global_counts
            attributes["local_tokens_per_expert"] = local_counts
    elif (node.kind is WorkloadKind.COLLECTIVE
          and "moe_invocation_id" in attributes):
        attributes.pop("moe_invocation_id")
    contract_fields = []
    for field in fields(node):
        if field.name in ("node_id", "rank"):
            continue
        value = getattr(node, field.name)
        if field.name == "shape":
            value = dict(value)
        if field.name == "attributes":
            value = attributes
        contract_fields.append((field.name, value))
    return (
        "pickle_protocol_5",
        pickle.dumps(
            (
                type(node).__module__,
                type(node).__qualname__,
                tuple(contract_fields),
            ),
            protocol=5,
        ),
    )


def _memo_contracts(graph: WorkloadGraph):
    """Freeze every node contract that can affect emitted KernelSpecs."""

    return {
        node.node_id: _logical_node_contract(node)
        for node in graph.nodes
        if node.kind is not WorkloadKind.FUSED_MOE
    }


def _fused_moe_routing_relabelling(node, hardware_id, hardware):
    """Return this MoE node's canonical routing, or ``None`` to keep its own.

    The grouped projections read the expert axis only through the physical
    class ``canonical_expert_axis`` names, so every node in one class lowers
    to the same KernelSpec tuple and the run-local memo may share it.  A
    recipe that also emits a kernel reading the axis as an ordered prefix
    stream keeps its exact histogram instead.
    """

    if "owned_experts" not in node.attributes:
        return None
    recipe = select_fused_moe_recipe(hardware_id, node.shape, hardware)
    if ROUTING_ORDER_SENSITIVE_RECIPE_VARIANTS.intersection(recipe):
        return None
    return canonical_expert_axis(
        node.attributes["tokens_per_expert"],
        node.attributes["owned_experts"],
    )


class KernelLoweringMemo:
    """Reuse exact KernelSpec templates only within one simulation run."""

    def __init__(self):
        self._specs = {}
        self._hardware_objects = {}
        self._hit_count = 0
        self._miss_count = 0

    @property
    def entry_count(self) -> int:
        return len(self._specs)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def lower(
        self,
        node: WorkloadNode,
        logical_node_contract,
        hardware,
        build_specs: Callable[[], tuple[KernelSpec, ...]],
    ) -> tuple[KernelSpec, ...]:
        if not isinstance(node, WorkloadNode):
            raise TypeError("kernel lowering memo requires WorkloadNode")
        hardware_identity = id(hardware)
        if hardware_identity in self._hardware_objects:
            if self._hardware_objects[hardware_identity] is not hardware:
                raise RuntimeError(
                    "kernel lowering memo hardware identity collision")
        else:
            self._hardware_objects[hardware_identity] = hardware
        if not isinstance(logical_node_contract, tuple):
            raise TypeError(
                "kernel lowering memo requires a frozen logical node "
                "contract")
        try:
            hash(logical_node_contract)
        except TypeError as exc:
            raise TypeError(
                "kernel lowering memo requires a hashable logical node "
                "contract") from exc
        # Every lowering builder is a pure function of the node contract and
        # HWSpec.  Rank placement and graph dependencies remain on KernelCall
        # and therefore cannot change the emitted KernelSpec tuple.
        key = (logical_node_contract, hardware_identity)
        if key in self._specs:
            self._hit_count += 1
            return self._specs[key]
        specs = build_specs()
        if not isinstance(specs, tuple):
            raise TypeError("kernel lowering memo requires a KernelSpec tuple")
        if any(not isinstance(spec, KernelSpec) for spec in specs):
            raise TypeError(
                "kernel lowering memo entries must contain only KernelSpec")
        self._specs[key] = specs
        self._miss_count += 1
        return specs


def _target_model_semantics_contract(
    graph: WorkloadGraph,
) -> str | None:
    contracts = tuple(
        node.attributes["model_semantics_contract"]
        for node in graph.nodes
        if "model_semantics_contract" in node.attributes
    )
    if not contracts:
        return None
    if len(contracts) != 1:
        raise ValueError(
            "target-model workload must declare exactly one semantics "
            "contract")
    contract = contracts[0]
    if contract not in (
            "kimi_k2_6_vllm_0_27_1_logical_v1",
            "glm_5_2_vllm_0_27_1_logical_v1",
            "deepseek_v3_vllm_0_27_1_logical_v1"):
        raise ValueError(
            f"unknown target-model semantics contract: {contract!r}")
    return contract


def _validate_routing_provenance(node: WorkloadNode) -> Mapping[str, object]:
    if "routing_provenance" not in node.attributes:
        raise ValueError(
            f"FUSED_MOE node {node.node_id!r} lacks routing_provenance")
    if "tokens_per_expert" not in node.attributes:
        raise ValueError(
            f"FUSED_MOE node {node.node_id!r} lacks tokens_per_expert")
    if "moe_invocation_id" not in node.attributes:
        raise ValueError(
            f"FUSED_MOE node {node.node_id!r} lacks moe_invocation_id")
    routing_tokens = node.shape["tokens"]
    domain = _graph_token_domain(node)
    if domain is not None:
        logical_tokens, model_input_tokens, _ = domain
        if routing_tokens != model_input_tokens:
            raise ValueError(
                "FUSED_MOE model-input tokens disagree with its graph "
                "token-domain contract")
        routing_tokens = logical_tokens
    return validate_expert_routing_provenance(
        node.attributes["routing_provenance"],
        tokens_per_expert=tuple(node.attributes["tokens_per_expert"]),
        tokens=routing_tokens,
        num_experts=node.shape["experts"],
        topk=node.shape["topk"],
        moe_invocation_id=node.attributes["moe_invocation_id"],
    )


def _validate_ep_fused_moe_graph(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
) -> None:
    """Validate the complete Serving EP contract before kernel lowering."""

    ep_nodes = []
    for node in graph.nodes:
        if node.kind is not WorkloadKind.FUSED_MOE:
            continue
        if "expert_parallel_size" not in node.attributes:
            raise ValueError(
                f"FUSED_MOE node {node.node_id!r} lacks "
                "expert_parallel_size")
        expert_parallel_size = node.attributes["expert_parallel_size"]
        if (isinstance(expert_parallel_size, bool)
                or not isinstance(expert_parallel_size, int)
                or expert_parallel_size < 1):
            raise TypeError(
                "FUSED_MOE expert_parallel_size must be an integer >= 1")
        if node.pool_name not in deployment.worker_pools:
            raise ValueError(
                f"FUSED_MOE node references unknown pool {node.pool_name!r}")
        expected_expert_parallel_size = (
            deployment.worker_pools[
                node.pool_name].parallel_plan.config.expert_parallel_size)
        if expert_parallel_size != expected_expert_parallel_size:
            raise ValueError(
                "FUSED_MOE expert_parallel_size disagrees with the "
                f"deployment plan: {expert_parallel_size} != "
                f"{expected_expert_parallel_size}")
        if expected_expert_parallel_size > 1:
            ep_nodes.append(node)
    ep_pools = tuple(
        pool
        for pool in deployment.worker_pools.values()
        if pool.parallel_plan.config.enable_expert_parallel
    )
    if not ep_pools:
        if ep_nodes:
            raise ValueError(
                "EP FUSED_MOE nodes require an expert-parallel worker pool")
        return
    if not ep_nodes:
        if all(
                node.kind is WorkloadKind.PD_KV_TRANSFER
                for node in graph.nodes):
            return
        raise ValueError(
            "expert-parallel deployment does not contain EP FUSED_MOE nodes")
    ep_pool_names = {node.pool_name for node in ep_nodes}
    if len(ep_pool_names) != 1:
        raise NotImplementedError(
            "one workload graph may bind only one expert-parallel worker pool")
    ep_pool_name = next(iter(ep_pool_names))
    ep_pool = deployment.worker_pools[ep_pool_name]
    parallelism = ep_pool.parallel_plan.config
    common_ep2 = (
        len(ep_pool.ranks) == 2
        and parallelism.worker_count == 2
        and parallelism.tensor_parallel_size == 2
        and parallelism.expert_parallel_size == 2
        and parallelism.pipeline_parallel_size == 1
        and parallelism.sequence_parallel_size == 1
        and parallelism.context_parallel_size == 1
        and parallelism.data_parallel_size == 1
    )
    independently_scheduled_ep2 = deployment.mode in (
        "colocated", "pd_disagg")
    attention_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "attention"
        and worker_role_phase(pool.role) == "colocated"
    )
    af_ep2 = (
        deployment.mode == "af_disagg"
        and len(deployment.worker_pools) == 2
        and worker_role_operator_group(ep_pool.role) == "ffn"
        and worker_role_phase(ep_pool.role) == "colocated"
        and len(attention_pools) == 1
        and attention_pools[0].hardware_id == ep_pool.hardware_id
        and len(attention_pools[0].ranks) == 2
        and attention_pools[0].parallel_plan.config.worker_count == 2
        and attention_pools[0].parallel_plan.config.tensor_parallel_size == 2
        and attention_pools[0].parallel_plan.config.expert_parallel_size == 1
        and attention_pools[0].parallel_plan.config.pipeline_parallel_size == 1
        and attention_pools[0].parallel_plan.config.sequence_parallel_size == 1
        and attention_pools[0].parallel_plan.config.context_parallel_size == 1
        and attention_pools[0].parallel_plan.config.data_parallel_size == 1
    )
    if (not common_ep2 or not (independently_scheduled_ep2 or af_ep2)):
        raise NotImplementedError(
            "rank-local EP lowering supports independently scheduled "
            "colocated/PD or standalone AF TP=EP=2 pools with "
            "DP=PP=SP=CP=1")
    tuple(
        ep_pool.collective_for(operation)
        for operation in ("all_reduce", "all_gather")
    )
    required_attributes = {
        "tokens_per_expert",
        "local_tokens_per_expert",
        "expert_parallel_size",
        "expert_rank",
        "owned_experts",
        "moe_invocation_id",
        "expert_placement_strategy",
        "routing_provenance",
        "launch_policy",
    }
    groups = {}
    for node in ep_nodes:
        missing = required_attributes - set(node.attributes)
        if missing:
            raise ValueError(
                f"EP FUSED_MOE node {node.node_id!r} lacks attributes: "
                f"{sorted(missing)!r}")
        invocation_id = node.attributes["moe_invocation_id"]
        if not isinstance(invocation_id, str) or not invocation_id:
            raise TypeError(
                "EP FUSED_MOE moe_invocation_id must be a non-empty string")
        if invocation_id not in groups:
            groups[invocation_id] = []
        groups[invocation_id].append(node)

    for invocation_id, invocation_nodes in groups.items():
        first = invocation_nodes[0]
        pool_name = first.pool_name
        if pool_name not in deployment.worker_pools:
            raise ValueError(
                f"EP FUSED_MOE references unknown pool {pool_name!r}")
        pool = deployment.worker_pools[pool_name]
        plan = pool.parallel_plan
        ep_size = first.attributes["expert_parallel_size"]
        if len(invocation_nodes) != ep_size:
            raise ValueError(
                f"EP MoE invocation {invocation_id!r} requires one node per "
                f"rank: {len(invocation_nodes)} != {ep_size}")
        if plan.config.expert_parallel_size != ep_size:
            raise ValueError(
                "EP MoE invocation size disagrees with its deployment plan")
        if first.dtype != "bf16":
            raise NotImplementedError(
                "rank-local EP FUSED_MOE validation currently requires BF16")
        required_shape = {
            "tokens", "experts", "topk", "hidden", "intermediate"}
        if set(first.shape) != required_shape:
            raise ValueError(
                "EP FUSED_MOE shape must contain exactly tokens, experts, "
                "topk, hidden, and intermediate")
        if any(first.shape[name] <= 0 for name in required_shape):
            raise ValueError("EP FUSED_MOE shape dimensions must be positive")
        first_graph_domain = _graph_token_domain(first)
        if first_graph_domain is None:
            routing_tokens = first.shape["tokens"]
        else:
            routing_tokens, model_input_tokens, _ = first_graph_domain
            if first.shape["tokens"] != model_input_tokens:
                raise ValueError(
                    "EP FUSED_MOE physical tokens disagree with its graph "
                    "token-domain contract")
        if (
            first.shape["experts"] % ep_size
            or first.shape["topk"] > first.shape["experts"]
        ):
            raise NotImplementedError(
                "rank-local EP lowering requires experts divisible by EP and "
                "top-k no larger than the expert count")
        global_counts = tuple(first.attributes["tokens_per_expert"])
        expert_count = first.shape["experts"]
        if len(global_counts) != expert_count:
            raise ValueError(
                "EP global routing count must match the global expert count")
        if any(isinstance(count, bool) or not isinstance(count, int)
               or count < 0 for count in global_counts):
            raise TypeError(
                "EP global routing counts must be non-negative integers")
        if sum(global_counts) != routing_tokens * first.shape["topk"]:
            raise ValueError(
                "EP global routing counts must sum to logical input rows * "
                "top-k")
        if any(count > routing_tokens for count in global_counts):
            raise ValueError(
                "EP global routing count cannot exceed input tokens for any "
                "expert")

        expert_group = plan.rank_groups.for_rank("expert", 0)
        if len(expert_group) != ep_size:
            raise ValueError(
                "EP deployment rank group disagrees with expert_parallel_size")
        expected_participants = tuple(
            pool.physical_rank_for_logical_rank(rank)
            for rank in expert_group)
        nodes_by_rank = {}
        all_owned_experts = []
        for node in invocation_nodes:
            if (node.pool_name != pool_name
                    or dict(node.shape) != dict(first.shape)
                    or node.dtype != first.dtype
                    or node.variant != "full"
                    or node.attributes["expert_parallel_size"] != ep_size
                    or tuple(node.attributes["tokens_per_expert"])
                    != global_counts
                    or _graph_token_domain(node) != first_graph_domain
                    or node.attributes["moe_invocation_id"] != invocation_id
                    or dict(node.attributes["routing_provenance"])
                    != dict(first.attributes["routing_provenance"])):
                raise ValueError(
                    "EP MoE rank nodes disagree on the full-input invocation")
            if node.attributes["expert_placement_strategy"] != "linear":
                raise NotImplementedError(
                    "rank-local EP FUSED_MOE currently requires linear "
                    "expert placement")
            expert_rank = node.attributes["expert_rank"]
            if (isinstance(expert_rank, bool)
                    or not isinstance(expert_rank, int)
                    or not 0 <= expert_rank < ep_size):
                raise ValueError("EP FUSED_MOE expert_rank is outside the group")
            if expert_rank in nodes_by_rank:
                raise ValueError(
                    "EP MoE invocation contains a duplicate expert rank")
            expected_logical_rank = expert_group[expert_rank]
            expected_physical_rank = pool.physical_rank_for_logical_rank(
                expected_logical_rank)
            if node.rank != expected_physical_rank:
                raise ValueError(
                    "EP FUSED_MOE physical rank disagrees with expert_rank")
            owned_experts = tuple(node.attributes["owned_experts"])
            expected_owned_experts = tuple(
                expert_id
                for expert_id, owner_rank in enumerate(plan.expert_to_shard)
                if owner_rank == expert_rank
            )
            if owned_experts != expected_owned_experts:
                raise ValueError(
                    "EP FUSED_MOE owned_experts disagrees with the deployment "
                    "expert placement")
            local_counts = tuple(
                node.attributes["local_tokens_per_expert"])
            if len(local_counts) != expert_count:
                raise ValueError(
                    "EP local routing count must match the global expert count")
            if any(isinstance(count, bool) or not isinstance(count, int)
                   or count < 0 for count in local_counts):
                raise TypeError(
                    "EP local routing counts must be non-negative integers")
            expected_local_counts = tuple(
                count if expert_id in owned_experts else 0
                for expert_id, count in enumerate(global_counts)
            )
            if local_counts != expected_local_counts:
                raise ValueError(
                    "EP local routing counts disagree with global routing and "
                    "expert ownership")
            predecessors = graph.predecessors(node.node_id)
            if len(predecessors) != 1:
                raise ValueError(
                    "EP FUSED_MOE must have exactly one rank-local router "
                    "predecessor")
            router = graph.node(predecessors[0])
            if router.kind is not WorkloadKind.ROUTER or router.rank != node.rank:
                raise ValueError(
                    "EP FUSED_MOE predecessor must be the same-rank router")
            nodes_by_rank[expert_rank] = node
            all_owned_experts.extend(owned_experts)
        if tuple(sorted(all_owned_experts)) != tuple(range(expert_count)):
            raise ValueError(
                "EP expert ownership must cover every global expert exactly "
                "once")
        if set(nodes_by_rank) != set(range(ep_size)):
            raise ValueError("EP MoE invocation does not cover every expert rank")

        node_ids = {node.node_id for node in invocation_nodes}
        successor_ids = {
            edge.destination
            for edge in graph.edges
            if edge.source in node_ids
        }
        if len(successor_ids) != 1:
            raise ValueError(
                "EP MoE rank nodes must converge on exactly one output "
                "all-reduce")
        collective = graph.node(next(iter(successor_ids)))
        if (collective.kind is not WorkloadKind.COLLECTIVE
                or collective.pool_name != pool_name
                or collective.attributes["operation"] != "all_reduce"
                or collective.attributes["bytes_semantics"] != "per_rank_input"
                or collective.attributes["reduction"] != "sum"
                or tuple(collective.attributes["participants"])
                != expected_participants
                or collective.attributes["moe_invocation_id"] != invocation_id):
            raise ValueError(
                "EP MoE output must use the matching TP-group all-reduce")
        if _graph_token_domain(collective) != first_graph_domain:
            raise ValueError(
                "EP MoE output all-reduce must retain the producer graph "
                "token-domain contract")
        expected_bytes = (
            first.shape["tokens"] * first.shape["hidden"] * 2)
        if collective.attributes["bytes"] != expected_bytes:
            raise ValueError(
                "EP MoE output all-reduce bytes must equal B * H * BF16")
        if (len(graph.predecessors(collective.node_id)) != ep_size
                or set(graph.predecessors(collective.node_id)) != node_ids):
            raise ValueError(
                "EP MoE output all-reduce must wait for every expert rank")

    if any(
            node.kind is WorkloadKind.COLLECTIVE
            and node.attributes["operation"] == "all_to_all"
            for node in graph.nodes):
        raise ValueError(
            "TP=2, DP=1 EP workload must not emit all-to-all communication")


def _vector_shape(node: WorkloadNode) -> Mapping[str, int]:
    shape = dict(node.shape)
    elements = shape["elements"]
    if elements <= 0:
        raise ValueError(f"{node.kind.value} requires positive shape['elements']")
    del shape["elements"]
    shape["element_count"] = elements
    return shape




def _selected_target_compute_spec(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
    kind: KernelKind,
    variant: str,
    shape: Mapping[str, int],
    attributes: Mapping[str, object],
) -> KernelSpec:
    launch = select_launch_config(
        hardware_id,
        kind,
        shape,
        node.dtype,
        variant,
        node.attributes,
        hardware=hardware,
    )
    prediction_shape = dict(shape)
    prediction_attributes = dict(attributes)
    if kind is not KernelKind.FLASH_ATTN:
        # Graph-replayed target nodes carry padded physical rows; the
        # launch uses them, the prediction keeps the logical work (the same
        # contract as the generic single-kernel lowering).
        prediction_shape, domain = _logical_prediction_shape(node, shape)
        prediction_attributes.update(
            _kernel_graph_token_attributes(domain, shape))
    return KernelSpec(
        kind=kind,
        hardware_id=hardware_id,
        shape=prediction_shape,
        dtype=node.dtype,
        variant=launch.execution_variant,
        launch=launch,
        attributes=prediction_attributes,
    )


def _target_rope_spec(
        node: WorkloadNode, hardware_id: str, hardware) -> KernelSpec:
    if node.variant != "rotary":
        raise ValueError(
            f"target-model RoPE requires rotary variant, got {node.variant!r}")
    attributes = {
        name: node.attributes[name]
        for name in (
            "rope_layout", "rope_type", "rope_theta", "rope_factor",
            "rope_original_max_position_embeddings", "rope_beta_fast",
            "rope_beta_slow", "rope_mscale", "rope_mscale_all_dim",
        )
    }
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.VECTOR, "mla_rope",
        _vector_shape(node), attributes)


def _target_layernorm_spec(
        node: WorkloadNode, hardware_id: str, hardware) -> KernelSpec:
    if node.variant != "layernorm":
        raise ValueError(
            "target-model LayerNorm requires layernorm variant")
    attributes = {
        name: node.attributes[name]
        for name in ("affine", "epsilon")
    }
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.VECTOR, "layernorm",
        node.shape, attributes)


# DeepSeek-V3 dense projections that the official FP8 checkpoint stores as
# 128x128 block-scaled e4m3 weights (vLLM ``DeepseekV2`` builds each with the
# checkpoint ``quant_config``).  The absorbed MLA W_UK/W_UV batched GEMMs,
# the router and the LM head are not in this set.
_DSV3_BLOCK_FP8_LINEAR_POSITIONS = frozenset({
    "fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj",
    "gate_up_proj", "down_proj",
})


def _dsv3_block_fp8_checkpoint(
    graph: WorkloadGraph, target_contract: str | None,
) -> bool:
    """Whether the graph is a DeepSeek-V3 step of a block-FP8 checkpoint.

    The DeepSeek-V3 model contract admits FP8 weights only as the official
    128x128 e4m3 block-scaled checkpoint with dynamic 1x128 activation groups
    (``model_config``).  The MLP linears carry the checkpoint dtype resolved
    from the model (``weight_dtype_for_module``); the MLA projections are
    emitted by the builder shared with Kimi/GLM with a fixed ``bf16`` label
    that does not describe this checkpoint, so the checkpoint format is read
    from the MLP linears and applies to every projection in
    ``_DSV3_BLOCK_FP8_LINEAR_POSITIONS``.
    """

    if target_contract != "deepseek_v3_vllm_0_27_1_logical_v1":
        return False
    dtypes = {
        node.attributes["weight_dtype"] for node in graph.nodes
        if node.kind is WorkloadKind.LINEAR
        and node.attributes["position"] in ("gate_up_proj", "down_proj")
    }
    if len(dtypes) != 1:
        raise ValueError(
            "DeepSeek-V3 step must carry one MLP checkpoint weight dtype, "
            f"got {sorted(dtypes)}")
    return dtypes == {"fp8"}


def _block_fp8_linear_node(
    node: WorkloadNode, block_fp8_checkpoint: bool,
) -> WorkloadNode:
    """Declare the static block-FP8 weight format of one dense projection.

    The declaration is a launch-selection attribute only: a hardware that
    owns a block-FP8 GEMM selector routes the linear there, every other
    hardware ignores it, and no prediction attribute changes.
    """

    if (not block_fp8_checkpoint
            or node.kind is not WorkloadKind.LINEAR
            or node.attributes["position"]
            not in _DSV3_BLOCK_FP8_LINEAR_POSITIONS):
        return node
    if "weight_quantization" in node.attributes:
        raise ValueError(
            f"linear {node.node_id!r} already declares weight_quantization")
    return dataclasses.replace(node, attributes={
        **node.attributes,
        "weight_quantization": BLOCK_FP8_WEIGHT_QUANTIZATION,
    })


def _target_gemm_specs(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
    block_fp8_checkpoint: bool,
) -> tuple[KernelSpec, ...]:
    if node.variant == "batched_gemm":
        attributes = {
            name: node.attributes[name]
            for name in (
                "batch_layout", "a_batch_stride_elements",
                "b_batch_stride_elements", "c_batch_stride_elements",
                "weight_dtype", "weight_group_size",
            )
        }
        domain = _graph_token_domain(node)
        if domain is not None:
            # Strides describe the logical prediction operands; the padded
            # physical strides stay in the node's selection attributes.
            logical_tokens, model_input_tokens, _ = domain
            if node.shape["m"] != model_input_tokens:
                raise ValueError(
                    "padded batched GEMM rows disagree with the graph domain")
            attributes["a_batch_stride_elements"] = (
                logical_tokens * node.shape["k"])
            attributes["c_batch_stride_elements"] = (
                logical_tokens * node.shape["n"])
        return (_selected_target_compute_spec(
            node, hardware_id, hardware, KernelKind.GEMM, "batched_gemm",
            node.shape, attributes),)
    if node.kind is WorkloadKind.ROUTER:
        if node.variant != "gemm":
            raise ValueError(
                f"target-model router requires gemm variant, got "
                f"{node.variant!r}")
        attributes = {
            name: node.attributes[name]
            for name in (
                "score_function", "topk_method", "expert_group_count",
                "topk_group_count", "normalize_topk_probability",
                "weight_dtype", "weight_group_size",
            )
        }
        main = _selected_target_compute_spec(
            node, hardware_id, hardware, KernelKind.GEMM, "moe_router",
            node.shape, attributes)
        if (main.launch.tiling is not None
                or main.launch.parameters[
                    "split_k_partial_bytes_per_element"] == 0):
            return (main,)
        # An untiled router launch that writes external partials was observed
        # with its separate split-K reducer; the reducer launch is part of the
        # same observation.
        reducer_launch = select_gemm_reducer_launch(
            hardware_id, node.shape, node.dtype, "moe_router",
            node.attributes, hardware=hardware)
        element_count = (
            node.shape["b"] * node.shape["m"] * node.shape["n"])
        reducer = KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape={"element_count": element_count},
            dtype=main.dtype,
            variant=reducer_launch.execution_variant,
            launch=reducer_launch,
            attributes={
                # The split count is not observed; two partials is its lower
                # bound (the reducer prices the declared limitation).
                "split_k": 2,
                **_derived_graph_token_attributes(
                    main, {"element_count": element_count}),
            },
        )
        return main, reducer
    return _gemm_node_specs(
        _block_fp8_linear_node(node, block_fp8_checkpoint),
        hardware_id, hardware)


def _target_mla_attention_specs(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
) -> tuple[KernelSpec, ...]:
    if node.variant not in (
            "mla_prefill", "mla_decode", "mla_absorbed_prefill"):
        raise ValueError(
            f"unknown target MLA attention variant: {node.variant!r}")
    attributes = {
        name: node.attributes[name]
        for name in (
            "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens",
            "full_cuda_graphs", "max_cudagraph_capture_size")
    }
    if "attention_backend" in node.attributes:
        attributes["attention_backend"] = node.attributes["attention_backend"]
    graph_fields = (
        "graph_logical_tokens", "graph_model_input_tokens",
        "graph_runtime_mode")
    present_graph_fields = tuple(
        name for name in graph_fields if name in node.attributes)
    if present_graph_fields:
        if present_graph_fields != graph_fields:
            raise ValueError(
                "target MLA graph token domain is incomplete: "
                f"present={present_graph_fields!r}")
        if node.variant != "mla_decode":
            raise ValueError("only MLA decode carries a graph token domain")
        # Padded requests (query 0, seq_len 0) add no attention work; the
        # KernelSpec keeps the logical requests and the replay domain.
        attributes.update({name: node.attributes[name]
                           for name in graph_fields})
    main = _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.FLASH_ATTN, node.variant,
        node.shape, attributes)
    if node.variant in ("mla_prefill", "mla_absorbed_prefill"):
        return (main,)
    tiling = main.launch.tiling
    if tiling is None or tiling.num_kv_splits is None:
        raise ValueError("A800 MLA decode launch must declare KV splits")
    if tiling.num_kv_splits == 1:
        return (main,)
    combine_shape = {
        "batch": node.shape["batch"],
        "q_len": node.shape["q_len"],
        "kv_len": node.shape["kv_len"],
        "head_dim": node.shape["v_head_dim"],
        "num_query_heads": node.shape["num_query_heads"],
        "num_kv_heads": 1,
    }
    combine_attributes = {
        **{name: attributes[name] for name in (
            "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens")},
        "fa_ksteps": None,
        "fa_eff_mfrac": 1.0,
        "num_kv_splits": tiling.num_kv_splits,
    }
    combine_selection_attributes = dict(combine_attributes)
    if present_graph_fields:
        combine_selection_attributes.update(
            {name: node.attributes[name] for name in graph_fields})
    combine_launch = select_launch_config(
        hardware_id, KernelKind.FLASH_ATTN, combine_shape, node.dtype,
        "combine", combine_selection_attributes, hardware=hardware)
    combine = KernelSpec(
        kind=KernelKind.FLASH_ATTN,
        hardware_id=hardware_id,
        shape=combine_shape,
        dtype=node.dtype,
        variant="combine",
        launch=combine_launch,
        attributes={
            **{name: combine_attributes[name]
               for name in (
                   "causal", "kv_layout", "kv_dist", "fa_ksteps",
                   "fa_eff_mfrac", "q_lens", "kv_lens")},
            **({"graph_runtime_mode": node.attributes["graph_runtime_mode"]}
               if present_graph_fields else {}),
        },
    )
    return main, combine


def _target_sparse_indexer_spec(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
) -> KernelSpec:
    if node.variant == "dsa_index_reuse":
        attributes = {
            name: node.attributes[name]
            for name in (
                "reuse_semantics", "topk_capacity", "topk_index_dtype")
        }
        shape = _vector_shape(node)
    elif node.variant in (
            "dsa_indexer_prefill", "dsa_indexer_decode",
            "dsa_indexer_mixed"):
        attributes = {
            name: node.attributes[name]
            for name in (
                "causal", "score_accumulator_dtype", "score_function",
                "cache_layout", "cache_quantization", "q_lens", "kv_lens",
                "request_phases", "topk_capacity", "topk_index_dtype",
                "topk_sentinel",
            )
        }
        shape = node.shape
    else:
        raise ValueError(
            f"unknown DSA indexer variant: {node.variant!r}")
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.SPARSE_ATTN_INDEXER,
        node.variant, shape, attributes)


def _target_sparse_attention_spec(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
) -> KernelSpec:
    attributes = {
        name: node.attributes[name]
        for name in (
            "causal", "indexer_type", "kv_layout", "kv_dist", "q_lens",
            "kv_lens", "selected_kv_lens", "topk_capacity",
            "topk_index_dtype",
        )
    }
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.SPARSE_ATTN, node.variant,
        node.shape, attributes)


def _target_kv_cache_update_spec(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
) -> KernelSpec:
    shape = _vector_shape(node)
    if node.variant == "mla_kv_cache":
        attribute_names = ("cache_layout", "cache_write_semantics")
    elif node.variant == "dsa_fp8_index_cache":
        attribute_names = (
            "cache_layout", "cache_write_semantics", "quantization")
    else:
        raise ValueError(
            f"unknown target KV cache update variant: {node.variant!r}")
    attributes = {
        name: node.attributes[name]
        for name in attribute_names
    }
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.KV_CACHE,
        node.variant, shape, attributes)


def _flash_attn_selection_attributes(
    node: WorkloadNode,
    hardware_id: str,
) -> dict[str, object]:
    """Return the captured ABI fields without replacing logical work fields."""

    selection_attributes = dict(node.attributes)
    model_input_fields = {"model_input_q_lens", "model_input_kv_lens"}
    present_model_input_fields = model_input_fields & set(
        selection_attributes)
    if present_model_input_fields:
        if (present_model_input_fields != model_input_fields
                or "graph_runtime_mode" not in selection_attributes):
            raise ValueError(
                "FLASH_ATTN graph domains must be complete before launch "
                f"selection: present={sorted(present_model_input_fields)}")
        # vLLM-Ascend re-derives the FIA launch at every FULL-graph replay
        # via ``update_graph_params`` with the runtime ``seq_lens_list``:
        # real requests keep their logical lengths and the padded slots are
        # zero-filled (``model_runner_v1.py:732-734``).  Publish the logical
        # request lengths alongside the captured slot arrays so hardware
        # launch selectors can reflect that replay ABI.  The keys are inert
        # for selectors that do not consume them.
        selection_attributes["graph_logical_q_lens"] = tuple(
            selection_attributes["q_lens"])
        selection_attributes["graph_logical_kv_lens"] = tuple(
            selection_attributes["kv_lens"])
        selection_attributes["q_lens"] = tuple(
            selection_attributes["model_input_q_lens"])
        selection_attributes["kv_lens"] = tuple(
            selection_attributes["model_input_kv_lens"])
    normalized = dict(normalize_flash_attn_execution_attributes(
        hardware_id, node.shape, selection_attributes))
    normalized["is_decode"] = node.variant.lower() == "decode"
    return normalized


def _compute_spec(node: WorkloadNode, hardware_id: str, hardware) -> KernelSpec:
    if node.dtype is None or not node.dtype.strip():
        raise ValueError(f"compute workload {node.node_id!r} requires dtype")
    if node.variant is None or not node.variant.strip():
        raise ValueError(f"compute workload {node.node_id!r} requires variant")
    kind = _SINGLE_COMPUTE_KIND[node.kind]
    if kind is KernelKind.VECTOR and node.kind is WorkloadKind.NORM:
        physical_shape = node.shape
    elif kind is KernelKind.KV_CACHE:
        physical_shape = _vector_shape(node)
    elif kind is KernelKind.VECTOR:
        physical_shape = _vector_shape(node)
    elif kind in (KernelKind.GEMM, KernelKind.FLASH_ATTN):
        physical_shape = node.shape
    else:
        raise ValueError(f"unhandled compute shape kind: {kind.value}")
    if kind is KernelKind.FLASH_ATTN:
        shape = dict(physical_shape)
        token_domain = None
    else:
        shape, token_domain = _logical_prediction_shape(node, physical_shape)
    selection_attributes = dict(node.attributes)
    prediction_attributes = {}
    if (kind is KernelKind.VECTOR
            and node.kind is WorkloadKind.ELEMENTWISE
            and node.variant not in (
                "elementwise", "add", "masked-fill-zero")):
        raise ValueError(
            "unsupported elementwise workload variant: "
            f"{node.variant!r}")
    if kind is KernelKind.FLASH_ATTN:
        selection_attributes = _flash_attn_selection_attributes(
            node, hardware_id)
        prediction_attributes = {
            name: node.attributes[name]
            for name in (
                "causal", "kv_layout", "kv_dist", "fa_ksteps",
                "fa_eff_mfrac", "q_lens", "kv_lens")
        }
        model_input_fields = (
            "model_input_q_lens", "model_input_kv_lens",
        )
        present_model_input_fields = tuple(
            name for name in model_input_fields if name in node.attributes)
        if present_model_input_fields:
            if (present_model_input_fields != model_input_fields
                    or "graph_runtime_mode" not in node.attributes):
                raise ValueError(
                    "FLASH_ATTN graph prediction domains must be complete")
            prediction_attributes.update({
                name: node.attributes[name]
                for name in (*model_input_fields, "graph_runtime_mode")
            })
    elif kind is KernelKind.VECTOR:
        if node.kind in (WorkloadKind.ELEMENTWISE, WorkloadKind.VECTOR):
            prediction_attribute_names = ("operation", "reads", "writes")
        elif node.kind is WorkloadKind.GATHER:
            prediction_attribute_names = (
                "gather_factor", "index_count", "index_bytes", "scattered")
        elif node.kind is WorkloadKind.REDUCE:
            prediction_attribute_names = ("reduction_factor",)
        elif node.kind in (
                WorkloadKind.NORM, WorkloadKind.ACTIVATION,
                WorkloadKind.ROTARY, WorkloadKind.VECTOR):
            prediction_attribute_names = ()
        else:
            raise AssertionError(
                f"unhandled VECTOR workload kind: {node.kind.value}")
        prediction_attributes = {
            name: node.attributes[name]
            for name in prediction_attribute_names
        }
        if (node.kind is WorkloadKind.ELEMENTWISE
                and "helper_launch" in node.attributes):
            # Helper/glue emission contract attribute: structural counting in
            # the alignment harness reads it from spec.attributes, so the
            # prediction-attribute filter must forward it explicitly.
            prediction_attributes["helper_launch"] = (
                node.attributes["helper_launch"])
        if (node.kind is WorkloadKind.GATHER
                and token_domain is not None):
            logical_tokens, model_input_tokens, _ = token_domain
            index_count = prediction_attributes["index_count"]
            if index_count % model_input_tokens:
                raise ValueError(
                    "GATHER index_count is not an integral graph-token "
                    "multiple")
            prediction_attributes["index_count"] = (
                index_count // model_input_tokens * logical_tokens)
    elif kind is KernelKind.KV_CACHE:
        prediction_attributes = {
            name: node.attributes[name]
            for name in (
                "cache_layout", "cache_head_layout", "kv_cache_dtype",
                "cache_storage_matches_input", "slot_mapping_all_valid",
                "vector_copy_base_alignment_bytes",
            )
        }
    elif kind is KernelKind.GEMM:
        prediction_attributes = {}
        if "helper_launch" in node.attributes:
            # Helper/glue emission contract attribute (e.g. the router-run
            # stamp on the ROUTER GEMM): structural counting in the
            # alignment harness reads it from spec.attributes.
            prediction_attributes["helper_launch"] = (
                node.attributes["helper_launch"])
    else:
        raise ValueError(
            f"unhandled prediction attributes for {kind.value}")
    vector_variants = {
        WorkloadKind.ROTARY: "rotary",
        WorkloadKind.REDUCE: "reduce",
        WorkloadKind.GATHER: "gather",
        WorkloadKind.VECTOR: "elementwise",
    }
    if kind is KernelKind.VECTOR:
        if node.kind in (WorkloadKind.NORM, WorkloadKind.ACTIVATION):
            variant = node.variant
            prediction_attributes = {}
        elif node.kind is WorkloadKind.ELEMENTWISE:
            elementwise_variants = {
                "elementwise": "elementwise",
                "add": "elementwise",
                "masked-fill-zero": "elementwise",
            }
            if node.variant not in elementwise_variants:
                raise ValueError(
                    "unsupported elementwise workload variant: "
                    f"{node.variant!r}")
            variant = elementwise_variants[node.variant]
        elif node.kind in vector_variants:
            variant = vector_variants[node.kind]
        else:
            raise ValueError(f"no VECTOR variant for {node.kind.value}")
    elif kind is KernelKind.KV_CACHE:
        if node.variant != "paged_kv_cache":
            raise ValueError(
                "KV cache workload requires canonical paged_kv_cache "
                f"variant, got {node.variant!r}")
        variant = node.variant
    elif kind in (KernelKind.GEMM, KernelKind.FLASH_ATTN):
        variant = node.variant
    else:
        raise ValueError(f"unhandled compute variant kind: {kind.value}")
    if kind is not KernelKind.FLASH_ATTN:
        prediction_attributes.update(_kernel_graph_token_attributes(
            token_domain, physical_shape))
    launch = select_launch_config(
        hardware_id, kind, physical_shape, node.dtype, variant,
        selection_attributes, hardware=hardware)
    return KernelSpec(
        kind=kind,
        hardware_id=hardware_id,
        shape=shape,
        dtype=node.dtype,
        variant=launch.execution_variant,
        launch=launch,
        attributes=prediction_attributes,
    )


def _flash_attn_specs(node: WorkloadNode, hardware_id: str, hardware):
    # The backend partitions the request ABI it is called with: the padded
    # model-input requests inside a FULL Graph replay (vLLM pads attention
    # metadata only there), the logical requests otherwise.
    if ("graph_runtime_mode" in node.attributes
            and node.attributes["graph_runtime_mode"] == "FULL"
            and "model_input_q_lens" in node.attributes):
        q_lens = tuple(node.attributes["model_input_q_lens"])
        kv_lens = tuple(node.attributes["model_input_kv_lens"])
    else:
        q_lens = tuple(node.attributes["q_lens"])
        kv_lens = tuple(node.attributes["kv_lens"])
    parts = select_flash_attn_forward_partition(
        hardware_id, node.shape, node.variant,
        {**node.attributes, "q_lens": q_lens, "kv_lens": kv_lens})
    if not parts:
        return _whole_flash_attn_specs(
            node, hardware_id, hardware, output_fill_rows=None)
    # The backend runs the call as several pathways over disjoint request
    # subsets, each on its own requests, then (when the backend allocates a
    # fresh result) slice-assigns that result into the shared output buffer.
    # The output buffer belongs to the whole call: it is zero-filled once,
    # ahead of every pathway, over all of the call's query rows.  The first
    # pathway carries that fill; the others share the buffer and emit none.
    call_rows = sum(q_lens)
    specs = []
    for position, part in enumerate(parts):
        part_q_lens = tuple(q_lens[index] for index in part.request_indexes)
        part_kv_lens = tuple(kv_lens[index] for index in part.request_indexes)
        attributes = {
            name: value for name, value in node.attributes.items()
            if name not in ("model_input_q_lens", "model_input_kv_lens")
        }
        attributes.update({
            "q_lens": part_q_lens,
            "kv_lens": part_kv_lens,
            "flash_attn_forward_role": part.role,
        })
        part_node = dataclasses.replace(
            node,
            shape={
                **node.shape,
                "batch": len(part_q_lens),
                "q_len": max(part_q_lens),
                "kv_len": max(part_kv_lens),
            },
            variant=part.variant,
            attributes=attributes,
        )
        specs.extend(_whole_flash_attn_specs(
            part_node, hardware_id, hardware,
            output_fill_rows=call_rows if position == 0 else 0))
        if part.output_copy_launch is not None:
            if (part.output_copy_launch.algorithm
                    != "native_d2d_flash_attention_copy"):
                raise ValueError(
                    "FlashAttention pathway output copy must be the native "
                    "D2D FlashAttention copy")
            if node.dtype != "bf16":
                raise ValueError(
                    "FlashAttention pathway output copy supports bf16 only")
            specs.append(KernelSpec(
                kind=KernelKind.MEMCPY,
                hardware_id=hardware_id,
                transfer=PhysicalTransferConfig(
                    algorithm="native_d2d_flash_attention_copy",
                    bytes=part.output_copy_elements * 2,
                    direction="d2d",
                    link="dram",
                ),
            ))
    return tuple(specs)


def _whole_flash_attn_specs(
        node: WorkloadNode, hardware_id: str, hardware,
        output_fill_rows: int | None):
    """Lower one backend FlashAttention invocation.

    ``output_fill_rows`` is ``None`` for a call lowered whole, whose output
    fill covers its own execution shape; a pathway of a partitioned call
    passes the query rows of the whole call it zero-fills, or ``0`` when an
    earlier pathway of the same call already carries that fill.
    """

    main = _compute_spec(node, hardware_id, hardware)
    tiling = main.launch.tiling
    if tiling is None or tiling.num_kv_splits is None:
        raise ValueError("FLASH_ATTN launch must declare num_kv_splits")
    splits = tiling.num_kv_splits
    execution_attributes = _flash_attn_selection_attributes(
        node, hardware_id)
    setup_specs = tuple(
        KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape=selection.shape,
            dtype=selection.dtype,
            variant=selection.variant,
            launch=selection.launch,
            attributes={
                **selection.attributes,
                **_derived_graph_token_attributes(main, selection.shape),
            },
        )
        for selection in select_flash_attn_setup_launches(
            hardware_id, main.shape, main.dtype, main.variant,
            execution_attributes, hardware)
    )
    compute_specs = [main]
    if splits > 1:
        selection_attributes = execution_attributes
        selection_attributes["num_kv_splits"] = splits
        combine_launch = select_launch_config(
            hardware_id, KernelKind.FLASH_ATTN, main.shape, main.dtype,
            "combine", selection_attributes, hardware=hardware)
        compute_specs.append(KernelSpec(
            kind=KernelKind.FLASH_ATTN,
            hardware_id=hardware_id,
            shape=main.shape,
            dtype=main.dtype,
            variant="combine",
            launch=combine_launch,
            attributes=main.attributes,
        ))

    def copy_spec(dtype: str, elements: int, launch) -> KernelSpec:
        if launch.algorithm == "native_d2d_flash_attention_copy":
            dtype_bytes = {"bf16": 2, "float32": 4}
            if dtype not in dtype_bytes:
                raise ValueError(
                    f"unsupported native FlashAttention copy dtype: {dtype!r}")
            # A native copy launch is a physical D2D transfer rather than a
            # Vector compute kernel.  The selector's element count is the
            # source-defined output view: one read and one write per value.
            return KernelSpec(
                kind=KernelKind.MEMCPY,
                hardware_id=hardware_id,
                transfer=PhysicalTransferConfig(
                    algorithm="native_d2d_flash_attention_copy",
                    bytes=elements * dtype_bytes[dtype],
                    direction="d2d",
                    link="dram",
                ),
            )
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape={"element_count": elements},
            dtype=dtype,
            variant="elementwise",
            launch=launch,
            attributes={
                "reads": 1,
                "writes": 1,
                "operation": "copy",
                **_derived_graph_token_attributes(
                    main, {"element_count": elements}),
            },
        )

    def fill_spec(dtype: str, elements: int, launch) -> KernelSpec:
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape={"element_count": elements},
            dtype=dtype,
            variant="elementwise",
            launch=launch,
            attributes={
                "reads": 0,
                "writes": 1,
                "operation": "fill",
                **_derived_graph_token_attributes(
                    main, {"element_count": elements}),
            },
        )

    # The backend-owned output buffer zero-fill (vLLM Attention.forward
    # use_output branch) runs before the framework copies that surround the
    # captured FA invocation, on the same normalized padded execution shape.
    if output_fill_rows is None:
        fill_shape = main.shape
    else:
        # One row per query token of the whole call.
        fill_shape = {**main.shape, "batch": output_fill_rows, "q_len": 1}
    output_fills = () if output_fill_rows == 0 else tuple(
        fill_spec(dtype, elements, launch)
        for dtype, elements, launch in select_flash_attn_output_fill_launches(
            hardware_id, fill_shape, main.dtype, execution_attributes,
            hardware))
    copy_launches = select_flash_attn_copy_launches(
        # A framework copy surrounds the physical captured FA invocation.
        # Keep its selector on the same normalized execution ABI as main,
        # setup, and combine: q_lens/kv_lens here describe the captured
        # request slots, never the logical active-row workload.
        hardware_id, main.shape, execution_attributes, hardware)
    before = tuple(
        copy_spec(dtype, elements, launch)
        for position, dtype, elements, launch in copy_launches
        if position == "before")
    after = tuple(
        copy_spec(dtype, elements, launch)
        for position, dtype, elements, launch in copy_launches
        if position == "after")

    # vLLM-Ascend's ``Attention.forward`` performs source-visible TensorMove
    # assignments around the fused FIA call.  CANN's FIA selector does not
    # expose those framework moves, so the workload builder carries the
    # source-authored logical/physical row plan and 910B materializes it here.
    # This is intentionally a discrete source contract; no serving duration
    # or residual correction participates in the launch choice.
    source_copies = ()
    if (hardware.kernel_memory_model == "npu_tile"
            and hardware.aic_aiv_separated
            and "source_output_copy_plan" in node.attributes):
        source_copy_plan = node.attributes["source_output_copy_plan"]
        if (not isinstance(source_copy_plan, tuple)
                or not source_copy_plan
                or any(role not in ("logical", "physical")
                       for role in source_copy_plan)):
            raise ValueError(
                "910B FlashAttention source_output_copy_plan must contain "
                "logical/physical roles")
        q_lens = node.attributes["q_lens"]
        model_input_q_lens = (
            node.attributes["model_input_q_lens"]
            if "model_input_q_lens" in node.attributes else q_lens)
        if (not isinstance(q_lens, tuple)
                or not isinstance(model_input_q_lens, tuple)):
            raise ValueError(
                "910B FlashAttention output-copy row domains must be tuples")
        logical_rows = sum(q_lens)
        physical_rows = sum(model_input_q_lens)
        if logical_rows <= 0 or physical_rows < logical_rows:
            raise ValueError(
                "910B FlashAttention output-copy rows are inconsistent")
        hidden = node.shape["num_query_heads"] * node.shape["head_dim"]
        selection_attributes = {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
            "launch_policy": node.attributes["launch_policy"],
        }
        copy_attributes = {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
        }
        generated = []
        for index, role in enumerate(source_copy_plan):
            rows = logical_rows if role == "logical" else physical_rows
            shape = {"element_count": rows * hidden}
            launch = select_launch_config(
                hardware_id,
                KernelKind.VECTOR,
                shape,
                node.dtype,
                "elementwise",
                selection_attributes,
                hardware=hardware,
            )
            generated.append(KernelSpec(
                kind=KernelKind.VECTOR,
                hardware_id=hardware_id,
                shape=shape,
                dtype=node.dtype,
                variant=launch.execution_variant,
                launch=launch,
                attributes={
                    **copy_attributes,
                    **_derived_graph_token_attributes(main, shape),
                },
            ))
        source_copies = tuple(generated)
    return (setup_specs + output_fills + before + tuple(compute_specs)
            + after + source_copies)


_NORM_INPUT_MATERIALIZATION_FIELDS = (
    "input_materialization",
    "input_outer_groups",
    "input_rows_per_group",
    "input_group_stride_elements",
    "input_row_stride_elements",
)


def _norm_specs(node: WorkloadNode, hardware_id: str, hardware):
    """Lower a source-required input materialization before the Norm call."""

    main = _compute_spec(node, hardware_id, hardware)
    present = tuple(
        name for name in _NORM_INPUT_MATERIALIZATION_FIELDS
        if name in node.attributes)
    if not present:
        return (main,)
    if present != _NORM_INPUT_MATERIALIZATION_FIELDS:
        raise ValueError(
            "Norm input materialization contract is incomplete: "
            f"present={present!r}")
    if node.attributes["input_materialization"] != (
            "grouped_strided_to_contiguous"):
        raise ValueError(
            "unsupported Norm input materialization: "
            f"{node.attributes['input_materialization']!r}")
    dimensions = {
        name: node.attributes[name]
        for name in _NORM_INPUT_MATERIALIZATION_FIELDS[1:]
    }
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in dimensions.values()):
        raise TypeError(
            "Norm input materialization dimensions must be integers")
    if any(value <= 0 for value in dimensions.values()):
        raise ValueError(
            "Norm input materialization dimensions must be positive")
    rows = node.shape["rows"]
    hidden = node.shape["hidden"]
    if dimensions["input_outer_groups"] * dimensions[
            "input_rows_per_group"] != rows:
        raise ValueError(
            "Norm grouped input rows disagree with the normalized shape")
    if dimensions["input_row_stride_elements"] != hidden:
        raise ValueError(
            "Norm grouped input row stride must equal hidden width")
    if dimensions["input_group_stride_elements"] <= (
            dimensions["input_rows_per_group"] * hidden):
        raise ValueError(
            "Norm grouped input must contain a physical inter-group gap")
    selection_attributes = {
        "operation": "copy",
        "reads": 1,
        "writes": 1,
        "launch_policy": node.attributes["launch_policy"],
        "input_materialization": node.attributes["input_materialization"],
        **dimensions,
    }
    copy_shape = {"element_count": rows * hidden}
    copy_launch = select_launch_config(
        hardware_id,
        KernelKind.VECTOR,
        copy_shape,
        node.dtype,
        "elementwise",
        selection_attributes,
        hardware=hardware,
    )
    copy = KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape=copy_shape,
        dtype=node.dtype,
        variant=copy_launch.execution_variant,
        launch=copy_launch,
        attributes={
            "reads": 1,
            "writes": 1,
            "operation": "copy",
            "input_materialization": node.attributes[
                "input_materialization"],
            **dimensions,
            **_derived_graph_token_attributes(main, copy_shape),
        },
    )
    return copy, main


def _ascend_qkv_rmsnorm_rope_fusions(
        graph: WorkloadGraph, deployment: DeploymentPlan,
        hardware_by_id: Mapping[str, object]):
    """Identify source-exact vLLM-Ascend Q/K Norm plus RoPE programs.

    The compiler pass matches a structural three-op chain and replaces it
    with one physical split-QKV/RMSNorm/RoPE task. This matcher mirrors that
    graph contract; it does not inspect model names, Serving case IDs, or
    measured durations.
    """

    # The canonical launch observation contradicts the fused-program
    # premise on every captured NPU serving case: the launch signatures of
    # the {qwen3-4b,qwen3-8b}/tp1_qps16 alignment captures under
    # serving_dataset/sharegpt/<hardware>/
    # contain only the separate AddRmsNormBias, RmsNorm and _triton_rope
    # kernels and no split_qkv_rmsnorm_rope signature, and the msprof
    # per-kernel counts agree (73 RmsNorm + 72 AddRmsNormBias + 36
    # _triton_rope per step for a 36-layer model).  Emitting one fused task
    # therefore under-counts both the launch occurrence and its device work.
    # Keep the matcher inert until a capture actually shows the fused
    # signature; the Q/K Norm and RoPE nodes then lower through their own
    # measured vector programs.
    return {}, {}
    successors = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        successors[edge.source].append(edge.destination)
    head_to_group = {}
    member_to_head = {}
    for rotary in graph.nodes:
        if (rotary.kind is not WorkloadKind.ROTARY
                or rotary.variant != "rotary"
                or rotary.dtype != "bf16"):
            continue
        if rotary.pool_name not in deployment.worker_pools:
            continue
        pool = deployment.worker_pools[rotary.pool_name]
        if pool.hardware_id not in hardware_by_id:
            continue
        hardware = hardware_by_id[pool.hardware_id]
        if (hardware.kernel_memory_model != "npu_tile"
                or not hardware.aic_aiv_separated):
            continue
        k_predecessors = graph.predecessors(rotary.node_id)
        if len(k_predecessors) != 1:
            continue
        k_norm = graph.node(k_predecessors[0])
        q_predecessors = graph.predecessors(k_norm.node_id)
        if len(q_predecessors) != 1:
            continue
        q_norm = graph.node(q_predecessors[0])
        group = (q_norm, k_norm, rotary)
        if any(
                node.kind is not WorkloadKind.NORM
                or node.variant != "rmsnorm"
                or node.dtype != "bf16"
                for node in group[:2]):
            continue
        if any(
                node.pool_name != rotary.pool_name
                or node.rank != rotary.rank
                for node in group[:2]):
            continue
        if (tuple(successors[q_norm.node_id]) != (k_norm.node_id,)
                or tuple(successors[k_norm.node_id]) != (rotary.node_id,)):
            continue
        materialization = "grouped_strided_to_contiguous"
        if any(
                "input_materialization" not in node.attributes
                or node.attributes["input_materialization"] != materialization
                for node in group[:2]):
            continue
        if (set(rotary.shape) != {
                "elements", "seq_len", "num_query_heads", "num_kv_heads",
                "head_dim"}
                or set(q_norm.shape) != {"rows", "hidden"}
                or set(k_norm.shape) != {"rows", "hidden"}):
            continue
        tokens = rotary.shape["seq_len"]
        query_heads = rotary.shape["num_query_heads"]
        kv_heads = rotary.shape["num_kv_heads"]
        head_dim = rotary.shape["head_dim"]
        if head_dim != 128:
            # QKNormRopeFusionPass registers only the source program whose
            # compile-time HEAD_DIM is 128. This is a versioned compiler
            # eligibility rule, not an empirical performance threshold.
            continue
        if (q_norm.shape != {
                "rows": tokens * query_heads, "hidden": head_dim}
                or k_norm.shape != {
                    "rows": tokens * kv_heads, "hidden": head_dim}
                or rotary.shape["elements"] != (
                    tokens * (query_heads + kv_heads) * head_dim)):
            continue
        stride = (query_heads + 2 * kv_heads) * head_dim
        expected_materialization = (
            (tokens, query_heads, stride, head_dim),
            (tokens, kv_heads, stride, head_dim),
        )
        actual_materialization = tuple(
            (
                node.attributes["input_outer_groups"],
                node.attributes["input_rows_per_group"],
                node.attributes["input_group_stride_elements"],
                node.attributes["input_row_stride_elements"],
            )
            for node in group[:2]
        )
        if actual_materialization != expected_materialization:
            continue
        graph_domains = tuple(_graph_token_domain(node) for node in group)
        if any(domain is not None for domain in graph_domains):
            if graph_domains[0] is None or any(
                    domain != graph_domains[0] for domain in graph_domains):
                raise ValueError(
                    "Ascend Q/K Norm RoPE fusion requires one shared graph "
                    "token-domain contract")
            _, model_input_tokens, _ = graph_domains[0]
            if tokens != model_input_tokens:
                raise ValueError(
                    "Ascend Q/K Norm RoPE fusion physical sequence length "
                    "disagrees with its graph model-input tokens")
        if any(node.node_id in member_to_head for node in group):
            raise ValueError("overlapping Ascend Q/K Norm RoPE fusion groups")
        head_to_group[q_norm.node_id] = group
        for member in group:
            member_to_head[member.node_id] = q_norm.node_id
    return head_to_group, member_to_head


def _ascend_qkv_rmsnorm_rope_spec(
        q_norm: WorkloadNode,
        k_norm: WorkloadNode,
        rotary: WorkloadNode,
        hardware_id: str,
        hardware,
) -> KernelSpec:
    physical_shape = {
        "seq_len": rotary.shape["seq_len"],
        "num_query_heads": rotary.shape["num_query_heads"],
        "num_kv_heads": rotary.shape["num_kv_heads"],
        "head_dim": rotary.shape["head_dim"],
        "rotary_dim": rotary.shape["head_dim"],
    }
    total_heads = (
        physical_shape["num_query_heads"]
        + 2 * physical_shape["num_kv_heads"])
    physical_shape["element_count"] = (
        physical_shape["seq_len"]
        * total_heads
        * physical_shape["head_dim"])
    domain = _graph_token_domain(rotary)
    if domain is not None:
        if (_graph_token_domain(q_norm) != domain
                or _graph_token_domain(k_norm) != domain):
            raise ValueError(
                "Ascend Q/K Norm RoPE fusion graph token domains disagree")
        logical_tokens, model_input_tokens, _ = domain
        if physical_shape["seq_len"] != model_input_tokens:
            raise ValueError(
                "Ascend Q/K Norm RoPE fusion physical sequence length "
                "disagrees with graph model-input tokens")
        shape = dict(physical_shape)
        shape["seq_len"] = logical_tokens
        shape["element_count"] = (
            logical_tokens * total_heads * shape["head_dim"])
    else:
        shape = dict(physical_shape)
    attributes = {"bias": False, "is_neox": True}
    selection_attributes = {
        **attributes,
        "launch_policy": rotary.attributes["launch_policy"],
    }
    launch = select_launch_config(
        hardware_id,
        KernelKind.VECTOR,
        physical_shape,
        rotary.dtype,
        "qkv_rmsnorm_rope",
        selection_attributes,
        hardware=hardware,
    )
    return KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape=shape,
        dtype=rotary.dtype,
        variant=launch.execution_variant,
        launch=launch,
        attributes={
            **attributes,
            **_kernel_graph_token_attributes(domain, physical_shape),
        },
    )


def _ascend_qkv_rmsnorm_rope_specs(
        q_norm: WorkloadNode,
        k_norm: WorkloadNode,
        rotary: WorkloadNode,
        hardware_id: str,
        hardware,
) -> tuple[KernelSpec, ...]:
    """Full spec tuple for the Ascend QKV-RMSNorm-RoPE fusion.

    The fused launch reads contiguous q/k rows, so the same-rank q/k Norm
    input materializations (``grouped_strided_to_contiguous``, one copy
    each, already validated by the fusion matcher) precede the fused
    kernel, followed by the V materializations whose count the workload
    builder resolved from the versioned emission observation.  Every helper
    copy is a plain VECTOR elementwise copy priced by the existing copy
    family; no serving duration or residual correction participates.
    """

    fused = _ascend_qkv_rmsnorm_rope_spec(
        q_norm, k_norm, rotary, hardware_id, hardware)

    def copy_spec(node: WorkloadNode, elements: int) -> KernelSpec:
        shape = {"element_count": elements}
        selection_attributes = {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
            "launch_policy": node.attributes["launch_policy"],
        }
        launch = select_launch_config(
            hardware_id,
            KernelKind.VECTOR,
            shape,
            node.dtype,
            "elementwise",
            selection_attributes,
            hardware=hardware,
        )
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape=shape,
            dtype=node.dtype,
            variant=launch.execution_variant,
            launch=launch,
            attributes={
                "operation": "copy",
                "reads": 1,
                "writes": 1,
                **_derived_graph_token_attributes(fused, shape),
            },
        )

    copies = [
        copy_spec(
            q_norm, q_norm.shape["rows"] * q_norm.shape["hidden"]),
        copy_spec(
            k_norm, k_norm.shape["rows"] * k_norm.shape["hidden"]),
    ]
    if "qkv_v_materialization_count" not in rotary.attributes:
        raise ValueError(
            "Ascend QKV-RMSNorm-RoPE fusion requires the "
            "qkv_v_materialization_count emission contract on the rotary "
            "node")
    v_count = rotary.attributes["qkv_v_materialization_count"]
    if (isinstance(v_count, bool) or not isinstance(v_count, int)
            or v_count < 0):
        raise ValueError(
            "qkv_v_materialization_count must be a non-negative integer; "
            f"found {v_count!r}")
    v_elements = (
        rotary.shape["seq_len"]
        * rotary.shape["num_kv_heads"]
        * rotary.shape["head_dim"])
    for _ in range(v_count):
        copies.append(copy_spec(rotary, v_elements))
    return tuple(copies) + (fused,)


def _block_fp8_input_quant_specs(
        main: KernelSpec, selection_shape: Mapping[str, int],
        hardware_id: str, hardware) -> tuple[KernelSpec, ...]:
    """The separate activation quantiser launched before a block-FP8 GEMM."""

    element_count = (
        selection_shape["b"] * selection_shape["m"] * selection_shape["k"])
    quant_launch = select_block_fp8_input_quant_launch(
        hardware_id, main.launch, element_count, hardware)
    if quant_launch is None:
        return ()
    shape = {"element_count": element_count}
    return (KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape=shape,
        dtype=main.dtype,
        variant=quant_launch.execution_variant,
        launch=quant_launch,
        attributes={
            "group_size": PER_TOKEN_GROUP_QUANT_GROUP_SIZE,
            **_derived_graph_token_attributes(main, shape),
        },
    ),)


def _gemm_specs(
        main: KernelSpec, selection_shape: Mapping[str, int],
        selection_variant: str,
        selection_attributes: Mapping[str, object],
        hardware_id: str, hardware):
    return (
        _block_fp8_input_quant_specs(main, selection_shape, hardware_id, hardware)
        + _gemm_main_and_reducer_specs(
            main, selection_shape, selection_variant, selection_attributes,
            hardware_id, hardware))


def _gemm_main_and_reducer_specs(
        main: KernelSpec, selection_shape: Mapping[str, int],
        selection_variant: str,
        selection_attributes: Mapping[str, object],
        hardware_id: str, hardware):
    tiling = main.launch.tiling
    if tiling is None:
        if main.launch.execution_variant == "gemv":
            return (main,)
        raise ValueError("production GEMM launch must declare tiling")
    if tiling.split_k is None:
        raise ValueError("production GEMM launch must declare split_k")
    if tiling.split_k == 1:
        return (main,)
    reducer_launch = select_gemm_reducer_launch(
        hardware_id, selection_shape, main.dtype, selection_variant,
        selection_attributes, hardware=hardware)
    if reducer_launch is None:
        return (main,)
    reducer = KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape={
            "element_count": (
                selection_shape["b"]
                * selection_shape["m"]
                * selection_shape["n"]),
        },
        dtype=main.dtype,
        variant=reducer_launch.execution_variant,
        launch=reducer_launch,
        attributes={
            "split_k": tiling.split_k,
            **_derived_graph_token_attributes(
                main,
                {"element_count": (
                    selection_shape["b"]
                    * selection_shape["m"]
                    * selection_shape["n"]),
                },
            ),
        },
    )
    return main, reducer


def _gemm_node_specs(node: WorkloadNode, hardware_id: str, hardware):
    return _gemm_specs(
        _compute_spec(node, hardware_id, hardware),
        node.shape,
        node.variant,
        node.attributes,
        hardware_id, hardware)


def _fused_moe_prediction_attributes(
    node: WorkloadNode,
    routing_relabelling=None,
) -> dict[str, object]:
    _validate_routing_provenance(node)
    shape = node.shape
    experts = shape["experts"]
    counts = tuple(node.attributes["tokens_per_expert"])
    if len(counts) != experts:
        raise ValueError("FUSED_MOE routing count must match expert count")
    expert_parallel_size = node.attributes["expert_parallel_size"]
    if expert_parallel_size == 1:
        local_counts = counts
        expert_rank = 0
        owned_experts = tuple(range(experts))
    elif expert_parallel_size > 1:
        local_counts = tuple(node.attributes["local_tokens_per_expert"])
        expert_rank = node.attributes["expert_rank"]
        owned_experts = tuple(node.attributes["owned_experts"])
    else:
        raise ValueError("expert_parallel_size must be positive")
    if routing_relabelling is not None:
        counts, local_counts = routing_relabelling
    attributes = {
        "tokens_per_expert": counts,
        "local_tokens_per_expert": local_counts,
        "expert_parallel_size": expert_parallel_size,
        "expert_rank": expert_rank,
        "owned_experts": owned_experts,
        "expert_weight_tile_rows": None,
    }
    target_names = {
        "routed_scaling_factor", "routed_weight_dtype", "weight_group_size",
    }
    present_target_names = target_names & set(node.attributes)
    if present_target_names:
        if present_target_names != target_names:
            raise ValueError(
                "target FUSED_MOE routing semantics must be complete: "
                f"present={sorted(present_target_names)}")
        for name in target_names:
            attributes[name] = node.attributes[name]
    if "fused_shared_experts" in node.attributes:
        attributes["fused_shared_experts"] = (
            node.attributes["fused_shared_experts"])
    domain = _graph_token_domain(node)
    attributes.update(_kernel_graph_token_attributes(domain, node.shape))
    return attributes


def _fused_moe_specs(node: WorkloadNode, hardware_id: str, hardware,
                     routing_relabelling=None):
    if node.variant != "full":
        raise ValueError("logical FUSED_MOE workload variant must be 'full'")
    selection_shape = node.shape
    domain = _graph_token_domain(node)
    if domain is None:
        shape = dict(selection_shape)
    else:
        logical_tokens, model_input_tokens, _ = domain
        if selection_shape["tokens"] != model_input_tokens:
            raise ValueError(
                "FUSED_MOE physical tokens disagree with graph model-input "
                "tokens")
        shape = dict(selection_shape)
        shape["tokens"] = logical_tokens
    prediction_attributes = _fused_moe_prediction_attributes(
        node, routing_relabelling)
    selection_attributes = dict(node.attributes)
    selection_attributes.pop("routing_provenance")
    if routing_relabelling is not None:
        global_counts, local_counts = routing_relabelling
        selection_attributes["tokens_per_expert"] = global_counts
        selection_attributes["local_tokens_per_expert"] = local_counts
    specs = build_fused_moe_physical_specs(
        hardware_id=hardware_id,
        shape=shape,
        selection_shape=selection_shape,
        dtype=node.dtype,
        prediction_attributes=prediction_attributes,
        selection_attributes=selection_attributes,
        hardware=hardware,
    )
    return _with_fused_moe_input_quant_specs(specs, hardware_id, hardware)


def _with_fused_moe_input_quant_specs(
        specs: tuple[KernelSpec, ...], hardware_id: str, hardware,
        ) -> tuple[KernelSpec, ...]:
    """Insert the activation quantiser launched before each fused-MoE GEMM."""

    result = []
    for spec in specs:
        if spec.kind is KernelKind.FUSED_MOE and spec.variant in ("w1", "w2"):
            tokens = spec.shape["tokens"]
            if spec.variant == "w1":
                element_count = tokens * spec.shape["hidden"]
            else:
                # A fused shared expert is one more routed column per token.
                kernel_topk = spec.shape["topk"] + (
                    spec.attributes["fused_shared_experts"]
                    if "fused_shared_experts" in spec.attributes else 0)
                element_count = (
                    tokens * kernel_topk * spec.shape["intermediate"])
            quant_launch = select_fused_moe_input_quant_launch(
                hardware_id, spec.launch, element_count, hardware)
            if quant_launch is not None:
                shape = {"element_count": element_count}
                result.append(KernelSpec(
                    kind=KernelKind.VECTOR,
                    hardware_id=hardware_id,
                    shape=shape,
                    dtype=spec.dtype,
                    variant=quant_launch.execution_variant,
                    launch=quant_launch,
                    attributes={
                        "group_size": PER_TOKEN_GROUP_QUANT_GROUP_SIZE,
                        **_derived_graph_token_attributes(spec, shape),
                    },
                ))
        result.append(spec)
    return tuple(result)


def _target_fused_moe_specs(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
) -> tuple[KernelSpec, ...]:
    routed_weight_dtype = node.attributes["routed_weight_dtype"]
    if routed_weight_dtype in ("bf16", "fp8"):
        return _fused_moe_specs(node, hardware_id, hardware)
    if routed_weight_dtype != "int4":
        raise ValueError(
            f"unsupported target routed weight dtype: {routed_weight_dtype!r}")
    attributes = _fused_moe_prediction_attributes(node)
    return (_selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.FUSED_MOE,
        "routed_experts", node.shape, attributes),)


def _transfer_spec(node: WorkloadNode, hardware_id: str) -> KernelSpec:
    attributes = dict(node.attributes)
    if "bytes" not in attributes:
        raise ValueError(f"{node.kind.value} requires explicit bytes")
    if node.kind in _COMMUNICATION_KINDS:
        _graph_token_domain(node)
        kind = KernelKind.COMMUNICATION
        transfer = PhysicalTransferConfig(
            algorithm=attributes["algorithm"],
            bytes=attributes["bytes"],
            direction=attributes["direction"],
            link=attributes["link"],
            topology=attributes["topology"],
            participants=tuple(attributes["participants"]),
            operation=attributes["operation"],
            bytes_semantics=attributes["bytes_semantics"],
            source_rank=attributes["source_rank"],
            destination_rank=attributes["destination_rank"],
            ring_order=tuple(attributes["ring_order"]),
            reduction=attributes["reduction"],
            backend=attributes["backend"],
            resource_id=attributes["resource_id"],
            backend_config_sha256=attributes["backend_config_sha256"],
            implementation=(
                attributes["implementation"]
                if "implementation" in attributes else None),
        )
    elif node.kind is WorkloadKind.MEMCPY:
        kind = KernelKind.MEMCPY
        transfer = PhysicalTransferConfig(
            algorithm=attributes["algorithm"], bytes=attributes["bytes"],
            direction=attributes["direction"], link=attributes["link"],
        )
    elif node.kind is WorkloadKind.KV_OFFLOAD:
        kind = KernelKind.KV_OFFLOAD
        source = attributes["source_tier"]
        destination = attributes["destination_tier"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("KV_OFFLOAD source_tier must be a non-empty string")
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError(
                "KV_OFFLOAD destination_tier must be a non-empty string")
        source_is_hbm = source.lower() == "hbm"
        destination_is_hbm = destination.lower() == "hbm"
        if source_is_hbm and not destination_is_hbm:
            non_hbm_tier = destination
        elif destination_is_hbm and not source_is_hbm:
            non_hbm_tier = source
        else:
            raise ValueError(
                "KV_OFFLOAD requires exactly one HBM endpoint, got "
                f"{source!r} -> {destination!r}")
        transfer = PhysicalTransferConfig(
            algorithm=attributes["algorithm"], bytes=attributes["bytes"],
            direction=attributes["direction"], link=attributes["link"],
            tier=non_hbm_tier,
        )
    else:
        raise NotImplementedError(f"no transfer lowering for {node.kind.value}")
    return KernelSpec(
        kind=kind, hardware_id=hardware_id, transfer=transfer,
    )


def _dense_tp_producer_output_bytes(node: WorkloadNode) -> int:
    if node.dtype != "bf16":
        raise ValueError(
            "validated dense TP collective producer dtype must be bf16")
    if node.kind is WorkloadKind.GATHER:
        if node.variant != "embedding":
            raise ValueError(
                "dense TP gather collective producer must use the embedding "
                "variant")
        required_shape = {"elements", "rows", "hidden"}
        if set(node.shape) != required_shape:
            raise ValueError(
                "dense TP embedding producer shape fields disagree with the "
                "collective payload contract")
        elements = node.shape["rows"] * node.shape["hidden"]
        if node.shape["elements"] != elements:
            raise ValueError(
                "dense TP embedding elements disagree with rows and hidden")
    elif node.kind is WorkloadKind.LINEAR:
        if node.variant != "gemm":
            raise ValueError(
                "dense TP linear collective producer must use the GEMM "
                "variant")
        required_shape = {"b", "m", "n", "k"}
        if set(node.shape) != required_shape:
            raise ValueError(
                "dense TP GEMM producer shape fields disagree with the "
                "collective payload contract")
        elements = node.shape["b"] * node.shape["m"] * node.shape["n"]
    elif node.kind is WorkloadKind.LM_HEAD:
        if node.variant not in ("gemm", "gemv"):
            raise ValueError(
                "dense TP LM-head collective producer must use a GEMM/GEMV "
                "variant")
        required_shape = {"b", "m", "n", "k"}
        if set(node.shape) != required_shape:
            raise ValueError(
                "dense TP LM-head producer shape fields disagree with the "
                "collective payload contract")
        elements = node.shape["b"] * node.shape["m"] * node.shape["n"]
    else:
        raise ValueError(
            "dense TP collective predecessor must be an embedding, linear, "
            f"or LM-head node, got {node.kind.value!r}")
    if elements <= 0:
        raise ValueError(
            "dense TP collective producer output must contain elements")
    return elements * 2


def _validate_collective_node(
    node: WorkloadNode,
    pool,
    graph: WorkloadGraph,
) -> None:
    attributes = node.attributes
    operation = attributes["operation"]
    collective = pool.collective_for(operation)
    logical_rank = pool.logical_rank_for_physical_rank(node.rank)
    participants = pool.physical_rank_group("tensor", logical_rank)
    if operation == "all_reduce":
        expected_reduction = "sum"
        expected_ring_order = _collective_ring_order(
            collective.algorithm, participants)
    elif operation == "all_gather":
        expected_reduction = None
        expected_ring_order = _collective_ring_order(
            collective.algorithm, participants)
    elif operation == "all_to_all":
        expected_reduction = None
        expected_ring_order = ()
    else:
        raise ValueError(
            f"unknown collective operation: {operation!r}")
    if node.rank != participants[0]:
        raise ValueError(
            "collective workload leader must be logical tensor rank zero")
    if tuple(attributes["participants"]) != participants:
        raise ValueError(
            "collective workload participants disagree with the physical "
            "tensor rank group")
    expected = {
        "algorithm": collective.algorithm,
        "backend": collective.backend,
        "backend_config_sha256": collective.backend_config_sha256,
        "bytes_semantics": "per_rank_input",
        "direction": "collective",
        "link": collective.link,
        "operation": collective.operation,
        "participants": participants,
        "source_rank": None,
        "destination_rank": None,
        "ring_order": expected_ring_order,
        "reduction": expected_reduction,
        "topology": collective.topology,
        "resource_id": collective.resource_id,
    }
    mismatched = {
        name: (attributes[name], value)
        for name, value in expected.items()
        if attributes[name] != value
    }
    if mismatched:
        raise ValueError(
            "collective workload disagrees with deployment config: "
            f"{mismatched!r}")
    parallelism = pool.parallel_plan.config
    dense_tp_only = (
        parallelism.tensor_parallel_size == 2
        and parallelism.expert_parallel_size == 1
        and parallelism.pipeline_parallel_size == 1
        and parallelism.sequence_parallel_size == 1
        and parallelism.context_parallel_size == 1
        and parallelism.data_parallel_size == 1
    )
    if not dense_tp_only:
        return
    incoming_edges = tuple(
        edge for edge in graph.edges if edge.destination == node.node_id)
    if len(incoming_edges) != len(participants):
        raise ValueError(
            "dense TP collective must have one producer per participant")
    if any(edge.kind != "barrier" for edge in incoming_edges):
        raise ValueError(
            "dense TP collective producer edges must be barriers")
    predecessor_ids = tuple(edge.source for edge in incoming_edges)
    producers = tuple(graph.node(node_id) for node_id in predecessor_ids)
    if (any(producer.pool_name != pool.name for producer in producers)
            or {producer.rank for producer in producers}
            != set(participants)):
        raise ValueError(
            "dense TP collective producers disagree with its pool/rank group")
    producer_kinds = {producer.kind for producer in producers}
    if len(producer_kinds) != 1:
        raise ValueError(
            "dense TP collective producers must have one consistent kind")
    producer_contracts = {
        (
            tuple(sorted(producer.shape.items())),
            producer.dtype,
            producer.variant,
        )
        for producer in producers
    }
    if len(producer_contracts) != 1:
        raise ValueError(
            "dense TP collective producers must have the same shape, dtype, "
            "and variant")
    producer_bytes = tuple(
        _dense_tp_producer_output_bytes(producer)
        for producer in producers
    )
    if len(set(producer_bytes)) != 1:
        raise ValueError(
            "dense TP collective producers have inconsistent output bytes")
    expected_bytes = producer_bytes[0]
    if attributes["bytes"] != expected_bytes:
        raise ValueError(
            "dense TP collective payload bytes disagree with producer shape "
            f"and dtype: expected={expected_bytes}, "
            f"actual={attributes['bytes']}")
    producer_kind = producers[0].kind
    if producer_kind in (WorkloadKind.GATHER, WorkloadKind.LINEAR):
        expected_operation = "all_reduce"
    elif producer_kind is WorkloadKind.LM_HEAD:
        expected_operation = "all_gather"
    else:
        raise RuntimeError(
            "validated dense TP producer kind became unreachable: "
            f"{producer_kind.value!r}")
    if operation != expected_operation:
        raise ValueError(
            "dense TP collective operation disagrees with producer kind: "
            f"expected={expected_operation!r}, actual={operation!r}")


def _validate_pipeline_p2p_node(node: WorkloadNode, pool) -> None:
    attributes = node.attributes
    required = {
        "algorithm",
        "bytes",
        "bytes_semantics",
        "backend",
        "backend_config_sha256",
        "direction",
        "link",
        "topology",
        "participants",
        "operation",
        "source_rank",
        "destination_rank",
        "ring_order",
        "reduction",
        "resource_id",
        "source_stage",
        "destination_stage",
        "payload_tensor",
    }
    graph_domain_fields = set(_GRAPH_TOKEN_DOMAIN_FIELDS)
    present_graph_domain_fields = graph_domain_fields & set(attributes)
    if present_graph_domain_fields:
        if present_graph_domain_fields != graph_domain_fields:
            raise ValueError(
                "pipeline P2P graph token domains must be complete")
        required |= graph_domain_fields
    if set(attributes) != required:
        raise ValueError(
            "pipeline P2P workload attributes must match the canonical "
            f"contract: missing={sorted(required - set(attributes))!r}, "
            f"extra={sorted(set(attributes) - required)!r}")
    source_stage = attributes["source_stage"]
    destination_stage = attributes["destination_stage"]
    config = pool.pipeline_p2p_for(source_stage, destination_stage)
    source_ranks = pool.physical_ranks_for_pipeline_stage(source_stage)
    destination_ranks = pool.physical_ranks_for_pipeline_stage(
        destination_stage)
    if len(source_ranks) == 1 and len(destination_ranks) == 1:
        source_rank = source_ranks[0]
        destination_rank = destination_ranks[0]
        expected_payload_rank = 2
    elif (len(source_ranks) == len(destination_ranks)
          and len(source_ranks) in (2, 8)):
        parallelism = pool.parallel_plan.config
        tp2_scope = (
            len(source_ranks) == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.pipeline_parallel_size == 2
            and parallelism.expert_parallel_size == 1
        )
        target_scope = (
            len(source_ranks) == 8
            and parallelism.tensor_parallel_size == 8
            and parallelism.pipeline_parallel_size in (2, 4)
            and parallelism.enable_expert_parallel
            and parallelism.expert_parallel_size == 8
        )
        if (not (tp2_scope or target_scope)
                or not parallelism.is_rank_local):
            raise NotImplementedError(
                "multi-lane pipeline P2P lowering requires either exact "
                "TP2/PP2 or target TP=EP=8 with PP2/PP4")
        source_rank = attributes["source_rank"]
        if source_rank not in source_ranks:
            raise ValueError(
                "pipeline P2P source rank is outside the source stage")
        lane_index = source_ranks.index(source_rank)
        destination_rank = destination_ranks[lane_index]
        expected_payload_rank = 1
    else:
        raise NotImplementedError(
            "pipeline P2P lowering requires either one rank per stage or "
            "a validated two/eight-lane TP pipeline slice")
    if node.rank != source_rank:
        raise ValueError(
            "pipeline P2P workload rank must be its physical source rank")
    expected = {
        "algorithm": config.algorithm,
        "bytes_semantics": "payload",
        "backend": config.backend,
        "backend_config_sha256": config.backend_config_sha256,
        "direction": "point_to_point",
        "link": config.link,
        "topology": config.topology,
        "participants": (source_rank, destination_rank),
        "operation": "send_recv",
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "ring_order": (),
        "reduction": None,
        "resource_id": config.resource_id,
    }
    mismatched = {
        name: (attributes[name], value)
        for name, value in expected.items()
        if attributes[name] != value
    }
    if mismatched:
        raise ValueError(
            "pipeline P2P workload disagrees with deployment config: "
            f"{mismatched!r}")
    payload = attributes["payload_tensor"]
    if not isinstance(payload, Mapping):
        raise TypeError("pipeline P2P payload_tensor must be a mapping")
    payload_required = {"name", "shape", "dtype", "bytes"}
    if set(payload) != payload_required:
        raise ValueError(
            "pipeline P2P payload_tensor fields must match the canonical "
            f"contract: missing={sorted(payload_required - set(payload))!r}, "
            f"extra={sorted(set(payload) - payload_required)!r}")
    if payload["name"] not in ("hidden_states", "residual"):
        raise ValueError(
            "pipeline P2P payload_tensor name must be hidden_states or "
            "residual")
    shape = payload["shape"]
    if (not isinstance(shape, (list, tuple))
            or len(shape) != expected_payload_rank
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in shape)):
        raise TypeError(
            "pipeline P2P payload_tensor shape rank disagrees with the "
            "pipeline/TP contract")
    if any(value <= 0 for value in shape):
        raise ValueError(
            "pipeline P2P payload_tensor dimensions must be positive")
    if payload["dtype"] != "bf16":
        raise ValueError(
            "validated pipeline P2P payload_tensor dtype must be bf16")
    expected_elements = 1
    for dimension in shape:
        expected_elements *= dimension
    expected_bytes = expected_elements * 2
    if (payload["bytes"] != expected_bytes
            or attributes["bytes"] != expected_bytes):
        raise ValueError(
            "pipeline P2P payload bytes disagree with shape and dtype")


def _validate_pd_kv_transfer_node(
    node: WorkloadNode,
    deployment: DeploymentPlan,
) -> None:
    """Bind one request-scoped KV transfer to its configured P-to-D route."""

    if deployment.mode not in ("pd_disagg", "pd_af_disagg"):
        raise ValueError("PD KV transfer lowering requires a PD deployment")
    if node.shape:
        raise ValueError("PD KV transfer workload shape must be empty")
    if node.dtype is not None or node.variant is not None:
        raise ValueError(
            "PD KV transfer workload dtype and variant must both be null")
    attributes = node.attributes
    required = {
        "algorithm",
        "bytes",
        "bytes_semantics",
        "backend",
        "backend_config_sha256",
        "direction",
        "link",
        "topology",
        "participants",
        "operation",
        "source_rank",
        "destination_rank",
        "ring_order",
        "reduction",
        "resource_id",
        "destination_pool",
        "rank_mapping",
        "source_logical_rank",
        "destination_logical_rank",
        "source_tensor_parallel_size",
        "destination_tensor_parallel_size",
        "kv_head_start",
        "kv_head_end",
        "num_key_value_heads",
    }
    if set(attributes) != required:
        raise ValueError(
            "PD KV transfer attributes must match the canonical contract: "
            f"missing={sorted(required - set(attributes))!r}, "
            f"extra={sorted(set(attributes) - required)!r}")
    destination_pool_name = attributes["destination_pool"]
    if destination_pool_name not in deployment.worker_pools:
        raise ValueError(
            "PD KV transfer references an unknown destination pool: "
            f"{destination_pool_name!r}")
    source_pool = deployment.worker_pools[node.pool_name]
    destination_pool = deployment.worker_pools[destination_pool_name]
    if worker_role_phase(source_pool.role) != "prefill":
        raise ValueError("PD KV transfer source must have a prefill role")
    if worker_role_phase(destination_pool.role) != "decode":
        raise ValueError("PD KV transfer destination must have a decode role")
    if (worker_role_operator_group(source_pool.role) == "ffn"
            or worker_role_operator_group(destination_pool.role) == "ffn"):
        raise ValueError(
            "PD KV transfer endpoints must be unsplit or Attention-owner "
            "pools, not FFN pools")
    rank_pairs = pd_kv_transfer_rank_pairs(
        source_pool,
        destination_pool,
        attributes["num_key_value_heads"],
    )
    routes = tuple(
        route for route in deployment.routes
        if route.payload == "kv"
        and route.source_pool == source_pool.name
        and route.destination_pool == destination_pool.name
    )
    if len(routes) != 1:
        raise ValueError(
            "PD KV transfer requires exactly one configured route: "
            f"{source_pool.name!r}->{destination_pool.name!r}")
    route = routes[0]
    matching_rank_pairs = tuple(
        rank_pair for rank_pair in rank_pairs
        if (rank_pair.source_logical_rank
            == attributes["source_logical_rank"]
            and rank_pair.destination_logical_rank
            == attributes["destination_logical_rank"]
            and rank_pair.source_rank == node.rank
            and rank_pair.source_rank == attributes["source_rank"]
            and rank_pair.destination_rank == attributes["destination_rank"]
            and rank_pair.kv_head_start == attributes["kv_head_start"]
            and rank_pair.kv_head_end == attributes["kv_head_end"])
    )
    if len(matching_rank_pairs) != 1:
        raise ValueError(
            "PD KV transfer workload disagrees with deployment config: "
            "not one canonical KV-head ownership edge")
    rank_pair = matching_rank_pairs[0]
    source_rank = rank_pair.source_rank
    destination_rank = rank_pair.destination_rank
    transfer_bytes = attributes["bytes"]
    if (isinstance(transfer_bytes, bool)
            or not isinstance(transfer_bytes, int)
            or transfer_bytes <= 0):
        raise TypeError("PD KV transfer bytes must be a positive integer")
    expected = {
        "algorithm": route.algorithm,
        "bytes_semantics": "payload",
        "backend": route.backend,
        "backend_config_sha256": route.backend_config_sha256,
        "direction": "point_to_point",
        "link": route.link,
        "topology": route.topology,
        "participants": (source_rank, destination_rank),
        "operation": "send_recv",
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "ring_order": (),
        "reduction": None,
        "resource_id": route.resource_id,
        "destination_pool": destination_pool.name,
        "rank_mapping": "kv_head_overlap",
        "source_logical_rank": rank_pair.source_logical_rank,
        "destination_logical_rank": rank_pair.destination_logical_rank,
        "source_tensor_parallel_size": (
            source_pool.parallel_plan.config.tensor_parallel_size),
        "destination_tensor_parallel_size": (
            destination_pool.parallel_plan.config.tensor_parallel_size),
        "kv_head_start": rank_pair.kv_head_start,
        "kv_head_end": rank_pair.kv_head_end,
        "num_key_value_heads": attributes["num_key_value_heads"],
    }
    mismatched = {
        name: (attributes[name], value)
        for name, value in expected.items()
        if attributes[name] != value
    }
    if mismatched:
        raise ValueError(
            "PD KV transfer workload disagrees with deployment config: "
            f"{mismatched!r}")


def _validate_af_activation_transfer_node(
    node: WorkloadNode,
    deployment: DeploymentPlan,
    graph: WorkloadGraph,
) -> None:
    """Bind one AF transfer to its configured route and physical TP lane."""

    attributes = node.attributes
    required = {
        "algorithm",
        "bytes",
        "bytes_semantics",
        "backend",
        "backend_config_sha256",
        "direction",
        "link",
        "topology",
        "participants",
        "operation",
        "source_rank",
        "destination_rank",
        "ring_order",
        "reduction",
        "resource_id",
        "destination_pool",
        "payload_tensor",
    }
    graph_domain_fields = set(_GRAPH_TOKEN_DOMAIN_FIELDS)
    present_graph_domain_fields = graph_domain_fields & set(attributes)
    if present_graph_domain_fields:
        if present_graph_domain_fields != graph_domain_fields:
            raise ValueError(
                "AF activation graph token domains must be complete")
        required |= graph_domain_fields
    if "residual_anchor" in attributes:
        # Only the observation-suppression marker is legal, and only on a
        # residual payload transfer (the builder stamps it nowhere else).
        if (attributes["residual_anchor"]
                != RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED
                or attributes["payload_tensor"]["name"] != "residual"):
            raise ValueError(
                "AF activation transfer carries an unknown residual_anchor "
                f"contract: {attributes['residual_anchor']!r}")
        required.add("residual_anchor")
    if set(attributes) != required:
        raise ValueError(
            "AF activation transfer attributes must match the canonical "
            f"contract: missing={sorted(required - set(attributes))!r}, "
            f"extra={sorted(set(attributes) - required)!r}")
    destination_pool_name = attributes["destination_pool"]
    if destination_pool_name not in deployment.worker_pools:
        raise ValueError(
            "AF activation transfer references an unknown destination pool: "
            f"{destination_pool_name!r}")
    source_pool = deployment.worker_pools[node.pool_name]
    destination_pool = deployment.worker_pools[destination_pool_name]
    routes = tuple(
        route for route in deployment.routes
        if route.payload == "activation"
        and route.source_pool == source_pool.name
        and route.destination_pool == destination_pool.name
    )
    if len(routes) != 1:
        raise ValueError(
            "AF activation transfer requires exactly one configured route: "
            f"{source_pool.name!r}->{destination_pool.name!r}")
    route = routes[0]
    source_parallelism = source_pool.parallel_plan.config
    destination_parallelism = destination_pool.parallel_plan.config
    if len(source_pool.ranks) == 1 and len(destination_pool.ranks) == 1:
        source_rank = source_pool.ranks[0]
        destination_rank = destination_pool.ranks[0]
        tp2_lane = False
    elif len(source_pool.ranks) == 2 and len(destination_pool.ranks) == 2:
        common_tp2 = all(
            parallelism.worker_count == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.pipeline_parallel_size == 1
            and parallelism.sequence_parallel_size == 1
            and parallelism.context_parallel_size == 1
            and parallelism.data_parallel_size == 1
            for parallelism in (
                source_parallelism, destination_parallelism)
        )
        operator_parallelism = {
            worker_role_operator_group(source_pool.role): source_parallelism,
            worker_role_operator_group(
                destination_pool.role): destination_parallelism,
        }
        dense_tp2 = all(
            parallelism.expert_parallel_size == 1
            for parallelism in (
                source_parallelism, destination_parallelism)
        )
        af_ep2 = (
            deployment.mode == "af_disagg"
            and set(operator_parallelism) == {"attention", "ffn"}
            and operator_parallelism["attention"].expert_parallel_size == 1
            and operator_parallelism["ffn"].expert_parallel_size == 2
        )
        exact_tp2 = common_tp2 and (dense_tp2 or af_ep2)
        if not exact_tp2:
            raise NotImplementedError(
                "two-rank AF activation lowering requires matching TP=2 "
                "lanes in the validated dense or FFN-EP2 scope")
        source_rank = attributes["source_rank"]
        if source_rank not in source_pool.ranks:
            raise ValueError(
                "AF activation source rank is outside the source pool")
        lane_index = source_pool.ranks.index(source_rank)
        destination_rank = destination_pool.ranks[lane_index]
        tp2_lane = True
    else:
        raise NotImplementedError(
            "AF activation lowering requires one rank per pool or matching "
            "two-lane TP=2 pools")
    if node.rank != source_rank:
        raise ValueError(
            "AF activation workload rank must be its physical source rank")
    expected = {
        "algorithm": route.algorithm,
        "bytes_semantics": "payload",
        "backend": route.backend,
        "backend_config_sha256": route.backend_config_sha256,
        "direction": "point_to_point",
        "link": route.link,
        "topology": route.topology,
        "participants": (source_rank, destination_rank),
        "operation": "send_recv",
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "ring_order": (),
        "reduction": None,
        "resource_id": route.resource_id,
        "destination_pool": destination_pool.name,
    }
    mismatched = {
        name: (attributes[name], value)
        for name, value in expected.items()
        if attributes[name] != value
    }
    if mismatched:
        raise ValueError(
            "AF activation workload disagrees with deployment route/lane: "
            f"{mismatched!r}")
    transfer_bytes = attributes["bytes"]
    if (isinstance(transfer_bytes, bool)
            or not isinstance(transfer_bytes, int)
            or transfer_bytes <= 0):
        raise TypeError("AF activation payload bytes must be a positive integer")
    payload = attributes["payload_tensor"]
    if not isinstance(payload, Mapping):
        raise TypeError("AF activation payload_tensor must be a mapping")
    payload_required = {"name", "shape", "dtype", "bytes"}
    if set(payload) != payload_required:
        raise ValueError(
            "AF activation payload_tensor fields must match the canonical "
            f"contract: missing={sorted(payload_required - set(payload))!r}, "
            f"extra={sorted(set(payload) - payload_required)!r}")
    tensor_name = payload["name"]
    if tensor_name not in ("hidden_states", "residual"):
        raise ValueError(
            "AF activation payload_tensor name must be hidden_states or "
            "residual")
    payload_shape = payload["shape"]
    if (not isinstance(payload_shape, (list, tuple))
            or len(payload_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in payload_shape)):
        raise TypeError(
            "AF activation payload_tensor shape must be a two-dimensional "
            "integer sequence")
    if any(value <= 0 for value in payload_shape):
        raise ValueError(
            "AF activation payload_tensor dimensions must be positive")
    payload_dtype = payload["dtype"]
    if payload_dtype not in ("bf16", "fp16"):
        raise ValueError(
            "AF activation payload_tensor dtype must be BF16 or FP16")
    expected_payload_bytes = payload_shape[0] * payload_shape[1] * 2
    payload_bytes = payload["bytes"]
    if (isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes <= 0):
        raise TypeError(
            "AF activation payload_tensor bytes must be a positive integer")
    if (payload_bytes != expected_payload_bytes
            or transfer_bytes != expected_payload_bytes):
        raise ValueError(
            "AF activation payload bytes disagree with shape and dtype")
    incoming_edges = tuple(
        edge for edge in graph.edges if edge.destination == node.node_id)
    if len(incoming_edges) != 1 or incoming_edges[0].kind != "ready":
        raise ValueError(
            "AF activation transfer requires exactly one ready producer edge")
    producer = graph.node(incoming_edges[0].source)
    if producer.pool_name != source_pool.name:
        raise ValueError(
            "AF activation producer must belong to the source pool")
    if tensor_name == "residual":
        if producer.kind is WorkloadKind.COLLECTIVE:
            if (producer.attributes["operation"] != "all_reduce"
                    or tuple(producer.attributes["participants"])
                    != source_pool.ranks
                    or source_rank not in producer.attributes["participants"]):
                raise ValueError(
                    "AF initial residual collective must be the canonical "
                    "source-pool all-reduce")
            producer_bytes = producer.attributes["bytes"]
        elif producer.kind in (
                WorkloadKind.NORM,
                WorkloadKind.GATHER,
                WorkloadKind.LINEAR,
        ):
            if (producer.rank != source_rank
                    or producer.dtype != payload_dtype):
                raise ValueError(
                    "AF residual producer rank or dtype disagrees with the "
                    "source lane")
            if producer.kind is WorkloadKind.NORM:
                if (producer.variant != "add_rmsnorm"
                        or set(producer.shape) != {"rows", "hidden"}):
                    raise ValueError(
                        "AF carried residual requires a canonical "
                        "add_rmsnorm producer")
                producer_shape = (
                    producer.shape["rows"], producer.shape["hidden"])
            elif producer.kind is WorkloadKind.GATHER:
                if (producer.variant != "embedding"
                        or set(producer.shape)
                        != {"elements", "rows", "hidden"}):
                    raise ValueError(
                        "AF initial residual requires a canonical embedding "
                        "producer")
                producer_shape = (
                    producer.shape["rows"], producer.shape["hidden"])
            elif producer.kind is WorkloadKind.LINEAR:
                if (producer.variant != "gemm"
                        or set(producer.shape) != {"b", "m", "n", "k"}):
                    raise ValueError(
                        "AF initial residual requires a canonical linear "
                        "producer")
                producer_shape = (
                    producer.shape["b"] * producer.shape["m"],
                    producer.shape["n"],
                )
            else:
                raise ValueError(
                    "AF residual payload has an unsupported producer kind")
            if tuple(payload_shape) != producer_shape:
                raise ValueError(
                    "AF residual payload shape disagrees with its producer")
            producer_bytes = producer_shape[0] * producer_shape[1] * 2
        else:
            raise ValueError(
                "AF residual payload has an unsupported producer kind")
    elif tp2_lane:
        if (producer.kind is not WorkloadKind.COLLECTIVE
                or producer.attributes["operation"] != "all_reduce"
                or tuple(producer.attributes["participants"])
                != source_pool.ranks):
            raise ValueError(
                "AF TP=2 hidden_states lanes must follow the source-pool "
                "all-reduce frontier")
        producer_bytes = producer.attributes["bytes"]
    elif tensor_name == "hidden_states":
        if producer.rank != source_rank:
            raise ValueError(
                "AF activation producer rank disagrees with the source rank")
        if producer.dtype != payload_dtype:
            raise ValueError(
                "single-rank AF hidden_states producer dtype disagrees with "
                "the payload")
        if producer.kind is WorkloadKind.LINEAR:
            if (producer.variant != "gemm"
                    or set(producer.shape) != {"b", "m", "n", "k"}):
                raise ValueError(
                    "single-rank AF linear producer requires canonical GEMM "
                    "dimensions")
            producer_elements = (
                producer.shape["b"]
                * producer.shape["m"]
                * producer.shape["n"]
            )
        elif producer.kind is WorkloadKind.FUSED_MOE:
            if (producer.variant != "full"
                    or set(producer.shape) != {
                        "tokens", "experts", "topk", "hidden",
                        "intermediate",
                    }):
                raise ValueError(
                    "single-rank AF fused-MoE producer requires canonical "
                    "dimensions")
            producer_elements = (
                producer.shape["tokens"] * producer.shape["hidden"])
        elif producer.kind is WorkloadKind.ELEMENTWISE:
            if (producer.variant != "add"
                    or set(producer.shape) != {"elements"}):
                raise ValueError(
                    "single-rank AF elementwise producer requires a canonical "
                    "add shape")
            producer_elements = producer.shape["elements"]
        else:
            raise ValueError(
                "single-rank AF hidden_states must follow a canonical linear, "
                "fused-MoE, or elementwise-add producer")
        producer_bytes = producer_elements * 2
    else:
        raise AssertionError("validated AF payload tensor name vanished")
    if transfer_bytes != producer_bytes:
        raise ValueError(
            "AF activation payload must contain one full source tensor: "
            f"name={tensor_name!r}, expected={producer_bytes}, "
            f"actual={transfer_bytes}")
    outgoing_edges = tuple(
        edge for edge in graph.edges if edge.source == node.node_id)
    if len(outgoing_edges) != 1 or outgoing_edges[0].kind != "data":
        raise ValueError(
            "AF activation transfer requires exactly one destination data edge")
    consumer = graph.node(outgoing_edges[0].destination)
    if (consumer.pool_name != destination_pool.name
            or consumer.rank != destination_rank):
        raise ValueError(
            "AF activation consumer disagrees with the destination pool/lane")
    if (consumer.kind is WorkloadKind.NORM
            and consumer.variant == "add_rmsnorm"
            and set(consumer.shape) == {"rows", "hidden"}):
        consumer_elements = (
            consumer.shape["rows"] * consumer.shape["hidden"])
        if tuple(payload_shape) != (
                consumer.shape["rows"], consumer.shape["hidden"]):
            raise ValueError(
                "AF activation payload shape disagrees with its "
                "add_rmsnorm consumer")
    elif (consumer.kind is WorkloadKind.ELEMENTWISE
          and consumer.variant == "add"
          and set(consumer.shape) == {"elements"}):
        consumer_elements = consumer.shape["elements"]
    else:
        raise ValueError(
            "AF hidden_states/residual payload pair must feed a canonical "
            "add_rmsnorm or elementwise-add consumer")
    if consumer_elements != payload_shape[0] * payload_shape[1]:
        raise ValueError(
            "AF activation payload shape disagrees with its destination "
            "consumer")


def _validate_af_activation_boundaries(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
) -> None:
    """Require one ordered hidden_states/residual pair on every AF lane."""

    af_nodes = tuple(
        node for node in graph.nodes
        if node.kind is WorkloadKind.AF_ACTIVATION_TRANSFER
    )
    if deployment.mode not in ("af_disagg", "pd_af_disagg"):
        if af_nodes:
            raise ValueError(
                "AF activation transfers require an AF deployment mode")
        return
    if not af_nodes:
        if all(
                node.kind is WorkloadKind.PD_KV_TRANSFER
                for node in graph.nodes):
            return
        raise ValueError(
            "AF deployment workload must contain activation transfers")

    for edge in graph.edges:
        source = graph.node(edge.source)
        destination = graph.node(edge.destination)
        if (source.pool_name != destination.pool_name
                and source.kind is not WorkloadKind.AF_ACTIVATION_TRANSFER):
            raise ValueError(
                "AF cross-pool dependencies must pass through an activation "
                "transfer node")

    for node in graph.nodes:
        requires_pair = False
        if (node.kind is WorkloadKind.NORM
                and node.variant == "add_rmsnorm"):
            requires_pair = True
        elif (node.kind is WorkloadKind.ELEMENTWISE
              and node.variant == "add"):
            pool_role = deployment.worker_pools[node.pool_name].role
            if pool_role == "attention" or pool_role.endswith("-attention"):
                requires_pair = True
        if not requires_pair:
            continue
        incoming_edges = tuple(
            edge for edge in graph.edges if edge.destination == node.node_id)
        if (len(incoming_edges) != 2
                or any(
                    graph.node(edge.source).kind
                    is not WorkloadKind.AF_ACTIVATION_TRANSFER
                    for edge in incoming_edges)):
            raise ValueError(
                "AF residual consumer requires exactly one hidden_states/"
                "residual activation pair")

    ancestor_cache = {}

    def ancestor_ids(node_id):
        if node_id in ancestor_cache:
            return ancestor_cache[node_id]
        ancestors = set()
        pending = list(graph.predecessors(node_id))
        while pending:
            predecessor_id = pending.pop()
            if predecessor_id in ancestors:
                continue
            ancestors.add(predecessor_id)
            pending.extend(graph.predecessors(predecessor_id))
        ancestor_cache[node_id] = frozenset(ancestors)
        return ancestor_cache[node_id]

    def establishes_residual(candidate):
        if candidate.kind is not WorkloadKind.NORM:
            return False
        if candidate.variant == "add_rmsnorm":
            return True
        if candidate.variant != "rmsnorm":
            return False
        successors = tuple(
            graph.node(edge.destination)
            for edge in graph.edges
            if edge.source == candidate.node_id
        )
        return (
            len(successors) == 1
            and successors[0].kind is WorkloadKind.LINEAR
        )

    pairs = {}
    for node in af_nodes:
        outgoing_edge, = tuple(
            edge for edge in graph.edges if edge.source == node.node_id)
        key = (
            node.pool_name,
            node.attributes["destination_pool"],
            node.attributes["source_rank"],
            node.attributes["destination_rank"],
            outgoing_edge.destination,
        )
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(node)

    boundaries = {}
    for pair_key, nodes in pairs.items():
        source_name, destination_name, source_rank, destination_rank, _ = (
            pair_key)
        names = tuple(
            node.attributes["payload_tensor"]["name"] for node in nodes)
        if len(nodes) != 2 or names != ("hidden_states", "residual"):
            raise ValueError(
                "AF boundary lane must contain one ordered hidden_states/"
                "residual payload pair: "
                f"lane={pair_key!r}, payloads={names!r}")
        hidden_node, residual_node = nodes
        hidden_payload = hidden_node.attributes["payload_tensor"]
        residual_payload = residual_node.attributes["payload_tensor"]
        if any(
                hidden_payload[field] != residual_payload[field]
                for field in ("shape", "dtype", "bytes")):
            raise ValueError(
                "AF hidden_states/residual payload pair must have identical "
                "shape, dtype, and byte count")
        hidden_incoming, = tuple(
            edge for edge in graph.edges
            if edge.destination == hidden_node.node_id)
        residual_incoming, = tuple(
            edge for edge in graph.edges
            if edge.destination == residual_node.node_id)
        payload_shape = tuple(hidden_payload["shape"])
        payload_dtype = hidden_payload["dtype"]
        current_residual_norms = tuple(
                candidate for candidate in graph.nodes
                if (candidate.node_id in ancestor_ids(hidden_incoming.source)
                    and candidate.pool_name == source_name
                    and candidate.rank == source_rank
                    and establishes_residual(candidate)
                    and candidate.dtype == payload_dtype
                and set(candidate.shape) == {"rows", "hidden"}
                and (candidate.shape["rows"], candidate.shape["hidden"])
                == payload_shape)
        )
        if not current_residual_norms:
            raise ValueError(
                "AF hidden_states producer has no current residual norm")
        latest_residual_norms = tuple(
            candidate for candidate in current_residual_norms
            if all(
                other.node_id == candidate.node_id
                or other.node_id in ancestor_ids(candidate.node_id)
                for other in current_residual_norms
            )
        )
        if len(latest_residual_norms) != 1:
            raise ValueError(
                "AF hidden_states producer must have one unique latest "
                "residual norm in the dependency DAG: "
                f"candidates={tuple(candidate.node_id for candidate in current_residual_norms)!r}, "
                f"latest={tuple(candidate.node_id for candidate in latest_residual_norms)!r}")
        residual_norm, = latest_residual_norms
        if residual_norm.variant == "add_rmsnorm":
            expected_residual_source = residual_norm.node_id
        elif residual_norm.variant == "rmsnorm":
            residual_predecessors = graph.predecessors(
                residual_norm.node_id)
            if len(residual_predecessors) == 1:
                expected_residual_source = residual_predecessors[0]
            elif not residual_predecessors:
                # The step-head input embedding was suppressed by the
                # versioned emission observation, so the initial residual
                # norm has no in-graph producer and the residual transfer
                # must carry the builder's observation marker and anchor on
                # the hidden-state producer of the same boundary.
                if ("residual_anchor" not in residual_node.attributes
                        or residual_node.attributes["residual_anchor"]
                        != RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED
                        or residual_incoming.source != hidden_incoming.source):
                    raise ValueError(
                        "AF initial residual norm lost its input producer "
                        "without the observation-suppression anchor")
                expected_residual_source = residual_incoming.source
            else:
                raise ValueError(
                    "AF initial residual norm requires exactly one input "
                    "producer")
        else:
            raise AssertionError(
                "validated residual RMSNorm variant vanished")
        if residual_incoming.source != expected_residual_source:
            raise ValueError(
                "AF residual transfer must use the current boundary residual "
                "producer: "
                f"expected={expected_residual_source!r}, "
                f"actual={residual_incoming.source!r}")
        boundary_key = (
            hidden_incoming.source,
            source_name,
            destination_name,
        )
        if boundary_key not in boundaries:
            boundaries[boundary_key] = []
        boundaries[boundary_key].append((
            source_rank,
            destination_rank,
            hidden_node.attributes["participants"],
        ))

    for (producer_id, source_name, destination_name), lanes in (
            boundaries.items()):
        source_pool = deployment.worker_pools[source_name]
        destination_pool = deployment.worker_pools[destination_name]
        expected_lanes = tuple(zip(
            source_pool.ranks,
            destination_pool.ranks,
            strict=True,
        ))
        actual_lanes = tuple(
            (source_rank, destination_rank)
            for source_rank, destination_rank, _ in lanes)
        actual_participants = tuple(
            tuple(participants) for _, _, participants in lanes)
        if (len(lanes) != len(expected_lanes)
                or len(set(actual_lanes)) != len(expected_lanes)
                or set(actual_lanes) != set(expected_lanes)
                or set(actual_participants) != set(expected_lanes)):
            raise ValueError(
                "AF boundary must cover each logical lane exactly once: "
                f"producer={producer_id!r}, source={source_name!r}, "
                f"destination={destination_name!r}, "
                f"expected_lanes={expected_lanes!r}, "
                f"actual_lanes={actual_lanes!r}")


def _validate_tp1_pipeline_graph(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
) -> None:
    """Validate every tensor and dependency at a TP=1 PP boundary."""

    target_contract = _target_model_semantics_contract(graph)
    for pool in deployment.worker_pools.values():
        parallelism = pool.parallel_plan.config
        pipeline_size = parallelism.pipeline_parallel_size
        if parallelism.tensor_parallel_size != 1 or pipeline_size == 1:
            continue
        if (pipeline_size < 2
                or deployment.mode != "colocated"
                or len(deployment.worker_pools) != 1
                or parallelism.worker_count != pipeline_size
                or parallelism.expert_parallel_size != 1
                or not parallelism.is_rank_local):
            raise NotImplementedError(
                "TP=1 pipeline lowering supports one colocated uniform PP>=2 "
                "pool with EP=SP=CP=DP=1")
        p2p_nodes = tuple(
            node for node in graph.nodes
            if node.pool_name == pool.name and node.kind is WorkloadKind.P2P
        )
        expected_count = 2 * (pipeline_size - 1)
        if len(p2p_nodes) != expected_count:
            raise ValueError(
                "TP=1 pipeline graph requires hidden_states and residual at "
                f"every boundary: expected={expected_count}, "
                f"actual={len(p2p_nodes)}")
        nodes_by_boundary = {
            (source_stage, source_stage + 1): {
                "hidden_states": [],
                "residual": [],
            }
            for source_stage in range(pipeline_size - 1)
        }
        allowed_cross_rank_edges = set()
        for node in p2p_nodes:
            _validate_pipeline_p2p_node(node, pool)
            attributes = node.attributes
            boundary = (
                attributes["source_stage"],
                attributes["destination_stage"],
            )
            if boundary not in nodes_by_boundary:
                raise ValueError(
                    "TP=1 pipeline P2P node crosses an unconfigured boundary: "
                    f"{boundary!r}")
            tensor_name = attributes["payload_tensor"]["name"]
            nodes_by_boundary[boundary][tensor_name].append(node)

        for boundary, nodes_by_tensor in nodes_by_boundary.items():
            source_stage, destination_stage = boundary
            hidden_nodes = tuple(nodes_by_tensor["hidden_states"])
            residual_nodes = tuple(nodes_by_tensor["residual"])
            if len(hidden_nodes) != 1 or len(residual_nodes) != 1:
                raise ValueError(
                    "each TP=1 pipeline boundary requires exactly one "
                    "hidden_states node and one residual node: "
                    f"boundary={boundary!r}, hidden={len(hidden_nodes)}, "
                    f"residual={len(residual_nodes)}")
            hidden = hidden_nodes[0]
            residual = residual_nodes[0]
            hidden_payload = hidden.attributes["payload_tensor"]
            residual_payload = residual.attributes["payload_tensor"]
            if (tuple(hidden_payload["shape"])
                    != tuple(residual_payload["shape"])
                    or hidden_payload["dtype"] != residual_payload["dtype"]
                    or hidden_payload["bytes"] != residual_payload["bytes"]
                    or hidden.attributes["bytes"]
                    != residual.attributes["bytes"]):
                raise ValueError(
                    "pipeline hidden_states and residual payload geometry "
                    f"must match at boundary {boundary!r}")

            hidden_incoming = tuple(
                edge for edge in graph.edges
                if edge.destination == hidden.node_id)
            residual_incoming = tuple(
                edge for edge in graph.edges
                if edge.destination == residual.node_id)
            hidden_outgoing = tuple(
                edge for edge in graph.edges if edge.source == hidden.node_id)
            residual_outgoing = tuple(
                edge for edge in graph.edges
                if edge.source == residual.node_id)
            if (len(hidden_incoming) != 1
                    or hidden_incoming[0].kind != "ready"
                    or len(residual_incoming) != 1
                    or residual_incoming[0].source != hidden.node_id
                    or residual_incoming[0].kind != "ready"
                    or len(hidden_outgoing) != 1
                    or hidden_outgoing[0].destination != residual.node_id
                    or hidden_outgoing[0].kind != "ready"
                    or len(residual_outgoing) != 1
                    or residual_outgoing[0].kind != "data"):
                raise ValueError(
                    "pipeline tensor pair must be an ordered source -> "
                    "hidden_states -> residual -> destination chain: "
                    f"boundary={boundary!r}")
            source = graph.node(hidden_incoming[0].source)
            destination = graph.node(residual_outgoing[0].destination)
            source_ranks = pool.physical_ranks_for_pipeline_stage(source_stage)
            destination_ranks = pool.physical_ranks_for_pipeline_stage(
                destination_stage)
            if (len(source_ranks) != 1 or len(destination_ranks) != 1):
                raise RuntimeError(
                    "TP=1 pipeline stage must resolve to one physical rank")
            valid_source_geometry = (
                source.kind is WorkloadKind.LINEAR
                and source.variant == "gemm"
                and set(source.shape) == {"b", "m", "n", "k"}
                and source.shape["b"] == 1
            )
            if target_contract is not None:
                valid_source_geometry = source.kind in (
                    WorkloadKind.LINEAR, WorkloadKind.ELEMENTWISE)
            if (source.rank != source_ranks[0]
                    or source.dtype != "bf16"
                    or not valid_source_geometry):
                raise ValueError(
                    "pipeline tensor pair must follow one supported BF16 "
                    f"source-stage frontier: boundary={boundary!r}")
            source_outgoing = tuple(
                edge for edge in graph.edges if edge.source == source.node_id)
            if (len(source_outgoing) != 1
                    or source_outgoing[0].destination != hidden.node_id
                    or source_outgoing[0].kind != "ready"):
                raise ValueError(
                    "pipeline transfer must start at the final source-stage "
                    f"frontier: boundary={boundary!r}")
            source_stage_compute = tuple(
                node for node in graph.nodes
                if (node.pool_name == pool.name
                    and node.rank == source_ranks[0]
                    and node.kind is not WorkloadKind.P2P)
            )
            source_stage_terminals = tuple(
                node for node in source_stage_compute
                if not any(
                    edge.source == node.node_id
                    and graph.node(edge.destination).rank == source_ranks[0]
                    and graph.node(edge.destination).kind
                    is not WorkloadKind.P2P
                    for edge in graph.edges
                )
            )
            if tuple(node.node_id for node in source_stage_terminals) != (
                    source.node_id,):
                raise ValueError(
                    "pipeline source stage must have exactly one final compute "
                    f"frontier: boundary={boundary!r}, terminals="
                    f"{tuple(node.node_id for node in source_stage_terminals)!r}")
            if (destination.rank != destination_ranks[0]
                    or destination.kind is not WorkloadKind.NORM
                    or destination.dtype != "bf16"
                    or set(destination.shape) != {"rows", "hidden"}):
                raise ValueError(
                    "pipeline tensor pair must release one BF16 destination "
                    f"input norm: boundary={boundary!r}")
            payload_shape = tuple(hidden_payload["shape"])
            if target_contract is None:
                expected_shape = (source.shape["m"], source.shape["n"])
            else:
                expected_shape = payload_shape
            destination_shape = (
                destination.shape["rows"], destination.shape["hidden"])
            if (payload_shape != expected_shape
                    or destination_shape != expected_shape):
                raise ValueError(
                    "pipeline source, payload, and destination tensor shapes "
                    f"disagree at boundary {boundary!r}")
            if tuple(graph.predecessors(destination.node_id)) != (
                    residual.node_id,):
                raise ValueError(
                    "destination stage must wait for the complete ordered "
                    f"tensor pair at boundary {boundary!r}")
            allowed_cross_rank_edges.add(
                (residual.node_id, destination.node_id, "data"))

        actual_cross_rank_edges = {
            (edge.source, edge.destination, edge.kind)
            for edge in graph.edges
            if graph.node(edge.source).rank != graph.node(edge.destination).rank
        }
        if actual_cross_rank_edges != allowed_cross_rank_edges:
            raise ValueError(
                "TP=1 pipeline graph contains a missing or unmodeled "
                "cross-rank dependency: "
                f"expected={sorted(allowed_cross_rank_edges)!r}, "
                f"actual={sorted(actual_cross_rank_edges)!r}")


def _validate_tp2_pp2_pipeline_graph(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
) -> None:
    """Validate the complete pinned vLLM TP=2, PP=2 boundary."""

    for pool in deployment.worker_pools.values():
        parallelism = pool.parallel_plan.config
        if (parallelism.tensor_parallel_size,
                parallelism.pipeline_parallel_size) != (2, 2):
            continue
        if (deployment.mode != "colocated"
                or len(deployment.worker_pools) != 1
                or parallelism.worker_count != 4
                or parallelism.expert_parallel_size != 1
                or not parallelism.is_rank_local):
            raise NotImplementedError(
                "TP=2, PP=2 lowering supports exactly one colocated pool "
                "with EP=SP=CP=DP=1")
        source_ranks = pool.physical_ranks_for_pipeline_stage(0)
        destination_ranks = pool.physical_ranks_for_pipeline_stage(1)
        if len(source_ranks) != 2 or len(destination_ranks) != 2:
            raise RuntimeError(
                "TP=2, PP=2 lowering requires two ranks per pipeline stage")
        p2p_nodes = tuple(
            node for node in graph.nodes
            if node.pool_name == pool.name and node.kind is WorkloadKind.P2P
        )
        if len(p2p_nodes) != 4:
            raise ValueError(
                "TP=2, PP=2 boundary requires exactly four P2P lane nodes")
        expected_pairs = tuple(zip(source_ranks, destination_ranks))
        nodes_by_tensor = {
            "hidden_states": [],
            "residual": [],
        }
        for node in p2p_nodes:
            _validate_pipeline_p2p_node(node, pool)
            attributes = node.attributes
            if (attributes["source_stage"],
                    attributes["destination_stage"]) != (0, 1):
                raise ValueError(
                    "TP=2, PP=2 P2P nodes must cross stage 0 to stage 1")
            pair = (
                attributes["source_rank"],
                attributes["destination_rank"],
            )
            if pair not in expected_pairs:
                raise ValueError(
                    "TP=2, PP=2 P2P endpoint pair is not a canonical PP lane")
            tensor_name = attributes["payload_tensor"]["name"]
            if tensor_name not in nodes_by_tensor:
                raise ValueError(
                    "TP=2, PP=2 boundary payload must be hidden_states or "
                    "residual")
            nodes_by_tensor[tensor_name].append(node)

        all_gathers = {}
        for tensor_name in ("hidden_states", "residual"):
            tensor_nodes = tuple(nodes_by_tensor[tensor_name])
            if len(tensor_nodes) != 2:
                raise ValueError(
                    "TP=2, PP=2 boundary requires two P2P lanes per tensor")
            actual_pairs = tuple(
                (
                    node.attributes["source_rank"],
                    node.attributes["destination_rank"],
                )
                for node in tensor_nodes
            )
            if actual_pairs != expected_pairs:
                raise ValueError(
                    "TP=2, PP=2 tensor lanes must follow ordered TP rank "
                    "pairs")
            successor_sets = tuple(
                {
                    edge.destination
                    for edge in graph.edges
                    if edge.source == node.node_id
                }
                for node in tensor_nodes
            )
            if (any(len(successors) != 1 for successors in successor_sets)
                    or successor_sets[0] != successor_sets[1]):
                raise ValueError(
                    "both TP=2, PP=2 P2P lanes must converge on exactly one "
                    "tensor all-gather")
            collective_id = next(iter(successor_sets[0]))
            collective = graph.node(collective_id)
            if (collective.kind is not WorkloadKind.COLLECTIVE
                    or collective.attributes["operation"] != "all_gather"
                    or collective.attributes["bytes_semantics"]
                    != "per_rank_input"
                    or tuple(collective.attributes["participants"])
                    != destination_ranks):
                raise ValueError(
                    "TP=2, PP=2 P2P lanes must feed the destination TP "
                    "all-gather")
            if tuple(graph.predecessors(collective_id)) != tuple(
                    node.node_id for node in tensor_nodes):
                raise ValueError(
                    "TP=2, PP=2 all-gather must wait for both matching P2P "
                    "lanes")
            barrier_edges = tuple(
                edge for edge in graph.edges
                if edge.destination == collective_id
            )
            if any(edge.kind != "barrier" for edge in barrier_edges):
                raise ValueError(
                    "TP=2, PP=2 P2P-to-all-gather dependencies must be "
                    "barriers")
            if (collective.attributes["direction"] != "collective"
                    or collective.attributes["source_rank"] is not None
                    or collective.attributes["destination_rank"] is not None
                    or tuple(collective.attributes["ring_order"])
                    != destination_ranks
                    or collective.attributes["reduction"] is not None):
                raise ValueError(
                    "TP=2, PP=2 boundary all-gather attributes are not "
                    "canonical")
            all_gathers[tensor_name] = collective
        if (all_gathers["hidden_states"].node_id
                == all_gathers["residual"].node_id):
            raise ValueError(
                "hidden_states and residual require distinct all-gathers")
        destination_group_all_gathers = tuple(
            node for node in graph.nodes
            if (node.pool_name == pool.name
                and node.kind is WorkloadKind.COLLECTIVE
                and node.attributes["operation"] == "all_gather"
                and tuple(node.attributes["participants"])
                == destination_ranks)
        )
        if len(destination_group_all_gathers) != 3:
            raise ValueError(
                "TP=2, PP=2 graph requires exactly two boundary all-gathers "
                "and one destination-stage LM-head all-gather")
        boundary_all_gather_ids = {
            all_gathers["hidden_states"].node_id,
            all_gathers["residual"].node_id,
        }
        lm_head_all_gathers = tuple(
            node for node in destination_group_all_gathers
            if node.node_id not in boundary_all_gather_ids
        )
        if len(lm_head_all_gathers) != 1:
            raise ValueError(
                "TP=2, PP=2 graph requires one non-boundary destination "
                "all-gather")
        lm_head_all_gather = lm_head_all_gathers[0]
        lm_head_predecessors = tuple(
            graph.node(node_id)
            for node_id in graph.predecessors(lm_head_all_gather.node_id)
        )
        if (len(lm_head_predecessors) != 2
                or tuple(node.rank for node in lm_head_predecessors)
                != destination_ranks
                or any(node.kind is not WorkloadKind.LM_HEAD
                       for node in lm_head_predecessors)
                or any(
                    edge.kind != "barrier"
                    for edge in graph.edges
                    if edge.destination == lm_head_all_gather.node_id
                )
                or any(edge.source == lm_head_all_gather.node_id
                       for edge in graph.edges)):
            raise ValueError(
                "TP=2, PP=2 destination LM-head all-gather topology is not "
                "canonical")

        destination_successors = []
        for tensor_name in ("hidden_states", "residual"):
            collective = all_gathers[tensor_name]
            release_edges = tuple(
                edge for edge in graph.edges
                if edge.source == collective.node_id
            )
            if any(edge.kind != "data" for edge in release_edges):
                raise ValueError(
                    "TP=2, PP=2 all-gather must release destination compute "
                    "through data dependencies")
            destination_successors.append(tuple(
                edge.destination for edge in release_edges))
        if (destination_successors[0] != destination_successors[1]
                or len(destination_successors[0]) != 2):
            raise ValueError(
                "both TP=2, PP=2 boundary all-gathers must release the same "
                "two destination-rank compute nodes")
        destination_nodes = tuple(
            graph.node(node_id) for node_id in destination_successors[0])
        if tuple(node.rank for node in destination_nodes) != destination_ranks:
            raise ValueError(
                "TP=2, PP=2 boundary destination nodes disagree with the "
                "destination TP group")
        if any(
                node.kind is not WorkloadKind.NORM
                or node.dtype != "bf16"
                or set(node.shape) != {"rows", "hidden"}
                or node.shape["rows"] <= 0
                or node.shape["hidden"] <= 0
                for node in destination_nodes):
            raise ValueError(
                "TP=2, PP=2 boundary must release two BF16 decoder input "
                "normalization nodes")
        first_shape = dict(destination_nodes[0].shape)
        if any(dict(node.shape) != first_shape for node in destination_nodes[1:]):
            raise ValueError(
                "TP=2, PP=2 destination-rank input shapes must match")
        for node in destination_nodes:
            successor_edges = tuple(
                edge for edge in graph.edges if edge.source == node.node_id)
            if (len(successor_edges) != 1
                    or successor_edges[0].kind != "data"):
                raise ValueError(
                    "each TP=2, PP=2 destination input norm must feed one "
                    "rank-local QKV node")
            qkv = graph.node(successor_edges[0].destination)
            if (qkv.kind is not WorkloadKind.LINEAR
                    or qkv.rank != node.rank
                    or qkv.dtype != "bf16"
                    or qkv.variant != "gemm"
                    or set(qkv.shape) != {"b", "m", "n", "k"}
                    or qkv.shape["b"] != 1
                    or qkv.shape["m"] != first_shape["rows"]
                    or qkv.shape["k"] != first_shape["hidden"]):
                raise ValueError(
                    "TP=2, PP=2 destination input norm shape disagrees with "
                    "its rank-local QKV successor")
        all_gather_ids = tuple(
            all_gathers[name].node_id
            for name in ("hidden_states", "residual")
        )
        if any(tuple(graph.predecessors(node.node_id)) != all_gather_ids
               for node in destination_nodes):
            raise ValueError(
                "each TP=2, PP=2 destination node must wait for both tensor "
                "all-gathers")

        source_predecessors = tuple(
            graph.predecessors(node.node_id) for node in p2p_nodes)
        if (any(len(predecessors) != 1
                for predecessors in source_predecessors)
                or len({predecessors[0]
                        for predecessors in source_predecessors}) != 1):
            raise ValueError(
                "all TP=2, PP=2 P2P lanes must share one completed source "
                "frontier")
        ready_edges = tuple(
            edge for edge in graph.edges
            if edge.destination in {node.node_id for node in p2p_nodes}
        )
        if (len(ready_edges) != 4
                or any(edge.kind != "ready" for edge in ready_edges)):
            raise ValueError(
                "TP=2, PP=2 source-to-P2P dependencies must contain four "
                "ready edges")
        source_frontier = graph.node(source_predecessors[0][0])
        if (source_frontier.kind is not WorkloadKind.COLLECTIVE
                or source_frontier.attributes["operation"] != "all_reduce"
                or tuple(source_frontier.attributes["participants"])
                != source_ranks):
            raise ValueError(
                "TP=2, PP=2 P2P lanes must follow the source-stage TP "
                "all-reduce frontier")
        source_compute_nodes = tuple(
            graph.node(node_id)
            for node_id in graph.predecessors(source_frontier.node_id)
        )
        if (len(source_compute_nodes) != 2
                or tuple(node.rank for node in source_compute_nodes)
                != source_ranks
                or any(
                    node.kind is not WorkloadKind.LINEAR
                    or node.dtype != "bf16"
                    or node.variant != "gemm"
                    or set(node.shape) != {"b", "m", "n", "k"}
                    or node.shape["b"] != 1
                    for node in source_compute_nodes)
                or any(
                    edge.kind != "barrier"
                    for edge in graph.edges
                    if edge.destination == source_frontier.node_id
                )):
            raise ValueError(
                "TP=2, PP=2 source frontier must follow two rank-local BF16 "
                "linear nodes")
        source_shape = {
            "rows": source_compute_nodes[0].shape["m"],
            "hidden": source_compute_nodes[0].shape["n"],
        }
        if (any(node.shape["m"] != source_shape["rows"]
                or node.shape["n"] != source_shape["hidden"]
                for node in source_compute_nodes[1:])
                or first_shape != source_shape):
            raise ValueError(
                "TP=2, PP=2 source linear, destination norm, and QKV "
                "tensor shapes must agree")

        total_elements = source_shape["rows"] * source_shape["hidden"]
        if total_elements % 2:
            raise ValueError(
                "TP=2, PP=2 boundary tensor elements must divide TP size")
        expected_wire_elements = total_elements // 2
        expected_wire_bytes = expected_wire_elements * 2
        if source_frontier.attributes["bytes"] != total_elements * 2:
            raise ValueError(
                "TP=2, PP=2 source all-reduce bytes disagree with the "
                "source [T,H] tensor")
        for node in p2p_nodes:
            payload = node.attributes["payload_tensor"]
            if (tuple(payload["shape"]) != (expected_wire_elements,)
                    or payload["bytes"] != expected_wire_bytes
                    or node.attributes["bytes"] != expected_wire_bytes):
                raise ValueError(
                    "TP=2, PP=2 P2P wire shape/bytes disagree with the "
                    "destination [T,H] tensor and TP size")
        for collective in all_gathers.values():
            if collective.attributes["bytes"] != expected_wire_bytes:
                raise ValueError(
                    "TP=2, PP=2 all-gather bytes disagree with the canonical "
                    "per-rank wire slice")


def _eager_attn_specs(node: WorkloadNode, hardware_id: str, hardware):
    required = ("batch", "q_len", "kv_len", "head_dim", "num_query_heads")
    missing = [name for name in required if name not in node.shape]
    if missing:
        raise ValueError(f"eager attention missing shape fields: {missing}")
    b = node.shape["batch"]
    h = node.shape["num_query_heads"]
    q = node.shape["q_len"]
    kv = node.shape["kv_len"]
    d = node.shape["head_dim"]
    shapes = (
        {"b": b * h, "m": q, "n": kv, "k": d},
        {"element_count": b * h * q * kv},
        {"b": b * h, "m": q, "n": d, "k": kv},
    )
    result = []
    common_attributes = dict(node.attributes)
    if "launch_policy" not in common_attributes:
        raise ValueError(
            "eager attention lowering requires explicit launch_policy")
    for name in ("tensor_parallel_size", "expert_parallel_size"):
        if name not in common_attributes:
            raise ValueError(
                f"eager attention lowering requires explicit {name}")
    for index, (kind, shape, variant) in enumerate((
        (KernelKind.GEMM, shapes[0], "gemm"),
        (KernelKind.VECTOR, shapes[1], "softmax"),
        (KernelKind.GEMM, shapes[2], "gemm"),
    )):
        selection_attributes = dict(common_attributes)
        if index in (0, 2):
            selection_attributes.update({
                "a_layout": "row_major",
                "b_layout": "row_major",
                "c_layout": "row_major",
                "trans_a": False,
                "trans_b": index == 0,
                "epilogue": "none",
                "has_bias": False,
                "position": (
                    "attention_qk" if index == 0 else "attention_pv"),
            })
        launch = select_launch_config(
            hardware_id, kind, shape, node.dtype, variant,
            selection_attributes, hardware=hardware)
        spec = KernelSpec(
            kind, hardware_id, shape, dtype=node.dtype,
            variant=launch.execution_variant,
            launch=launch)
        if kind is KernelKind.GEMM:
            result.extend(_gemm_specs(
                spec, shape, variant, selection_attributes,
                hardware_id, hardware))
        elif kind is KernelKind.VECTOR:
            result.append(spec)
        else:
            raise ValueError(
                f"unhandled eager-attention kernel kind: {kind.value}")
    return tuple(result)


def lower_workload(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
    hardware_by_id: Mapping[str, object],
    memo: KernelLoweringMemo | None = None,
) -> KernelPlan:
    if memo is not None and not isinstance(memo, KernelLoweringMemo):
        raise TypeError("kernel lowering memo must be KernelLoweringMemo")
    if memo is None:
        logical_node_contracts = None
    else:
        logical_node_contracts = _memo_contracts(graph)
    target_contract = _target_model_semantics_contract(graph)
    block_fp8_checkpoint = _dsv3_block_fp8_checkpoint(graph, target_contract)
    if target_contract is None:
        _validate_ep_fused_moe_graph(graph, deployment)
    _validate_tp1_pipeline_graph(graph, deployment)
    _validate_tp2_pp2_pipeline_graph(graph, deployment)
    if deployment.mode in ("af_disagg", "pd_af_disagg"):
        for node in graph.nodes:
            if node.kind is WorkloadKind.AF_ACTIVATION_TRANSFER:
                _validate_af_activation_transfer_node(
                    node, deployment, graph)
        _validate_af_activation_boundaries(graph, deployment)
    fusion_heads, fusion_member_heads = (
        _ascend_qkv_rmsnorm_rope_fusions(
            graph, deployment, hardware_by_id))
    calls = []
    final_call_by_node = {}
    call_ids_by_node = {}
    for node_id in graph.topological_order():
        node = graph.node(node_id)
        if (node_id in fusion_member_heads
                and fusion_member_heads[node_id] != node_id):
            continue
        if node.pool_name not in deployment.worker_pools:
            raise ValueError(f"workload node references unknown pool {node.pool_name!r}")
        pool = deployment.worker_pools[node.pool_name]
        if node.rank is None:
            raise ValueError(
                f"workload node {node.node_id!r} requires an explicit rank")
        if node.rank not in pool.ranks:
            raise ValueError(
                f"workload node {node.node_id!r} rank {node.rank} is not "
                f"assigned to pool {pool.name!r}: {pool.ranks!r}")
        hardware_id = pool.hardware_id
        hardware = require_hardware(hardware_id, hardware_by_id)
        if node_id in fusion_heads:
            group = fusion_heads[node_id]
            group_ids = tuple(member.node_id for member in group)
            group_id_set = set(group_ids)
            external_node_dependencies = tuple(dict.fromkeys(
                predecessor
                for member in group
                for predecessor in graph.predecessors(member.node_id)
                if predecessor not in group_id_set
            ))
            external_dependencies = tuple(
                final_call_by_node[predecessor]
                for predecessor in external_node_dependencies)

            def build_fused_specs():
                return _ascend_qkv_rmsnorm_rope_specs(
                    *group, hardware_id, hardware)

            if memo is None:
                specs = build_fused_specs()
            else:
                composite_contract = tuple(
                    logical_node_contracts[member_id]
                    for member_id in group_ids)
                specs = memo.lower(
                    node, composite_contract, hardware, build_fused_specs)
            if not specs:
                raise ValueError(
                    f"workload node {node.node_id!r} lowered to no "
                    "KernelSpecs")
            # The materialization helpers and the fused kernel form one
            # source-level chained string: each helper's only in-string
            # predecessor is the previous spec, so the fused head lowers to
            # a call per spec sharing the group's source identity.
            fused_call_ids = []
            previous = None
            for index, spec in enumerate(specs):
                call_id = (
                    node.node_id if len(specs) == 1
                    else f"{node.node_id}:{index}")
                dependencies = (
                    external_dependencies if previous is None
                    else (previous,))
                calls.append(KernelCall(
                    call_id,
                    spec,
                    node.pool_name,
                    node.rank,
                    "compute",
                    group_ids,
                    dependencies,
                ))
                fused_call_ids.append(call_id)
                previous = call_id
            for member_id in group_ids:
                final_call_by_node[member_id] = previous
                call_ids_by_node[member_id] = tuple(fused_call_ids)
            continue
        if node.kind in (
                WorkloadKind.COLLECTIVE,
                WorkloadKind.P2P,
                WorkloadKind.AF_ACTIVATION_TRANSFER):
            _validate_graph_domain_boundary(node, graph)
        if node.kind is WorkloadKind.COLLECTIVE:
            _validate_collective_node(node, pool, graph)
        if node.kind is WorkloadKind.P2P:
            _validate_pipeline_p2p_node(node, pool)
        if node.kind is WorkloadKind.PD_KV_TRANSFER:
            _validate_pd_kv_transfer_node(node, deployment)
        routing_relabelling = None
        if node.kind is WorkloadKind.FUSED_MOE and target_contract is None:
            # The relabelling only names the physical class, so the routing
            # this node actually carries still has to answer for itself: keep
            # its provenance check on every step rather than on the steps that
            # happen to open a new class.
            _validate_routing_provenance(node)
            routing_relabelling = _fused_moe_routing_relabelling(
                node, hardware_id, hardware)
        def build_specs():
            if target_contract is not None:
                if node.kind is WorkloadKind.ROTARY:
                    return (_target_rope_spec(node, hardware_id, hardware),)
                if (node.kind is WorkloadKind.NORM
                        and node.variant == "layernorm"):
                    return (_target_layernorm_spec(
                        node, hardware_id, hardware),)
                if node.kind is WorkloadKind.KV_UPDATE:
                    return (_target_kv_cache_update_spec(
                        node, hardware_id, hardware),)
                if (node.kind is WorkloadKind.GATHER
                        and node.variant == "dsa_index_reuse"):
                    return (_target_sparse_indexer_spec(
                        node, hardware_id, hardware),)
                if node.kind is WorkloadKind.FLASH_ATTN:
                    return _target_mla_attention_specs(
                        node, hardware_id, hardware)
                if node.kind is WorkloadKind.EAGER_ATTN:
                    if node.variant.startswith("dsa_indexer_"):
                        return (_target_sparse_indexer_spec(
                            node, hardware_id, hardware),)
                    if node.variant.startswith("dsa_sparse_"):
                        return (_target_sparse_attention_spec(
                            node, hardware_id, hardware),)
                    raise ValueError(
                        "unknown target-model EAGER_ATTN logical variant: "
                        f"{node.variant!r}")
                if node.kind is WorkloadKind.FUSED_MOE:
                    return _target_fused_moe_specs(
                        node, hardware_id, hardware)
                if node.kind in (
                        WorkloadKind.LINEAR, WorkloadKind.LM_HEAD,
                        WorkloadKind.ROUTER):
                    return _target_gemm_specs(
                        node, hardware_id, hardware, block_fp8_checkpoint)
            if node.kind is WorkloadKind.FLASH_ATTN:
                return _flash_attn_specs(node, hardware_id, hardware)
            if node.kind is WorkloadKind.NORM:
                return _norm_specs(node, hardware_id, hardware)
            if node.kind is WorkloadKind.FUSED_MOE:
                return _fused_moe_specs(
                    node, hardware_id, hardware, routing_relabelling)
            if node.kind in (
                    WorkloadKind.LINEAR, WorkloadKind.LM_HEAD,
                    WorkloadKind.ROUTER):
                return _gemm_node_specs(node, hardware_id, hardware)
            if node.kind in _SINGLE_COMPUTE_KIND:
                return (_compute_spec(node, hardware_id, hardware),)
            if node.kind is WorkloadKind.EAGER_ATTN:
                return _eager_attn_specs(node, hardware_id, hardware)
            if node.kind in _COMMUNICATION_KINDS or node.kind in (
                    WorkloadKind.MEMCPY, WorkloadKind.KV_OFFLOAD):
                return (_transfer_spec(node, hardware_id),)
            raise NotImplementedError(
                f"no lowering rule for {node.kind.value}")

        if memo is None:
            specs = build_specs()
        elif node.kind is WorkloadKind.FUSED_MOE:
            specs = memo.lower(
                node,
                _logical_node_contract(node, routing_relabelling),
                hardware,
                build_specs,
            )
        else:
            specs = memo.lower(
                node,
                logical_node_contracts[node.node_id],
                hardware,
                build_specs,
            )
        if not specs:
            raise ValueError(
                f"workload node {node.node_id!r} lowered to no KernelSpecs")
        external_dependencies = tuple(
            final_call_by_node[pred] for pred in graph.predecessors(node.node_id))
        previous = None
        node_call_ids = []
        for index, spec in enumerate(specs):
            call_id = node.node_id if len(specs) == 1 else f"{node.node_id}:{index}"
            dependencies = external_dependencies if previous is None else (previous,)
            if spec.transfer is None:
                resource = "compute"
            elif spec.kind is KernelKind.COMMUNICATION:
                resource = spec.transfer.resource_id
            else:
                resource = spec.transfer.link
            rank = node.rank
            calls.append(KernelCall(
                call_id,
                spec,
                node.pool_name,
                rank,
                resource,
                (node.node_id,),
                dependencies,
            ))
            node_call_ids.append(call_id)
            previous = call_id
        final_call_by_node[node.node_id] = previous
        call_ids_by_node[node.node_id] = tuple(node_call_ids)
    if set(call_ids_by_node) != {node.node_id for node in graph.nodes}:
        raise AssertionError("kernel lowering did not consume every logical node")
    calls_by_id = {call.call_id: call for call in calls}
    if any(
            node_id not in calls_by_id[call_id].source_node_ids
            for node_id, call_ids in call_ids_by_node.items()
            for call_id in call_ids):
        raise AssertionError(
            "each logical source node must map to its lowered KernelCall")
    materialized_plan = KernelPlan(calls)
    topological_calls = materialized_plan.topological_order()
    block_repeats = []
    for repeat in graph.block_repeats:
        repeated_node_ids = set(repeat.node_ids)
        block_call_ids = tuple(
            call.call_id
            for call in topological_calls
            if any(
                node_id in repeated_node_ids
                for node_id in call.source_node_ids
            )
        )
        feedback_by_destination = {}
        for edge in repeat.feedback_edges:
            destination_call_id = call_ids_by_node[edge.destination][0]
            source_call_id = final_call_by_node[edge.source]
            if destination_call_id not in feedback_by_destination:
                feedback_by_destination[destination_call_id] = []
            feedback_by_destination[destination_call_id].append(source_call_id)
        feedback_dependencies = tuple(
            (
                call_id,
                tuple(feedback_by_destination[call_id]),
            )
            for call_id in block_call_ids
            if call_id in feedback_by_destination
        )
        block_repeats.append(KernelBlockRepeat(
            repeat.block_id,
            block_call_ids,
            repeat.repeat_count,
            feedback_dependencies,
        ))
    return KernelPlan(calls, block_repeats)
