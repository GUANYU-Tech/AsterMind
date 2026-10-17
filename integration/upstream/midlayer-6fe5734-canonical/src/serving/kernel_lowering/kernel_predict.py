"""Public KernelPlan prediction entry point."""

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Mapping

from src.kernel.kernel_base import predict, prediction_equivalence_attributes
from src.kernel.kernel_spec import KernelPrediction, KernelSpec

from .kernel_plan import KernelPlan, hardware_for_call


_NONE_CONTRACT = ("none",)
# One frozen contract shape per concrete type, resolved once by the same
# ordered predicate chain the shapes are defined by.  The chain is a function
# of the type alone, so caching its answer cannot change any frozen contract.
_CONTRACT_FREEZERS: dict = {}
_CONTRACT_DATACLASS_FIELD_NAMES: dict = {}


def _freeze_none(value):
    del value
    return _NONE_CONTRACT


def _freeze_enum(value):
    kind = type(value)
    return (
        "enum",
        kind.__module__,
        kind.__qualname__,
        freeze_contract_value(value.value),
    )


def _freeze_bool(value):
    return ("bool", value)


def _freeze_int(value):
    return ("int", value)


def _freeze_float(value):
    return ("float", value.hex())


def _freeze_str(value):
    return ("str", value)


def _freeze_mapping(value):
    keys = tuple(value)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError(
            "kernel prediction memo mappings require string keys")
    return (
        "mapping",
        tuple(
            (key, freeze_contract_value(value[key]))
            for key in sorted(keys)
        ),
    )


def _freeze_tuple(value):
    return ("tuple", tuple(freeze_contract_value(item) for item in value))


def _freeze_list(value):
    return ("list", tuple(freeze_contract_value(item) for item in value))


def _freeze_dataclass(value):
    kind = type(value)
    if kind in _CONTRACT_DATACLASS_FIELD_NAMES:
        names = _CONTRACT_DATACLASS_FIELD_NAMES[kind]
    else:
        names = tuple(field.name for field in fields(value))
        _CONTRACT_DATACLASS_FIELD_NAMES[kind] = names
    return (
        "dataclass",
        kind.__module__,
        kind.__qualname__,
        tuple(
            (name, freeze_contract_value(getattr(value, name)))
            for name in names
        ),
    )


def _resolve_contract_freezer(value):
    """Pick the frozen contract shape for one concrete type, in chain order."""

    if value is None:
        return _freeze_none
    if isinstance(value, Enum):
        return _freeze_enum
    if isinstance(value, bool):
        return _freeze_bool
    if isinstance(value, int):
        return _freeze_int
    if isinstance(value, float):
        return _freeze_float
    if isinstance(value, str):
        return _freeze_str
    if isinstance(value, Mapping):
        return _freeze_mapping
    if isinstance(value, tuple):
        return _freeze_tuple
    if isinstance(value, list):
        return _freeze_list
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze_dataclass
    raise TypeError(
        "kernel prediction memo cannot freeze contract value of type "
        f"{type(value).__name__}")


def freeze_contract_value(value):
    """Build an exact hash key from an immutable production contract."""

    kind = type(value)
    if kind in _CONTRACT_FREEZERS:
        return _CONTRACT_FREEZERS[kind](value)
    freezer = _resolve_contract_freezer(value)
    _CONTRACT_FREEZERS[kind] = freezer
    return freezer(value)


def _contract_cache_key(value):
    """Return the exact immutable contract used for run-local reuse."""

    return freeze_contract_value(value)


def _prediction_class_key(kernel_spec: KernelSpec):
    """Return the frozen contract that names this spec's prediction class.

    Most families are identified by their exact contract.  A family that
    declares canonical equivalence attributes reads part of its contract only
    through reductions coarser than the field itself, so every spec in one
    physical class is frozen as the canonical member and shares one
    prediction.  The substitution happens on the frozen contract, so the key
    is exactly the one the canonical spec would have produced.
    """

    attributes = prediction_equivalence_attributes(kernel_spec)
    if attributes is None:
        return _contract_cache_key(kernel_spec)
    return (
        "dataclass",
        type(kernel_spec).__module__,
        type(kernel_spec).__qualname__,
        tuple(
            (
                field.name,
                freeze_contract_value(
                    attributes if field.name == "attributes"
                    else getattr(kernel_spec, field.name)),
            )
            for field in fields(kernel_spec)
        ),
    )


class KernelPredictionMemo:
    """Reuse exact predictor results without retaining source KernelSpecs."""

    def __init__(self):
        self._predictions = {}
        self._hardware_objects = {}
        self._contract_keys_by_spec_identity = {}
        self._hit_count = 0
        self._miss_count = 0

    @property
    def entry_count(self) -> int:
        return len(self._predictions)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def predict(
        self,
        kernel_spec: KernelSpec,
        hardware,
        method: str,
        *,
        contract_key=None,
        equivalent_call_count: int = 1,
    ) -> KernelPrediction:
        if not isinstance(kernel_spec, KernelSpec):
            raise TypeError("kernel prediction memo requires KernelSpec")
        if (isinstance(equivalent_call_count, bool)
                or not isinstance(equivalent_call_count, int)):
            raise TypeError(
                "equivalent kernel call count must be an integer")
        if equivalent_call_count <= 0:
            raise ValueError(
                "equivalent kernel call count must be positive")
        hardware_identity = id(hardware)
        if hardware_identity in self._hardware_objects:
            if self._hardware_objects[hardware_identity] is not hardware:
                raise RuntimeError(
                    "kernel prediction memo hardware identity collision")
        else:
            self._hardware_objects[hardware_identity] = hardware
        if contract_key is None:
            contract_key = _prediction_class_key(kernel_spec)
        key = (
            method,
            hardware_identity,
            contract_key,
        )
        if key in self._predictions:
            self._hit_count += equivalent_call_count
            return self._predictions[key]
        prediction = predict(kernel_spec, hardware, method)
        if not isinstance(prediction, KernelPrediction):
            raise TypeError(
                "kernel_base.predict must return KernelPrediction")
        self._predictions[key] = prediction
        self._miss_count += 1
        self._hit_count += equivalent_call_count - 1
        return prediction

    def contract_key(self, kernel_spec: KernelSpec):
        if not isinstance(kernel_spec, KernelSpec):
            raise TypeError("kernel prediction memo requires KernelSpec")
        identity = id(kernel_spec)
        if identity in self._contract_keys_by_spec_identity:
            retained, contract_key = (
                self._contract_keys_by_spec_identity[identity])
            if retained is not kernel_spec:
                raise RuntimeError(
                    "kernel prediction memo KernelSpec identity collision")
            return contract_key
        contract_key = _prediction_class_key(kernel_spec)
        self._contract_keys_by_spec_identity[identity] = (
            kernel_spec, contract_key)
        return contract_key


def predict_kernel_plan(plan: KernelPlan, hardware_by_id: Mapping[str, object],
                        method: str = "midlayer",
                        memo: KernelPredictionMemo | None = None):
    if memo is not None and not isinstance(memo, KernelPredictionMemo):
        raise TypeError("kernel prediction memo must be KernelPredictionMemo")
    ordered_calls = plan.topological_order()
    if memo is None:
        predictions = {}
        for call in ordered_calls:
            predictions[call.call_id] = predict(
                call.kernel_spec,
                hardware_for_call(call, hardware_by_id), method)
        return predictions

    # The plan still carries every rank-local call and dependency.  Only the
    # pure predictor invocation is grouped by the complete frozen KernelSpec,
    # prediction method, and exact HWSpec object.  A rank whose spec differs
    # in any field therefore forms its own group automatically.
    identity_groups = {}
    for call in ordered_calls:
        hardware = hardware_for_call(call, hardware_by_id)
        identity_key = (method, id(hardware), id(call.kernel_spec))
        if identity_key not in identity_groups:
            identity_groups[identity_key] = [
                call.kernel_spec, hardware, []]
        elif identity_groups[identity_key][0] is not call.kernel_spec:
            raise RuntimeError("plan-local KernelSpec identity collision")
        identity_groups[identity_key][2].append(call.call_id)

    groups = {}
    for kernel_spec, hardware, call_ids in identity_groups.values():
        contract_key = memo.contract_key(kernel_spec)
        group_key = (method, id(hardware), contract_key)
        if group_key not in groups:
            groups[group_key] = [kernel_spec, hardware, []]
        groups[group_key][2].extend(call_ids)

    predictions = {}
    for _group_key, (kernel_spec, hardware, call_ids) in groups.items():
        prediction = memo.predict(
            kernel_spec,
            hardware,
            method,
            contract_key=_group_key[2],
            equivalent_call_count=len(call_ids),
        )
        for call_id in call_ids:
            predictions[call_id] = prediction
    return predictions
