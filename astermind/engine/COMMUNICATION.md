# Communication execution

The sole runtime is the source RuntimeCommunication state machine, planner and
native progressive-filling allocator. See `CELL_INTEGRATION.md` for the exact
host boundary and `../README.md` for configuration.

For each operator, Comm resolves physical scope and source policy. Supported
single-SU or complete cross-SU collectives can be predicted as one whole star-local
operation; `IntraSatelliteRoofline` uses the original native predictor context.
Unsupported whole-operation shapes use the source collective phase/P2P expansion
or raise the original explicit unsupported error. Variable partition and traffic
matrix byte semantics and phase barriers remain source-defined.

Cross-star P2P uses the existing route policy, then source path/protocol budgets
and one shared allocator. Current physical links, RF pools and terminal gates are
projected from the existing network. Every selected edge must be simultaneously
available. A route that requires intermediate storage waits/fails instead of
silently simulating store-and-forward. The retained isolated route forecast is
still conservative near a contact/horizon boundary.

Source star-local predictions do not contend globally or reprice in flight.
External capacity changes settle old rates before reallocating remaining bytes;
path loss retries an entire attempt and records discarded bytes. Explicit
source deadlines, cancellation and upstream failure isolation remain in force.
Native compute COMM estimates are not charged in addition to the actual Comm
completion callback. No old backend or legacy ProtoBridge wrapper remains active.

Reports contain policy/model selection, assumptions, partitions/rounds, raw
communication events and allocator statistics. `trace_service=true` additionally
records each allocation's resource capacities, flow demands and rates, enabling
conservation audit; this can be large for WideEP. Exact ticks are authoritative.
The frontend is unchanged and new modes are raw-report only.
