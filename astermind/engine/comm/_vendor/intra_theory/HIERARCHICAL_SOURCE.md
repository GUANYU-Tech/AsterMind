# Hierarchical source

`hierarchical_upstream.py` is an unchanged copy of
`surrogate/theory/whitebox_surrogate_v2.py` from
https://github.com/Hazuyuki/ProtoBridge at commit
`c0e4bdc434bd1f18bd3f8e80d80ca2263369ff1f`.
SHA256: `1e5560441ab2bbf75ea843cb1b3cee608eb059c62fb8a9be51431711ecfe3927`.

This additive copy does not replace the older intra profile. See the adjacent
LICENSE and upstream licensing notices. Adapter code is `comm/intra_hierarchical.py`.
The upstream so profile is an uncalibrated analytical lower-bound model;
passing integration tests does not constitute hardware or packet-level calibration.
The upstream `so_per_link_gbps` parameter is GB/s, despite its name: the adapter
converts Network Gbit/s by dividing by eight exactly once.
