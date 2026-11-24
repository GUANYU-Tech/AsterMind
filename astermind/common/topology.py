"""Undirected topology presets for an ordered collection of nodes."""

def pairs_for(nodes, kind):
    if kind == "fully_connected":
        return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]
    if kind not in {"ring", "chain", "switch"}:
        raise ValueError("不支持的拓扑类型")
    if kind == "switch":
        return []
    pairs = list(zip(nodes, nodes[1:]))
    if kind == "ring" and len(nodes) > 2:
        pairs.append((nodes[-1], nodes[0]))
    return pairs
