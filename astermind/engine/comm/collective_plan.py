"""Logical waves of the explicit full-mesh and binary-tree formation programs."""
from collections import defaultdict


def collective_waves(operation, algorithm, ranks, per_rank_bytes):
    if algorithm=="tree" and operation=="all_reduce":
        # ProtoBridge's default heap tree. Whole-message level barriers are
        # the declared analytical approximation to its pipelined packet tree.
        levels=defaultdict(list)
        for index,child in enumerate(ranks[1:],1):
            levels[(index+1).bit_length()-1].append((child,ranks[(index-1)//2],per_rank_bytes))
        down=[[(p,c,n) for c,p,n in levels[level]] for level in sorted(levels)]
        return [levels[level] for level in sorted(levels,reverse=True)]+down
    if algorithm!="fullmesh" or operation not in ("all_reduce","all_gather"):
        raise NotImplementedError(f"unsupported plan: {operation}/{algorithm}")
    if operation=="all_gather":
        return [[(a,b,per_rank_bytes) for a in ranks for b in ranks if a!=b]]
    base,remainder=divmod(per_rank_bytes,len(ranks))
    chunks={r:base+(i<remainder) for i,r in enumerate(ranks)}
    # Reduce each destination's slice, then distribute each owner's result.
    return [[(a,b,chunks[b]) for a in ranks for b in ranks if a!=b and chunks[b]],
            [(a,b,chunks[a]) for a in ranks for b in ranks if a!=b and chunks[a]]]
