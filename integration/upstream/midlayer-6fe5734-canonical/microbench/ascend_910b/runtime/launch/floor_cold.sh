#!/bin/bash
# Re-measure the complete non-empty AIV memory task floor in the cold
# instruction-cache regime.  The published model was fitted with 200 launches
# per point inside one process, so every launch after the first re-executed
# text the AIV instruction cache already held.  Production AIV launches sit on
# the other side of that boundary (aiv_icache_miss_rate 0.12-0.29 for every
# AI_VECTOR_CORE operator in a serving capture).  One msprof capture per
# process keeps exactly one cold launch; the warm launches of the same process
# reproduce the published regime as the control.
set -u
BASE=/mnt/gpfs/midlayer_yxh/r8floor
BIN=$BASE/build/aiv_memory_task_floor_bench
OUT=$BASE/caps
mkdir -p "$OUT"
for bd in 1 24 48; do
  for rep in 1 2 3 4; do
    d="$OUT/bd${bd}_r${rep}"
    rm -rf "$d"
    ASCEND_RT_VISIBLE_DEVICES=7 msprof --output="$d" --task-time=on \
      --ai-core=on --application="$BIN $bd 6 1" > /dev/null 2>&1
    echo "done bd=$bd rep=$rep"
  done
done
