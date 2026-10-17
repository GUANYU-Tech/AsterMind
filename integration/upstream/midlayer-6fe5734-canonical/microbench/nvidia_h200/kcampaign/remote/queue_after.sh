#!/bin/bash
# usage: queue_after.sh <pid_to_wait_for> <gpu> <plan>... : start a chain on <gpu> once <pid> exits.
# REPO in the environment selects the code snapshot, as for launch.sh.
K=/mnt/public/yangxinhao/h200_microbench/kcampaign
PID=$1; GPU=$2; shift 2
while kill -0 $PID 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') predecessor $PID exited; starting chain on gpu$GPU: $*"
exec $K/chain.sh $GPU "$@"
