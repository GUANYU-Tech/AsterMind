#!/bin/bash
# usage: chain.sh <gpu> <plan> [<plan> ...] : run plans back to back on one GPU
K=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/kcampaign
GPU=$1; shift
for PLAN in "$@"; do
  echo "$(date '+%F %T') START $PLAN gpu$GPU"
  $K/launch.sh $GPU $PLAN --windows 3 --concurrency 6 > $K/logs/${PLAN}_gpu${GPU}.log 2>&1
  echo "$(date '+%F %T') END $PLAN rc=$? $(tail -1 $K/logs/${PLAN}_gpu${GPU}.log)"
done
