#!/bin/bash
# usage: retry.sh <gpu> <window-ids e.g. 4,5,6> <plan>:<abs case-id file> [<plan>:<file> ...]
# Extra windows for cases excluded at assembly (spread / launch identity / clock), as in the
# WP9 retry rounds; assembly keeps the newest three accepted windows. One process per GPU
# (launch.sh takes gpu<N>.proc.lock and forces --concurrency 1).
K=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/kcampaign
GPU=$1; WIDS=$2; shift 2
for pair in "$@"; do
  PLAN=${pair%%:*}; IDS=${pair#*:}
  echo "$(date '+%F %T') RETRY $PLAN windows $WIDS cases $(wc -l < $IDS) gpu$GPU"
  $K/launch.sh $GPU $PLAN --window-ids $WIDS --concurrency 1 --case-ids $IDS >> $K/logs/${PLAN}_retry_gpu${GPU}.log 2>&1
  echo "$(date '+%F %T') END $PLAN rc=$? $(tail -1 $K/logs/${PLAN}_retry_gpu${GPU}.log)"
done
