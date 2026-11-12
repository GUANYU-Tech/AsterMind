#!/bin/bash
# usage: check_8gpu_machine.sh <out_dir>
#
# NVML-only (nvidia-smi) pre-flight for the 2026-09-27 8-GPU H200 slot; it
# creates no CUDA context and needs no GPU lock.  Fails (exit 1) unless:
#   - exactly 8 GPUs, all "NVIDIA H200", driver 575.57.08 (as on 2026-09-26);
#   - every GPU pair in `nvidia-smi topo -m` is NV18 (one NVLink/NVSwitch
#     domain), fabric state Completed / status Success on every GPU;
#   - clocks.max.sm and clocks.applications.graphics are 1980 MHz.
# It also compares the four GPUs seen on 2026-09-26 (UUIDs below) with the new
# index -> UUID map and prints whether they kept their indices; an index
# shift changes which physical GPU each gpu<N>.proc.lock and every running
# chain's --physical-gpu refers to (WARN, not FAIL: the operator decides).
# Idle clocks are expected to be below 1980 MHz; busy-clock checks happen in
# each job's clock log.
set -uo pipefail
OUT=${1:?usage: check_8gpu_machine.sh <out_dir>}
mkdir -p "$OUT"
FAIL=0
fail() { echo "FAIL: $*"; FAIL=1; }

# 2026-09-26 05:47 CST, host is-ddmett2r4j2rwpc5-devmachine-0 (4 GPUs):
declare -A OLD_UUID=(
  [0]=GPU-2720e41a-33a0-d94d-06c9-5799a013c261   # 00000000:3A:00.0
  [1]=GPU-ce913a0d-5c10-7df0-2274-3b14425b8904   # 00000000:4C:00.0
  [2]=GPU-d4c48f84-689c-9cf0-3377-1f5906f34447   # 00000000:5C:00.0
  [3]=GPU-d008ae25-3ae4-35f5-3029-e63a78509d91   # 00000000:AA:00.0
)

hostname | tee "$OUT/hostname.txt"
nvidia-smi > "$OUT/nvidia_smi.txt"
nvidia-smi topo -m > "$OUT/topo.txt"
nvidia-smi -q > "$OUT/nvidia_smi_q.txt"
nvidia-smi nvlink -s > "$OUT/nvlink_status.txt"
nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,driver_version,memory.total,clocks.sm,clocks.max.sm,clocks.applications.graphics,fabric.state,fabric.status,utilization.gpu,memory.used \
  --format=csv,noheader > "$OUT/gpus.csv"
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv > "$OUT/compute_apps.csv"
cat "$OUT/gpus.csv"

N=$(wc -l < "$OUT/gpus.csv")
[ "$N" = 8 ] || fail "expected 8 GPUs, found $N"
while IFS=',' read -r idx name bus uuid serial drv mem sm maxsm appgr fstate fstatus util used; do
  idx=${idx// /}; uuid=${uuid// /}
  [ "${name# }" = "NVIDIA H200" ] || fail "gpu$idx name '${name# }'"
  [ "${drv// /}" = "575.57.08" ] || fail "gpu$idx driver '${drv# }' (2026-09-26: 575.57.08)"
  [ "${maxsm// /}" = "1980MHz" ] || fail "gpu$idx clocks.max.sm '${maxsm# }'"
  [ "${appgr// /}" = "1980MHz" ] || fail "gpu$idx clocks.applications.graphics '${appgr# }'"
  # driver 575 prints e.g. "state  Completed" / "status Success"
  [ "${fstate// /}" = "stateCompleted" ] || fail "gpu$idx fabric.state '${fstate# }'"
  [ "${fstatus// /}" = "statusSuccess" ] || fail "gpu$idx fabric.status '${fstatus# }'"
done < "$OUT/gpus.csv"

# Every off-diagonal GPU<->GPU cell of the topology matrix must be NV18.
python3 - "$OUT/topo.txt" <<'EOF' || FAIL=1
import re, sys
rows = {}
for line in open(sys.argv[1]):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = re.match(r"^GPU(\d+)\s+(.*)$", line)
    if m:
        rows[int(m.group(1))] = m.group(2).split()
n = len(rows)
bad = [(i, j, rows[i][j]) for i in rows for j in range(n)
       if i != j and rows[i][j] != "NV18"]
print(f"topology: {n} GPUs, non-NV18 GPU pairs: {bad}")
sys.exit(1 if bad or n != 8 else 0)
EOF

echo "index continuity vs 2026-09-26:"
for i in 0 1 2 3; do
  now=$(awk -F', ' -v u="${OLD_UUID[$i]}" '$4==u {print $1}' "$OUT/gpus.csv")
  if [ -z "$now" ]; then echo "  WARN: 2026-09-26 gpu$i (${OLD_UUID[$i]}) not present -> different machine?";
  elif [ "$now" != "$i" ]; then echo "  WARN: 2026-09-26 gpu$i is now gpu$now (lock/chain GPU ids shift!)";
  else echo "  gpu$i unchanged"; fi
done
echo "compute apps currently on GPUs:"; cat "$OUT/compute_apps.csv"
if [ "$FAIL" = 0 ]; then echo "CHECK_8GPU: PASS"; else echo "CHECK_8GPU: FAIL"; fi
exit $FAIL
