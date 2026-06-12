#!/usr/bin/env bash
set -euo pipefail

SIM_DIR=/home/qcshi/simu5g-workspace/simu5g-1.4.4/simulations/nr/railway_handover
OUT_ROOT=CodexA3Sweep-20260603-rlbridge

cd "$SIM_DIR"
mkdir -p "results/$OUT_ROOT"

run_case() {
  local scenario="$1"
  local label="$2"
  shift 2

  local out="results/$OUT_ROOT/${scenario}__${label}"
  rm -rf "$out"
  echo "=== RUN $scenario / $label ==="
  ./run -u Cmdenv -c "$scenario" --result-dir="$out" "$@"
}

for scenario in Railway-300-DL-TrackOffset Railway-300-DL-Stress; do
  run_case "$scenario" default
  run_case "$scenario" a3_fast \
    '--**.cellularNic.nrPhy.railwayA3TttHandover=true' \
    '--**.cellularNic.nrPhy.railwayA3Hysteresis=1.5dB' \
    '--**.cellularNic.nrPhy.railwayA3TimeToTrigger=0.04s' \
    '--**.cellularNic.nrPhy.railwayA3MinNeighborRsrp=-130dB'
  run_case "$scenario" a3_mid \
    '--**.cellularNic.nrPhy.railwayA3TttHandover=true' \
    '--**.cellularNic.nrPhy.railwayA3Hysteresis=3dB' \
    '--**.cellularNic.nrPhy.railwayA3TimeToTrigger=0.16s' \
    '--**.cellularNic.nrPhy.railwayA3MinNeighborRsrp=-130dB'
  run_case "$scenario" a3_conservative \
    '--**.cellularNic.nrPhy.railwayA3TttHandover=true' \
    '--**.cellularNic.nrPhy.railwayA3Hysteresis=5dB' \
    '--**.cellularNic.nrPhy.railwayA3TimeToTrigger=0.48s' \
    '--**.cellularNic.nrPhy.railwayA3MinNeighborRsrp=-130dB'
done

LOCAL_ROOT=/mnt/c/Users/qichengshi/Desktop/RL_sim/results/simu5g/$OUT_ROOT/raw
rm -rf "$LOCAL_ROOT"
mkdir -p "$LOCAL_ROOT"

for d in "results/$OUT_ROOT"/*; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mkdir -p "$LOCAL_ROOT/$name"
  find "$d" -type f \( -name '*.sca' -o -name '*.vec' -o -name '*.vci' \) -exec cp {} "$LOCAL_ROOT/$name/" \;
done

echo "COPIED_TO=$LOCAL_ROOT"
