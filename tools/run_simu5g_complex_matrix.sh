#!/usr/bin/env bash
set -euo pipefail

SIM_DIR=${SIM_DIR:-/home/qcshi/simu5g-workspace/simu5g-1.4.4/simulations/nr/railway_handover}
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/c/Users/qichengshi/Desktop/RL_sim}
OUT_ROOT=${1:-RailwayComplexMatrix-$(date +%Y%m%d-%H%M%S)}

if ! command -v simu5g >/dev/null 2>&1; then
  exec opp_env run simu5g-1.4.4 -w /home/qcshi/simu5g-workspace --no-isolated \
    -c "SIM_DIR='$SIM_DIR' PROJECT_ROOT='$PROJECT_ROOT' bash '$PROJECT_ROOT/tools/run_simu5g_complex_matrix.sh' '$OUT_ROOT'"
fi

CONFIGS=(
  Railway-300-DL-TrackOffset-FixedA3
  Railway-300-DL-TrackOffset-RLTable
  Railway-300-DL-WeakCoverage-FixedA3
  Railway-300-DL-WeakCoverage-RLTable
  Railway-300-DL-HighNoise-FixedA3
  Railway-300-DL-HighNoise-RLTable
  Railway-300-DL-NlosFading-FixedA3
  Railway-300-DL-NlosFading-RLTable
  Railway-300-DL-HighInterference-FixedA3
  Railway-300-DL-HighInterference-RLTable
  Railway-300-DL-HeavyTraffic-FixedA3
  Railway-300-DL-HeavyTraffic-RLTable
  Railway-300-DL-Stress-FixedA3
  Railway-300-DL-Stress-RLTable
)

cd "$SIM_DIR"
mkdir -p "results/$OUT_ROOT"

for config in "${CONFIGS[@]}"; do
  out="results/$OUT_ROOT/$config"
  echo "=== RUN $config ==="
  ./run -u Cmdenv -c "$config" --result-dir="$out"
done

local_root="$PROJECT_ROOT/results/simu5g/$OUT_ROOT/raw"
mkdir -p "$local_root"

for d in "results/$OUT_ROOT"/*; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mkdir -p "$local_root/$name"
  find "$d" -type f \( -name '*.sca' -o -name '*.vec' -o -name '*.vci' \) -exec cp {} "$local_root/$name/" \;
done

echo "OUT_ROOT=$OUT_ROOT"
echo "COPIED_TO=$local_root"
