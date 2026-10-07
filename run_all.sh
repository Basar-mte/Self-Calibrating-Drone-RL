#!/usr/bin/env bash
# Train every agent in the comparison: 12 runs, 4 at a time (about 3 hours on a laptop GPU).
#   bash run_all.sh            then: python evaluate.py --runs "results/sac_*" --pid
set -u
PY=${PYTHON:-python}
JOBS=${JOBS:-4}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
queue=(
  "sac_calib_s0   0 observation.mode=calib"
  "sac_blind_s0   0 observation.mode=base"
  "sac_history_s0 0 observation.mode=history"
  "sac_oracle_s0  0 observation.mode=oracle"
  "sac_nominal_s0 0 observation.mode=base uncertainty.enabled=false"
  "sac_calib_s1   1 observation.mode=calib"
  "sac_blind_s1   1 observation.mode=base"
  "sac_history_s1 1 observation.mode=history"
  "sac_oracle_s1  1 observation.mode=oracle"
  "sac_calib_s2   2 observation.mode=calib"
  "sac_blind_s2   2 observation.mode=base"
  "sac_history_s2 2 observation.mode=history"
)
run_one() {
  set -- $1
  name=$1; seed=$2; shift 2
  echo "start $name $(date +%H:%M)"
  $PY -u train.py --run-name "$name" --seed "$seed" --set "$@" > "results/$name.log" 2>&1
  echo "end   $name $(date +%H:%M) (exit $?)"
}
export -f run_one
export PY
mkdir -p results
printf '%s\n' "${queue[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "{}"'
echo "all runs finished $(date +%H:%M)"
