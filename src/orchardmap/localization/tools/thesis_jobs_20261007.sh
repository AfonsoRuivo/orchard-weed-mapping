#!/bin/bash
# Sequential queue for the 2026-10-07 thesis job list (J1-J8). One heavy job at a time;
# a failing job is logged and the queue moves on.
#
#   tools/thesis_jobs_20261007.sh [first_job]      (e.g. J5 to resume from J5)
#
# Everything lands in data/thesis_jobs_20261007/: queue.log, jobs.tsv (job, status, run dir),
# entropy/, trunks/, exports/, and the pack for_thesis_20261007/ (+ .zip). If
# data/thesis_jobs_20261007/jobs_extra.sh exists when the main queue ends, it is run too (J10).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BAG="$HOME/bags/session_20260706_140610(best)"
JD="$ROOT/data/thesis_jobs_20261007"
mkdir -p "$JD"/{entropy,trunks,exports}
LOG="$JD/queue.log"; TSV="$JD/jobs.tsv"
[ -f "$TSV" ] || printf 'job\tstatus\trun_dir\tstarted\tfinished\n' > "$TSV"
source /opt/ros/humble/setup.bash; source "$HOME/ros2_ws/install/setup.bash"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
stamp() { date +%Y%m%d_%H%M%S; }
START_AT=${1:-J1}; started=0
want() { [ "$1" = "$START_AT" ] && started=1; [ $started = 1 ]; }
ROSPAT='lio_sam_|dlio_(odom|map)_node|kiss_icp_node|imu_fix_quat_node|_recorder\.py|proc_sampler|ros2 (bag|launch|run|topic|service)|static_transform_publisher|ros2-daemon'
cleanup_ros() {
  # leftovers from a crashed job must not starve the next one. SIGINT first: processes killed
  # with -9 can leave Fast DDS port mutexes in /dev/shm locked, after which every new ROS node
  # hangs at creation (happened on 2026-10-07: J1's recorders never started).
  pkill -INT -f "$ROSPAT"; sleep 6; pkill -9 -f "$ROSPAT"; sleep 2
  if ! pgrep -f "$ROSPAT" > /dev/null; then rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*; fi
  if ! timeout -s KILL 30 python3 "$ROOT/tools/ros_node_probe.py" > /dev/null 2>&1; then
    log "WARNING: ROS node creation still hangs after cleanup"
  fi
}
record() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$(date '+%F %T')" >> "$TSV"; }

liosam_job() {  # <job> <variant id>
  local job=$1 v=$2 out="$ROOT/data/lio_sam/runs/liosam_${2}_$(stamp)" t0
  t0=$(date '+%F %T'); log "$job start: LIO-SAM $v -> $out"
  python3 "$ROOT/tools/liosam_param_sweep.py" --bag "$BAG" --only "$v" --flat --out "$out" \
    --rate 0.5 --start-offset 10 > "$JD/${job}_stdout.log" 2>&1
  local rc=$?; cleanup_ros
  if [ $rc = 0 ] && [ -s "$out/trajectory.csv" ]; then st=ok; else st="failed(rc=$rc)"; fi
  log "$job end: $st $(cat "$out/export.log" 2>/dev/null)"; record "$job" "$st" "$out" "$t0"
}

mkdir -p "$ROOT/data/lio_sam/runs"
cleanup_ros
want J1 && liosam_job J1 baseline_loops
want J2 && liosam_job J2 loop_closure_off
want J3 && liosam_job J3 imu_rot_perm_tilt
want J4 && liosam_job J4 imu_rot_camlidar

if want J5; then
  t0=$(date '+%F %T'); log "J5 start: DLIO full repeat"
  d=$(RATE=1.0 START=6 "$ROOT/tools/dlio_test_run.sh" rep3 2>>"$JD/J5_stdout.log" | tail -1); cleanup_ros
  if [ -s "$d/dlio_odom.csv" ] && ! grep -q died "$d/run_info.txt"; then st=ok; else st=failed; fi
  log "J5 end: $st $d"; record J5 "$st" "$d" "$t0"
fi

# final optimized keyframes of the older LIO-SAM runs (written to exports/, old run dirs untouched)
SW="$ROOT/data/lio_sam/sweeps"
REP1="$SW/session_20260706_140610_best_6run_r05_20261004_220758/baseline_rep1"
REP2="$SW/session_20260706_140610_best_6run_r05_20261004_220758/baseline_rep2"
BASE="$SW/session_20260706_140610_best_rli_calib_20261004_142635/baseline"
for pair in "rep1:$REP1" "rep2:$REP2" "baseline:$BASE"; do
  [ -f "$JD/exports/liosam_${pair%%:*}/keyframes_final.csv" ] || \
    python3 "$ROOT/tools/liosam_export.py" "${pair#*:}" "$JD/exports/liosam_${pair%%:*}" >> "$LOG" 2>&1
done
jobdir() { awk -F'\t' -v j="$1" '$1==j && $2=="ok" {d=$3} END {print d}' "$TSV"; }
KISS="$ROOT/data/kiss_icp/runs/session_20260706_140610_best_tuned_rep1_20261005_093554/kiss_odom.csv"
DLIO="$ROOT/data/dlio/runs/session_20260706_140610_best_base_full_20261006_172420/dlio_odom.csv"

if want J6; then
  t0=$(date '+%F %T'); log "J6 start: map entropy"
  args=("liosam_rep1=$JD/exports/liosam_rep1/keyframes_final.csv")
  for j in J2 J3 J4; do d=$(jobdir $j); [ -n "$d" ] && args+=("liosam_${j}=$d/keyframes_final.csv"); done
  args+=("kiss_tuned_rep1=$KISS" "dlio_base_full=$DLIO")
  python3 "$ROOT/tools/proc_sampler.py" "map_entropy" "$JD/entropy" > /dev/null 2>&1 & SMP=$!
  python3 "$ROOT/analysis/map_entropy.py" --every 5 --out-dir "$JD/entropy" --png-dir "$JD/entropy" "${args[@]}" \
    2>&1 | grep -v "INFO" >> "$JD/J6_stdout.log"
  rc=${PIPESTATUS[0]}; kill -INT $SMP
  [ $rc = 0 ] && st=ok || st="failed(rc=$rc)"; log "J6 end: $st"; record J6 "$st" "$JD/entropy" "$t0"
fi

if want J7; then
  t0=$(date '+%F %T')
  PICKS=$(find "$HOME" -name picks_sv_fixed.csv -not -path "*/.cache/*" 2>/dev/null | head -1)
  if [ -z "$PICKS" ]; then
    log "J7 skipped: picks_sv_fixed.csv not found under $HOME"; record J7 "skipped(no picks file)" "" "$t0"
  else
    log "J7 start: trunk transfer, picks=$PICKS"; cp "$PICKS" "$JD/trunks/"
    args=("gmap_run=$ROOT/data/campinas_b/lio_sam/trajectory_20260706.csv"
          "liosam_rep1=$JD/exports/liosam_rep1/keyframes_final.csv")
    d=$(jobdir J2); [ -n "$d" ] && args+=("liosam_J2=$d/keyframes_final.csv")
    args+=("kiss_tuned_rep1=$KISS" "dlio_base_full=$DLIO")
    python3 "$ROOT/analysis/trunk_transfer_export.py" --picks "$PICKS" --out-dir "$JD/trunks" "${args[@]}" \
      2>&1 | grep -v INFO >> "$JD/J7_stdout.log"; rc=${PIPESTATUS[0]}
    big=$(find "$JD/trunks" -name 'dense_trunks_*.npz' -size +60M | wc -l)
    if [ $rc = 0 ] && [ "$big" -gt 0 ]; then
      log "J7: $big npz > 60 MB, redoing with every 2nd scan"; mkdir -p "$JD/trunks/every1_oversize"
      mv "$JD"/trunks/dense_trunks_*.npz "$JD/trunks/every1_oversize/"
      python3 "$ROOT/analysis/trunk_transfer_export.py" --every 2 --picks "$PICKS" --out-dir "$JD/trunks" "${args[@]}" \
        2>&1 | grep -v INFO >> "$JD/J7_stdout.log"; rc=${PIPESTATUS[0]}
    fi
    [ $rc = 0 ] && st=ok || st="failed(rc=$rc)"; log "J7 end: $st"; record J7 "$st" "$JD/trunks" "$t0"
  fi
fi

if want J8 || [ $started = 1 ]; then
  log "J8: building pack"
  python3 "$ROOT/tools/thesis_pack_20261007.py" >> "$LOG" 2>&1 && log "J8 end: ok" || log "J8 end: failed"
fi

if [ -x "$JD/jobs_extra.sh" ]; then log "extra jobs start"; "$JD/jobs_extra.sh" >> "$LOG" 2>&1; log "extra jobs end"; fi
log "queue finished"
