#!/bin/bash
# One DLIO run on the "best" bag, with a GPS-free divergence stop.
#
#   tools/dlio_test_run.sh <tag> [name:=value ...]
#
# name:=value pairs are DLIO params (e.g. odom/gicp/maxCorrespondenceDistance:=1.0) written
# to a per-run override yaml on top of dlio/dlio_orchard.yaml. Env knobs:
#   RATE (1.0)  START (6, bag-seconds; keep <= 8 so DLIO's IMU calibration sees no motion)
#   STOP (bag-seconds after which playback is cut, default: whole bag)
#   IMU_FIX_ARGS (extra -p args for liosam/imu_fix_quat_node.py)  BAG
# Output: data/dlio/runs/dlio_<tag>_<stamp>/{dlio_odom.csv,params_override.yaml,*.log,dlio_map.pcd,
#         resources.json (CPU s, peak RAM), run_entry.json (starvation warnings, resources)}

TAG=$1; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BAG=${BAG:-"$HOME/bags/session_20260706_140610(best)"}
T0=1783343182.081608   # first GPS fix of the best bag (bag-time zero used by all reports)
OUT="$ROOT/data/dlio/runs/dlio_${TAG}_$(date +%Y%m%d_%H%M%S)"   # <method>_<variant>_<stamp>; older runs: session_..._best_<tag>_<stamp>
mkdir -p "$OUT"
source /opt/ros/humble/setup.bash; source "$HOME/ros2_ws/install/setup.bash"

{ echo "/**:"; echo "  ros__parameters:"; echo "    use_sim_time: true"; for kv in "$@"; do echo "    ${kv%%:=*}: ${kv#*:=}"; done; } > "$OUT/params_override.yaml"
echo "rate=${RATE:-1.0} start=${START:-6} stop=${STOP:-} imu_fix_args=${IMU_FIX_ARGS:-}" > "$OUT/run_info.txt"

LARGS=(params_override:="$OUT/params_override.yaml")
[ -n "${IMU_FIX_ARGS:-}" ] && LARGS+=(imu_fix_args:="$IMU_FIX_ARGS")
ros2 launch "$ROOT/dlio/dlio_orchard.launch.py" "${LARGS[@]}" > "$OUT/dlio.log" 2>&1 & LP=$!
sleep 5
python3 -u "$ROOT/trajectories/recorders/odometry_recorder.py" "$OUT/dlio_odom.csv" > "$OUT/rec.log" 2>&1 & RP=$!
sleep 2
python3 "$ROOT/tools/proc_sampler.py" 'dlio_(odom|map)_node' "$OUT" > "$OUT/sampler.log" 2>&1 & SMP=$!
ros2 bag play "$BAG" --clock --rate "${RATE:-1.0}" --start-offset "${START:-6}" \
  --topics /velodyne_points /camera/camera/imu > "$OUT/bag.log" 2>&1 & BP=$!
while kill -0 $BP 2>/dev/null; do
  sleep 5
  pgrep dlio_odom_node > /dev/null || { echo "dlio_odom_node died" | tee -a "$OUT/run_info.txt"; kill -INT $BP; break; }
  L=$(tail -1 "$OUT/dlio_odom.csv" 2>/dev/null | cut -d, -f1)
  [ -n "${STOP:-}" ] && [ -n "$L" ] && [ "$L" != "t" ] && \
    python3 -c "import sys; sys.exit(0 if float('$L')-$T0 > ${STOP:-1e9} else 1)" && { kill -INT $BP; break; }
done
sleep 3
ros2 service call /save_pcd direct_lidar_inertial_odometry/srv/SavePCD "{leaf_size: 0.2, save_path: '$OUT'}" > "$OUT/save.log" 2>&1 &
SP=$!; sleep 20; kill $SP 2>/dev/null
kill -INT $SMP; kill -INT $RP; sleep 1; kill -INT $LP; sleep 4; kill -9 $LP 2>/dev/null
pkill -9 dlio_odom_node; pkill -9 dlio_map_node; pkill -f "[i]mu_fix_quat_node.py"
python3 - "$OUT" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
res = json.loads((d / 'resources.json').read_text()) if (d / 'resources.json').exists() else None
info = dict(run_dir=str(d), starvation_warnings=(d / 'bag.log').read_text(errors='ignore').count('starved'),
            resources=res, died='died' in (d / 'run_info.txt').read_text())
(d / 'run_entry.json').write_text(json.dumps(info, indent=2))
PY
echo "$OUT"
