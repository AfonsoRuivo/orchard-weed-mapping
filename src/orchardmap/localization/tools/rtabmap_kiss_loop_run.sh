#!/bin/bash
# One RTAB-Map (loop closure on) + tuned KISS-ICP run, rtabmap/rtabmap_kissicp_loop.launch.py, whole bag
# (all topics, so /gnss/fix reaches RTAB-Map as in the 2026-10-04 run). Rebuilt 2026-10-08 from the
# 2026-10-04 session's rtab_loop_run.sh (lost from /tmp), plus CPU/RAM sampling of KISS-ICP (+ground filter)
# and RTAB-Map (+map_assembler) separately.
#   tools/rtabmap_kiss_loop_run.sh <tag>        Env: RATE (1.0)  START (10)  BAG
# Output: data/rtabmap/runs/rtabmap_<tag>_<stamp>/{kiss_odom.csv,rtabmap_mappath.csv,rtabmap_kissicp_loop.db,
#         res_kiss/,res_rtab/ (resources.{csv,json}),run_entry.json,*.log}
TAG=$1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BAG=${BAG:-"$HOME/bags/session_20260706_140610(best)"}
OUT="$ROOT/data/rtabmap/runs/rtabmap_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/res_kiss" "$OUT/res_rtab"
source /opt/ros/humble/setup.bash; source "$HOME/ros2_ws/install/setup.bash"
cd "$ROOT"
cp rtabmap/rtabmap_kissicp_loop.launch.py "$OUT/launch_used.py"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
setsid ros2 launch rtabmap/rtabmap_kissicp_loop.launch.py use_foxglove:=false > "$OUT/launch.log" 2>&1 & LP=$!
sleep 15
python3 -u trajectories/recorders/kissicp_odometry_recorder.py "$OUT/kiss_odom.csv" > "$OUT/kiss_recorder.log" 2>&1 & K=$!
python3 -u trajectories/recorders/trajectory_recorder.py "$OUT/rtabmap_mappath.csv" > "$OUT/rtab_recorder.log" 2>&1 & T=$!
python3 tools/proc_sampler.py 'kiss_icp/kiss_icp_node|cloud_filter_node' "$OUT/res_kiss" > /dev/null 2>&1 & S1=$!
python3 tools/proc_sampler.py 'rtabmap_slam/rtabmap|rtabmap_util/map_assembler' "$OUT/res_rtab" > /dev/null 2>&1 & S2=$!
sleep 5
log "playing bag rate ${RATE:-1.0} offset ${START:-10}: $BAG"
ros2 bag play "$BAG" --clock --rate "${RATE:-1.0}" --start-offset "${START:-10}" > "$OUT/bag_play.log" 2>&1
log "bag finished (exit $?), settling"
sleep 15
kill -INT $S1 $S2 $K $T; sleep 5
kill -INT -- -$LP 2>/dev/null; sleep 15
kill -TERM -- -$LP 2>/dev/null; sleep 3
cp /tmp/rtabmap_kissicp_loop.db "$OUT/" 2>/dev/null && log "copied db"
python3 - "$OUT" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
r = lambda s: json.loads((d / s / 'resources.json').read_text()) if (d / s / 'resources.json').exists() else None
info = dict(run_dir=str(d), starvation_warnings=(d / 'bag_play.log').read_text(errors='ignore').count('starved'),
            kiss_odom_msgs=max(0, sum(1 for _ in open(d / 'kiss_odom.csv')) - 1) if (d / 'kiss_odom.csv').exists() else 0,
            resources_kiss=r('res_kiss'), resources_rtabmap=r('res_rtab'))
(d / 'run_entry.json').write_text(json.dumps(info, indent=2))
PY
log "DONE"
echo "$OUT"
