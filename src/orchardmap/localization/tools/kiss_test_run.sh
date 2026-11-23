#!/bin/bash
# One standalone KISS-ICP run on a bag with the TUNED configuration (report 2026-10-04/05 §4.3):
# ground filter kiss_icp/cloud_filter_node.py (drop z < -0.45 m, sensor frame) -> /velodyne_points_filtered,
# deskew off, voxel 0.5, max 20 pts/voxel, range 1-60 m, initial threshold 0.5, min motion 0.1, 500 iterations,
# 4 threads. Rebuilt 2026-10-08 from the 2026-10-04 session's kiss_only.sh/full_run.sh (lost from /tmp), plus
# CPU/RAM sampling.
#
#   tools/kiss_test_run.sh <tag> [-p name:=value ...]      (extra -p override the tuned values)
# Env: RATE (1.0)  START (10 bag-s)  BAG  ZMIN (-0.45; empty = no ground filter)
# Output: data/kiss_icp/runs/kiss_<tag>_<stamp>/{kiss_odom.csv,params.txt,*.log,resources.{csv,json},run_entry.json}
TAG=$1; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BAG=${BAG:-"$HOME/bags/session_20260706_140610(best)"}
ZMIN=${ZMIN--0.45}
OUT="$ROOT/data/kiss_icp/runs/kiss_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
source /opt/ros/humble/setup.bash; source "$HOME/ros2_ws/install/setup.bash"
cd "$ROOT"
K=(-p data.max_range:=60.0 -p data.min_range:=1.0 -p mapping.voxel_size:=0.5 -p mapping.max_points_per_voxel:=20
   -p adaptive_threshold.initial_threshold:=0.5 -p adaptive_threshold.min_motion_th:=0.1
   -p registration.max_num_iterations:=500 -p data.deskew:=false "$@")
echo "params: ${K[*]}" > "$OUT/params.txt"; echo "bag=$BAG rate=${RATE:-1.0} start=${START:-10}" >> "$OUT/params.txt"
TOPIC=/velodyne_points; FP=""
if [ -n "$ZMIN" ]; then
  python3 kiss_icp/cloud_filter_node.py --ros-args -p use_sim_time:=true -p z_min:=$ZMIN -p r_min:=0.0 > "$OUT/filter.log" 2>&1 & FP=$!
  TOPIC=/velodyne_points_filtered; echo "filter z_min=$ZMIN r_min=0.0" >> "$OUT/params.txt"
fi
ros2 run kiss_icp kiss_icp_node --ros-args -p use_sim_time:=true -p base_frame:=velodyne -p lidar_odom_frame:=odom \
  -p publish_odom_tf:=true -p publish_debug_clouds:=false -p registration.max_num_threads:=4 "${K[@]}" \
  -r pointcloud_topic:=$TOPIC > "$OUT/kiss.log" 2>&1 & KP=$!
sleep 4
python3 -u trajectories/recorders/kissicp_odometry_recorder.py "$OUT/kiss_odom.csv" > "$OUT/rec.log" 2>&1 & RP=$!
sleep 2
python3 tools/proc_sampler.py 'kiss_icp_node|cloud_filter_node' "$OUT" > "$OUT/sampler.log" 2>&1 & SMP=$!
ros2 bag play "$BAG" --clock --rate "${RATE:-1.0}" --start-offset "${START:-10}" --topics /velodyne_points > "$OUT/bag.log" 2>&1
sleep 3
kill -INT $SMP; [ -n "$FP" ] && kill -INT $FP; kill -INT $RP; sleep 2; kill -INT $KP; sleep 2
ps -eo pid,args | awk '$2 ~ /kiss_icp_node$/ {print $1}' | xargs -r kill -9
python3 - "$OUT" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
res = json.loads((d / 'resources.json').read_text()) if (d / 'resources.json').exists() else None
info = dict(run_dir=str(d), starvation_warnings=(d / 'bag.log').read_text(errors='ignore').count('starved'),
            odom_msgs=max(0, sum(1 for _ in open(d / 'kiss_odom.csv')) - 1) if (d / 'kiss_odom.csv').exists() else 0,
            resources=res)
(d / 'run_entry.json').write_text(json.dumps(info, indent=2))
PY
echo "$OUT"
