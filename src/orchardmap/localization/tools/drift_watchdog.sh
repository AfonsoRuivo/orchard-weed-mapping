#!/bin/bash
# Polls the currently-running OFAT sweep variant every 40s and kills its
# bag-play process if the trajectory drifts outside sane bounds, so the
# sweep driver's own error-handling advances to the next variant instead of
# burning ~20 more minutes on a run that's already diverged.
set -u
SWEEP_ROOT="$1"
DRIVER_PID="$2"
BAG_PATH="$3"
# Loose on purpose: this only needs to catch genuine blow-ups (runs have been
# seen exploding to z=500-1000m), not flag ordinary drift. The trajectory vs.
# GPS/point-cloud comparison done afterward is what actually judges accuracy,
# so a run that merely drifted a modest amount should be left to finish.
XY_LIMIT="${XY_LIMIT:-250}"
Z_LIMIT="${Z_LIMIT:-30}"
LOG="$SWEEP_ROOT/drift_watchdog.log"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "$(ts) watchdog started, watching $SWEEP_ROOT, driver pid=$DRIVER_PID" >> "$LOG"

while kill -0 "$DRIVER_PID" 2>/dev/null; do
  # find the active variant: newest subdir not yet present in manifest.json
  active=""
  for d in "$SWEEP_ROOT"/*/; do
    id=$(basename "$d")
    [ "$id" = "manifest.json" ] && continue
    if [ -f "$SWEEP_ROOT/manifest.json" ] && grep -q "\"id\": \"$id\"" "$SWEEP_ROOT/manifest.json" 2>/dev/null; then
      continue
    fi
    active="$d"
  done

  if [ -n "$active" ] && [ -f "$active/trajectory.csv" ]; then
    last=$(tail -1 "$active/trajectory.csv")
    x=$(echo "$last" | cut -d, -f2)
    y=$(echo "$last" | cut -d, -f3)
    z=$(echo "$last" | cut -d, -f4)
    if [ -n "$x" ] && [ "$x" != "x" ]; then
      diverged=$(python3 -c "
x,y,z = $x, $y, $z
print(1 if (abs(x) > $XY_LIMIT or abs(y) > $XY_LIMIT or abs(z) > $Z_LIMIT) else 0)
" 2>/dev/null)
      echo "$(ts) $(basename "$active") x=$x y=$y z=$z diverged=$diverged" >> "$LOG"
      if [ "$diverged" = "1" ]; then
        echo "$(ts) DRIFT DETECTED in $(basename "$active") - killing bag play" >> "$LOG"
        # pkill -f treats BAG_PATH as a regex, so paths containing
        # parentheses (seen in real bag names, e.g. "session_...(best)")
        # silently fail to match - kill by exact PID instead, found via a
        # literal (non-regex) substring match on each candidate's cmdline.
        for pid in $(pgrep -f "ros2 bag play"); do
          cmdline=$(ps -p "$pid" -o args= 2>/dev/null)
          case "$cmdline" in
            *"ros2 bag play $BAG_PATH"*) kill -9 "$pid" 2>/dev/null ;;
          esac
        done
      fi
    fi
  fi
  sleep 40
done

echo "$(ts) watchdog exiting, driver pid $DRIVER_PID no longer running" >> "$LOG"
