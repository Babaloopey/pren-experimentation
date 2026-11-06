#!/usr/bin/env bash
# Starts sim + EKF + fake perception + mission + RViz. Extra args go to ros2 launch,
# e.g.  run/sim.sh n_bones:=3 seed:=42 voice_miss_rate:=0.2
cd "$(dirname "$0")/.."
source run/env.sh
exec ros2 launch pren_demo demo.launch.py "$@"
