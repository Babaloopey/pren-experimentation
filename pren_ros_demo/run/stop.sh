#!/usr/bin/env bash
# Stops everything the demo started: launch (sim + RViz), keyboard, monitor, and any
# orphaned nodes left over from a closed terminal. Terminals/herdr panes stay open.
cd "$(dirname "$0")/.."
source run/env.sh

PATTERN='pren_demo|__node:=(ekf_filter_node|camera_tf|camera_truth_tf)|topic echo /(dog/ack|mission/state|cmd_mux/active)'

list() { pgrep -af -- "$PATTERN" | grep -v -e "run/stop.sh" -e pgrep; }

if [ -z "$(list)" ]; then
    echo "nothing running"
else
    echo "stopping:"
    list | cut -c1-110 | sed 's/^/  /'
    # SIGINT first: ros2 launch shuts its nodes down cleanly
    pkill -INT -f -- "$PATTERN"
    for _ in $(seq 1 20); do
        [ -z "$(list)" ] && break
        sleep 0.25
    done
    if [ -n "$(list)" ]; then
        echo "still alive after 5 s -> SIGTERM"
        pkill -TERM -f -- "$PATTERN"
        sleep 1
        [ -n "$(list)" ] && { echo "-> SIGKILL"; pkill -KILL -f -- "$PATTERN"; }
    fi
fi

# ROS CLI daemon for this domain caches the old graph (stale `ros2 node list`)
ros2 daemon stop >/dev/null 2>&1
echo "done"
