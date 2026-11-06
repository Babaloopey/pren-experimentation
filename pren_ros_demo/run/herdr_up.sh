#!/usr/bin/env bash
# Opens a "pren-ros-demo" tab in the current herdr workspace:
#   sim (launch + RViz) | keyboard driver
#   monitor             | shell
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
J() { python3 -c "import sys,json; d=json.load(sys.stdin)['result']; print($1)"; }
TAB_JSON=$(herdr tab create --workspace "${HERDR_WORKSPACE_ID:?run inside herdr}" --cwd "$WS" --label pren-ros-demo --no-focus)
SIM=$(echo "$TAB_JSON" | J "d['root_pane']['pane_id']")
KEY=$(herdr pane split "$SIM" --direction right --cwd "$WS" --no-focus | J "d['pane']['pane_id']")
MON=$(herdr pane split "$SIM" --direction down --cwd "$WS" --no-focus | J "d['pane']['pane_id']")
SH=$(herdr pane split "$KEY" --direction down --cwd "$WS" --no-focus | J "d['pane']['pane_id']")
herdr pane rename "$SIM" sim >/dev/null
herdr pane rename "$KEY" keyboard >/dev/null
herdr pane rename "$MON" monitor >/dev/null
herdr pane rename "$SH" shell >/dev/null
herdr pane run "$SIM" "run/sim.sh"
sleep 3
herdr pane run "$KEY" "run/keyboard.sh"
herdr pane run "$MON" "run/monitor.sh"
herdr pane run "$SH" "source run/env.sh && ros2 node list"
echo "tab ready: sim=$SIM keyboard=$KEY monitor=$MON shell=$SH"
