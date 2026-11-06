#!/usr/bin/env bash
# Live view of what the dog "thinks": mission state + acks + who is driving.
cd "$(dirname "$0")/.."
source run/env.sh
echo_topic() { ros2 topic echo "$1" std_msgs/msg/String --field data | grep --line-buffered -v '^---$' | sed -u "s|^|$2 |" | uniq; }
echo_topic /dog/ack        "ACK    " &
echo_topic /cmd_mux/active "DRIVER " &
echo_topic /mission/state  "MISSION"
