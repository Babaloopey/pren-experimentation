#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source run/env.sh
exec ros2 run pren_demo keyboard_teleop
