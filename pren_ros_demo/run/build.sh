#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source run/env.sh
colcon build --symlink-install "$@"
