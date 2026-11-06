# source me:  source run/env.sh
# Local-only ROS setup. Overrides the AIR_Week robot settings from ~/.bashrc
# (ROS_DISCOVERY_SERVER would otherwise send everything to the TurtleBot).
source /opt/ros/jazzy/setup.bash
export PATH="/usr/bin:$PATH"                      # system python 3.12, not brew (see AIR_Week README)
export ROS_OS_OVERRIDE=ubuntu:24.04:noble          # Mint -> rosdep
unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT
export ROS_DOMAIN_ID=${PREN_DOMAIN_ID:-17}
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
PREN_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$PREN_WS/install/setup.bash" ]; then source "$PREN_WS/install/setup.bash"; fi
