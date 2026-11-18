#!/usr/bin/env bash
# sources ROS and the built OCD, then runs the command (the ROS / OCD setup scripts read unset variables, so no `set -u`)
source /opt/ros/humble/setup.bash
source /opt/Open-Car-Dynamics/install/local_setup.sh
exec "$@"
