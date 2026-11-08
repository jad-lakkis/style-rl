#!/usr/bin/env bash
# Run once on a RunPod pod that uses the Docker image ros:humble (Ubuntu 22.04 with ROS 2). Builds Open-Car-Dynamics and installs the Python packages. Safe to run again.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
OCD_DIR="${OCD_DIR:-/workspace/Open-Car-Dynamics}"
if [ ! -f "$OCD_DIR/install/local_setup.sh" ]; then
  apt-get update
  apt-get install -y git build-essential cmake libboost-dev libeigen3-dev python3-pip python3-venv python3-colcon-common-extensions \
      ros-${ROS_DISTRO}-can-msgs ros-${ROS_DISTRO}-ros2-socketcan ros-${ROS_DISTRO}-geographic-msgs
  [ -d "$OCD_DIR" ] || git clone --recursive https://github.com/TUMFTM/Open-Car-Dynamics.git "$OCD_DIR"
  cd "$OCD_DIR"
  source /opt/ros/${ROS_DISTRO}/setup.bash
  colcon build --merge-install --packages-up-to ocd_vehicle_models_py --cmake-args -DCMAKE_BUILD_TYPE=Release
fi
source "$OCD_DIR/install/local_setup.sh"
python3 -c "import ocd_vehicle_models_py; print('OCD python bindings import OK')"
if [ -e /dev/nvidiactl ]; then
  echo "GPU found: installing the CUDA build of torch"
  pip3 install --no-cache-dir "numpy<2" torch
else
  pip3 install --no-cache-dir "numpy<2" torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
fi
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
pip3 install --no-cache-dir -r "$(dirname "$0")/../requirements.txt"
echo "setup done"
