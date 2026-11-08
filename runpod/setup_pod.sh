#!/usr/bin/env bash
# Run on a RunPod pod that uses the Docker image ros:humble (Ubuntu 22.04 with ROS 2). Installs the system packages, builds Open-Car-Dynamics once (it stays on the volume) and installs the Python packages. Safe to run again.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # absolute, because the build below changes directory
OCD_DIR="${OCD_DIR:-/workspace/Open-Car-Dynamics}"

# the container disk is reset when the pod restarts (only /workspace survives), so the system packages are installed every time
apt-get update -qq
apt-get install -y -qq git build-essential cmake libboost-dev libeigen3-dev python3-pip python3-venv python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-can-msgs ros-${ROS_DISTRO}-ros2-socketcan ros-${ROS_DISTRO}-geographic-msgs
export PIP_CACHE_DIR=/workspace/pip-cache   # on the volume: a restart does not download torch again

if [ ! -f "$OCD_DIR/install/local_setup.sh" ]; then
  [ -d "$OCD_DIR" ] || git clone --recursive https://github.com/TUMFTM/Open-Car-Dynamics.git "$OCD_DIR"
  cd "$OCD_DIR"
  set +u   # the ROS setup scripts read variables that are not set
  source /opt/ros/${ROS_DISTRO}/setup.bash
  set -u
  colcon build --merge-install --packages-up-to ocd_vehicle_models_py --cmake-args -DCMAKE_BUILD_TYPE=Release
fi
set +u
source "$OCD_DIR/install/local_setup.sh"
set -u
python3 -c "import ocd_vehicle_models_py; print('OCD python bindings import OK')"
if [ -e /dev/nvidiactl ]; then
  echo "GPU found: installing the CUDA build of torch"
  pip3 install "numpy<2" torch
else
  pip3 install "numpy<2" torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
fi
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
pip3 install -r "$HERE/../requirements.txt"
echo "setup done"
