#!/usr/bin/env bash
# Build the simulator of the F1 car (Ibrahim Khaled's ocd_lap package) on a Linux machine: a RunPod pod on the image ros:humble, or the Docker image of docker/. Safe to run again.
#
# What it builds: the pinned TUM Open-Car-Dynamics + the two patches of his repository (four-way dampers, heave springs, packers, road height), compiled into the python module ocd_native.
# What it needs: the BUILD COPY of his repository in $OCD_FIRST_LAP_DIR (default /workspace/ocd_first_lap), i.e. the files of branch sf70h-f1 of the private repository lbrahimKhaled/ocd_first_lap.
# Make it on the laptop without touching his clone (git archive only reads), send it to the pod, unpack it there:
#     git -C /c/Users/NANO/work/ocd_first_lap archive --format=tar.gz --prefix=ocd_first_lap/ -o ocd_first_lap_sf70h-f1.tgz origin/sf70h-f1
#     runpodctl send ocd_first_lap_sf70h-f1.tgz                         (on the pod: runpodctl receive <code>; tar xzf ocd_first_lap_sf70h-f1.tgz -C /workspace)
# This script changes only that copy and the clone of the public Open-Car-Dynamics inside it (vendor/), never his GitHub repository. His files are not in the style-rl repository.
# Then:  export OCD_FIRST_LAP_DIR=/workspace/ocd_first_lap   and use  --car sf70h_f1  (backend ocdlap:sf70h_f1).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
ROOT="${OCD_FIRST_LAP_DIR:-/workspace/ocd_first_lap}"
OCD_PIN=94f8fb187fb0ed22bba1d809bd74f66d1ff75af4          # the commit his patches apply to (scripts/bootstrap.sh of his repository)
[ -d "$ROOT/ocd_lap" ] || { echo "no build copy of ocd_first_lap in $ROOT (see the header of this script)"; exit 1; }
ls "$ROOT"/patches/*.patch >/dev/null

apt-get update -qq
apt-get install -y -qq git build-essential cmake libboost-dev libeigen3-dev python3-dev python3-pip
pip3 install -q "numpy<2" pybind11

cd "$ROOT"
mkdir -p vendor
[ -d vendor/Open-Car-Dynamics/.git ] || git clone --recursive https://github.com/TUMFTM/Open-Car-Dynamics.git vendor/Open-Car-Dynamics
cd vendor/Open-Car-Dynamics
n_patches=$(ls "$ROOT"/patches/*.patch | wc -l)
if [ "$(git branch --show-current)" != "ocd-first-lap" ] || [ "$(git rev-list --count "$OCD_PIN"..HEAD)" != "$n_patches" ]; then
  git checkout -q -B ocd-first-lap "$OCD_PIN"
  for p in "$ROOT"/patches/*.patch; do
    git -c user.name=ocdlap -c user.email=ocdlap@localhost am -q --committer-date-is-author-date "$p"
  done
fi
git submodule update --init --recursive
cd "$ROOT"
# the CMakeLists globs vendor/Open-Car-Dynamics/lib/* as include directories; GCC (unlike clang on his Mac) stops at the one file in it, lib/Readme.md
find "$ROOT/vendor/Open-Car-Dynamics/lib" -maxdepth 1 -type f -delete

if [ ! -f "$ROOT"/ocd_native*.so ]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python3)" -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
  cmake --build build -j "${BUILD_JOBS:-$(nproc)}"
fi
python3 - <<PY
import sys
sys.path.insert(0, "$ROOT")
import ocd_native
from ocd_lap import cars
from ocd_lap.backend import OCDBackend, StartPose, Inputs
b = OCDBackend(powertrain="engine", car="sf70h_f1", ers=False)
s = b.reset(cars.default_setup("sf70h_f1"), StartPose(0.0, 0.0, 0.0, 30.0))
for _ in range(10):
    s = b.step(Inputs(0.0, 1.0, 0.0), steps=20)
print("ocd_native OK; sf70h_f1 after 10 decisions at full throttle from 30 m/s: vx = %.2f m/s, gear %s" % (s["vx"], b.gear))
PY
echo "setup_ocdlap done: export OCD_FIRST_LAP_DIR=$ROOT"
