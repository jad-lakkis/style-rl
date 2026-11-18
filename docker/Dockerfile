# Open-Car-Dynamics + the Python stack of the driving environment, for a local machine (Docker Desktop on Windows/Mac/Linux). Same recipe as runpod/setup_pod.sh.
#   docker build -t ocd-drive -f docker/Dockerfile docker/
#   docker run --rm -it -v <path to this repository>:/workspace/style-rl ocd-drive python3 scripts/drive_smoke.py --backend ocd
FROM ros:humble
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update -qq && apt-get install -y -qq git build-essential cmake libboost-dev libeigen3-dev python3-pip python3-colcon-common-extensions \
        ros-humble-can-msgs ros-humble-ros2-socketcan ros-humble-geographic-msgs && rm -rf /var/lib/apt/lists/*
# Open-Car-Dynamics (TUM), built once into /opt/Open-Car-Dynamics/install. OCD_REF pins a commit or tag (empty = the default branch).
ARG OCD_REF=
RUN git clone --recursive https://github.com/TUMFTM/Open-Car-Dynamics.git /opt/Open-Car-Dynamics \
 && cd /opt/Open-Car-Dynamics && if [ -n "$OCD_REF" ]; then git checkout "$OCD_REF" && git submodule update --init --recursive; fi \
 && git rev-parse HEAD > /opt/ocd_commit.txt
RUN bash -c "source /opt/ros/humble/setup.bash && cd /opt/Open-Car-Dynamics && colcon build --merge-install --packages-up-to ocd_vehicle_models_py --cmake-args -DCMAKE_BUILD_TYPE=Release" \
 && rm -rf /opt/Open-Car-Dynamics/build /opt/Open-Car-Dynamics/log
RUN pip3 install "numpy<2" torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
 && pip3 install stable-baselines3 gymnasium pandas scipy tensorboard pytest matplotlib
COPY entrypoint.sh /entrypoint.sh
RUN sed -i "s/$//" /entrypoint.sh && chmod +x /entrypoint.sh
WORKDIR /workspace/style-rl
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
