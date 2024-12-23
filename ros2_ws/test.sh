#!/bin/bash

cd ~/street-gaussians-ns/ros2_ws
colcon build
source install/setup.bash
ros2 run sgn_pkg sgn_node
