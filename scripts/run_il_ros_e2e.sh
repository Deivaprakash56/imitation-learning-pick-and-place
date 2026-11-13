#!/bin/bash
# Project 3, real-arm path without hardware:
#   fake_arm + scripted teleop  ->  demo_recorder (HDF5)  ->  train_bc  ->  policy_runner  ->  success rate
# Exercises exactly the nodes you will run on the real arm; only the arm itself is fake.
# Usage: scripts/run_il_ros_e2e.sh [n_demos] [n_test] [teleop_mode: state|timer]
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
N_DEMOS=${1:-40}
N_TEST=${2:-20}
MODE=${3:-state}
source "$WS/env.sh" >/dev/null
export ROS_DOMAIN_ID=${SIM_DOMAIN_ID:-78}     # isolated from other ROS sessions on this machine
OUT="$WS/data/fake_arm_demos_${3:-state}.hdf5"
RUN="$WS/runs/bc_fake_arm_$MODE"
LOG="$WS/logs/p3_ros_e2e"
mkdir -p "$LOG" "$WS/data"
rm -f "$OUT"
JOINTS='[joint1,joint2,joint3,gripper]'
PIDS=()
NODES="$WS/install/il_arm/lib/il_arm/"      # `ros2 run` wrappers don't forward signals - kill the nodes by path
cleanup() { pkill -f "$NODES" 2>/dev/null || true; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
cleanup; sleep 1                             # leftovers from an interrupted run would answer our services

ros2 run il_arm fake_arm --ros-args -p teleop_mode:=$MODE > "$LOG/fake_arm.log" 2>&1 & PIDS+=($!)
ros2 run il_arm demo_recorder --ros-args -p out:="$OUT" -p joint_names:="$JOINTS" \
    -p action_topic:=/teleop/joint_command -p object_pose_topic:=/object_pose -p rate_hz:=15.0 \
    > "$LOG/recorder.log" 2>&1 & REC=$!; PIDS+=($REC)
sleep 3

echo "== recording $N_DEMOS teleop demos"
python -m il_arm.e2e_driver record --n "$N_DEMOS"
pkill -INT -f "${NODES}demo_recorder"; sleep 2

echo "== training BC on the recorded demos"
python -m il_arm.train_bc "$OUT" --chunk 3 --epochs 400 --out "$RUN" --threads 4 2>&1 | grep -E "train demos|saved"

echo "== running the policy on $N_TEST new objects"
ros2 run il_arm policy_runner --ros-args -p policy:="$RUN/best.pt" -p joint_names:="$JOINTS" \
    -p object_pose_topic:=/object_pose -p command_topic:=/arm_controller/joint_trajectory -p rate_hz:=15.0 \
    -p max_joint_step:=0.1 > "$LOG/runner.log" 2>&1 & PIDS+=($!)
sleep 3
python -m il_arm.e2e_driver run --n "$N_TEST" | tee "$LOG/run.txt"
grep RESULT "$LOG/run.txt" | awk '{print $3","$5}' > "$RUN/ros_e2e_result.csv"
