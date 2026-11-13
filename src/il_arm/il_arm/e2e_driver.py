"""Drives the fake-arm end-to-end test through the real recorder / runner services.

  python -m il_arm.e2e_driver record --n 40     # fake_arm + demo_recorder must be running
  python -m il_arm.e2e_driver run --n 20        # fake_arm + policy_runner must be running

One persistent node with service clients (the `ros2 service call` CLI starts a new process per
call, which is ~1 s of CPU each).
"""
import argparse
import sys
import time

import rclpy
from std_srvs.srv import Trigger


class Driver:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node('il_e2e_driver')
        self.clients = {}

    def call(self, name, timeout=30.0):
        if name not in self.clients:
            self.clients[name] = self.node.create_client(Trigger, name)
        c = self.clients[name]
        if not c.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f'service {name} not available')
        fut = c.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        if not fut.done():
            raise RuntimeError(f'{name} timed out')
        return fut.result()

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['record', 'run'])
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--run-seconds', type=float, default=4.0)
    args = ap.parse_args()
    d = Driver()
    ok = 0
    for i in range(args.n):
        d.call('/fake_arm/reset')
        d.spin_for(0.3)
        if args.mode == 'record':
            d.call('/demo_recorder/start')
            d.call('/fake_arm/teleop')
            success = d.call('/fake_arm/check').success
            d.call('/demo_recorder/stop_success' if success else '/demo_recorder/stop_fail')
        else:
            d.call('/policy_runner/start')
            d.spin_for(args.run_seconds)
            d.call('/policy_runner/stop')
            r = d.call('/fake_arm/check')
            success = r.success
            print(f'   test {i + 1}: {r.message}', flush=True)
        ok += success
        if args.mode == 'record' and (i + 1) % 10 == 0:
            print(f'   {i + 1} demos recorded ({ok} successful)', flush=True)
    print(f'RESULT {args.mode}: {ok} / {args.n}', flush=True)
    rclpy.try_shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
