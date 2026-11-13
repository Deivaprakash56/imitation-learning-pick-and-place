"""Collect scripted-expert demonstrations in simulation.

DART-style noise injection: the *executed* action is perturbed, but the *recorded* label is the
expert's clean action at the perturbed state. This teaches the policy to recover from its own drift,
which is the main failure mode of plain behaviour cloning.

  python -m il_arm.collect_demos --episodes 300 --out data/fetch_pick.hdf5 --dart-noise 0.2
"""
import argparse
import os

import numpy as np

from il_arm.dataset import DemoWriter
from il_arm.sim import ENV_ID, ScriptedExpert, flatten_obs, make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--out', default='data/fetch_pick.hdf5')
    ap.add_argument('--dart-noise', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-failures', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    env = make_env()
    expert = ScriptedExpert()
    rng = np.random.default_rng(args.seed)
    writer = DemoWriter(args.out, ENV_ID, mode='w')
    n_ok = 0
    for ep in range(args.episodes):
        o, _ = env.reset(seed=args.seed + ep)
        expert.reset()
        obs, acts = [], []
        done = False
        info = {}
        while not done:
            a = expert.act(o)
            obs.append(flatten_obs(o))
            acts.append(a)
            executed = a.copy()
            if args.dart_noise > 0:
                executed[:3] = np.clip(executed[:3] + rng.normal(0, args.dart_noise, 3), -1, 1)
            o, _, term, trunc, info = env.step(executed)
            done = term or trunc
        success = bool(info.get('is_success', False))
        n_ok += success
        if success or args.keep_failures:
            writer.add({'state': np.stack(obs)}, np.stack(acts), success, seed=args.seed + ep)
        if (ep + 1) % 50 == 0:
            print(f'{ep + 1}/{args.episodes} episodes, expert success {n_ok / (ep + 1):.2f}, stored {writer.n}')
    writer.close()
    print(f'wrote {writer.n} demos to {args.out}')


if __name__ == '__main__':
    main()
