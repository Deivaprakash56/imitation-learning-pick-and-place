"""Roll out a BC policy (or the expert) in simulation and report success rate.

  python -m il_arm.eval_policy runs/bc_v1/best.pt --episodes 100
  python -m il_arm.eval_policy expert --episodes 100
  python -m il_arm.eval_policy runs/bc_v1/best.pt --episodes 3 --render    # MuJoCo viewer
"""
import argparse

import numpy as np

from il_arm.sim import ScriptedExpert, flatten_obs, make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('policy', help='path to .pt or "expert"')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=100_000, help='disjoint from training seeds')
    ap.add_argument('--render', action='store_true')
    args = ap.parse_args()

    env = make_env(render_mode='human' if args.render else None)
    if args.policy == 'expert':
        expert = ScriptedExpert()
        reset, act = expert.reset, expert.act
    else:
        from il_arm.bc_policy import BCPolicy, ChunkExecutor
        ex = ChunkExecutor(BCPolicy.load(args.policy))
        reset, act = ex.reset, (lambda o: ex.act(flatten_obs(o)))

    succ = []
    for ep in range(args.episodes):
        o, _ = env.reset(seed=args.seed + ep)
        reset()
        done = False
        while not done:
            o, _, term, trunc, info = env.step(np.clip(act(o), -1, 1))
            done = term or trunc
        succ.append(bool(info.get('is_success', False)))
    print(f'{args.policy}: success {np.mean(succ):.3f} over {args.episodes} episodes')


if __name__ == '__main__':
    main()
