"""README figures for il_arm.

  MUJOCO_GL=egl python -m il_arm.plot_results filmstrip runs/bc_main/best.pt --out src/il_arm/docs
  python -m il_arm.plot_results demos data/fetch_pick_300.hdf5 --out src/il_arm/docs
  python -m il_arm.plot_results ablations runs/il_ablations.csv --out src/il_arm/docs
  python -m il_arm.plot_results loss runs/bc_main/loss.csv --out src/il_arm/docs
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
CAMERA = {'distance': 1.1, 'azimuth': 140.0, 'elevation': -22.0, 'lookat': np.array([1.3, 0.75, 0.55])}


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'axes.spines.top': False,
        'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'lines.linewidth': 2, 'legend.frameon': False})


def filmstrip(policy_path, out, seed=100_003, n_frames=6):
    import gymnasium as gym
    import gymnasium_robotics
    from il_arm.bc_policy import BCPolicy, ChunkExecutor
    from il_arm.sim import ENV_ID, flatten_obs
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode='rgb_array', max_episode_steps=50, width=360, height=300,
                   default_camera_config=CAMERA)
    ex = ChunkExecutor(BCPolicy.load(policy_path))
    # choose an episode whose goal is in the air (> 15 cm above the table) so every phase is visible
    for seed in range(seed, seed + 200):
        o, _ = env.reset(seed=seed)
        if o['desired_goal'][2] - o['observation'][5] > 0.15:
            break
    ex.reset()
    frames, grip, info, t = [env.render()], [], {}, 0
    done = False
    while not done:
        a = np.clip(ex.act(flatten_obs(o)), -1, 1)
        grip.append(a[3])
        o, _, term, trunc, info = env.step(a)
        frames.append(env.render())
        done = term or trunc
        t += 1
    # pick frames at phase changes: start, above object, grasp, lifted, mid-carry, end
    g = np.array(grip)
    close_t = int(np.argmax(g < 0)) if (g < 0).any() else len(g) // 3
    idx = sorted(set([0, max(close_t - 4, 1), close_t + 1, min(close_t + 6, t), min(close_t + 14, t), t]))
    titles = {0: 'start', idx[1]: 'above object', idx[2]: 'gripper closes', idx[3]: 'lift',
              idx[4]: 'carry to goal', t: 'goal reached' if info.get('is_success') else 'end (failed)'}
    fig, axes = plt.subplots(1, len(idx), figsize=(2.6 * len(idx), 2.6))
    for ax, i in zip(axes, idx):
        ax.imshow(frames[i])
        ax.set_title(f't={i}: {titles.get(i, "")}', fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    p = os.path.join(out, 'filmstrip.png')
    fig.savefig(p, dpi=120)
    import imageio
    gif = os.path.join(out, 'rollout.gif')
    imageio.mimsave(gif, [f[::2, ::2] for f in frames] + [frames[-1][::2, ::2]] * 10, duration=0.08, loop=0)
    print('wrote', p, gif, '| seed', seed, '| success:', info.get('is_success'))


def demos(path, out, n=12):
    from il_arm.dataset import load_demos
    ds = load_demos(path)
    fig = plt.figure(figsize=(13, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for k, (obs, act) in enumerate(ds[:n]):
        grip, obj, goal = obs[:, 0:3], obs[:, 3:6], obs[:, 25:28]
        ax.plot(*grip.T, color=BLUE, lw=1.2, alpha=0.8, label='gripper' if k == 0 else None)
        ax.plot(*obj.T, color=ORANGE, lw=1.2, alpha=0.9, label='object' if k == 0 else None)
        ax.scatter(*obj[0], color=ORANGE, s=14)
        ax.scatter(*goal[0], color=AQUA, marker='*', s=60, label='goal' if k == 0 else None)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(f'{n} of {len(ds)} demonstrations (DART-perturbed execution)')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=-60)

    obs, act = ds[0]
    ax = fig.add_subplot(1, 2, 2)
    names = ['dx', 'dy', 'dz', 'gripper (+open / −close)']
    for j, (nm, c) in enumerate(zip(names, [BLUE, ORANGE, AQUA, YELLOW])):
        ax.plot(act[:, j], color=c, lw=1.8 if j == 3 else 1.4, label=nm, drawstyle='steps-post' if j == 3 else 'default')
    close_t = int(np.argmax(act[:, 3] < 0))
    ax.axvline(close_t, color=INK2, lw=0.8, ls='--')
    ax.text(close_t + 0.5, 1.05, 'grasp', fontsize=8.5, color=INK2)
    ax.set_ylim(-1.15, 1.2)
    ax.set_xlabel('time step (0.04 s)')
    ax.set_title('Expert action labels for demo 0')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(out, 'demos.png')
    fig.savefig(p, dpi=125)
    print('wrote', p)


def ablations(csv_path, out):
    with open(csv_path) as f:
        rows = {r['name']: r for r in csv.DictReader(f)}
    s = lambda n: float(rows[n]['success']) if n in rows else np.nan  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1.25, 1]})

    ax = axes[0]
    ns = [25, 50, 100, 300]
    ys = [s('demos_25'), s('demos_50'), s('demos_100'), s('main_300')]
    ax.plot(ns, ys, color=BLUE, marker='o', ms=7, label='DART noise 0.2')
    for x, y in zip(ns, ys):
        if not np.isnan(y):
            ax.annotate(f'{y:.0%}', (x, y), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=9,
                        color=BLUE)
    nd = [(25, s('nodart_25')), (50, s('nodart_50')), (100, s('nodart_100')), (300, s('nodart_300'))]
    nd = [(x, y) for x, y in nd if not np.isnan(y)]
    if nd:
        ax.plot(*zip(*nd), color=ORANGE, marker='s', ms=7, ls='--', label='no noise (plain BC)')
        for x, y in nd:
            ax.annotate(f'{y:.0%}', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9,
                        color=ORANGE)
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel('number of demonstrations (log scale)')
    ax.set_title('Success rate vs. demonstrations')
    ax.legend(loc='lower right', fontsize=8.5)

    ax = axes[1]
    labels = ['full recipe\n(300, DART, chunk 5)', 'no DART\nnoise', 'no action\nchunking']
    vals = [s('main_300'), s('nodart_300'), s('chunk1_300')]
    bars = ax.bar(range(3), vals, color=[BLUE, ORANGE, ORANGE], width=0.6)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.0%}', ha='center', fontsize=9.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis='x', visible=False)
    ax.set_title('Remove one ingredient at a time (300 demos)')
    fig.tight_layout()
    p = os.path.join(out, 'ablations.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)


def loss(csv_path, out):
    d = np.loadtxt(csv_path, delimiter=',')
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(d[:, 0], d[:, 1], color=BLUE, label='train')
    ax.plot(d[:, 0], d[:, 2], color=ORANGE, label='validation (held-out demos)')
    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('weighted smooth-L1')
    ax.set_title('Behaviour-cloning loss (300 demos, chunk 5)')
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(out, 'bc_loss.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)


def teleop(out, data_dir='data', runs_dir='runs'):
    """Clock-driven vs state-driven teleop: one recorded demo each + policy success."""
    from il_arm.dataset import load_demos
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={'width_ratios': [1.2, 1.2, 0.7]})
    for ax, mode, title in [(axes[0], 'timer', 'Clock-driven operator'), (axes[1], 'state', 'State-driven operator')]:
        obs, act = load_demos(os.path.join(data_dir, f'fake_arm_demos_{mode}.hdf5'))[0]
        t = np.arange(len(act)) / 15.0
        ax.plot(t, obs[:, 2], color=BLUE, label='joint 3 measured (in obs)')
        ax.plot(t, act[:, 2], color=ORANGE, drawstyle='steps-post', label='joint 3 command (label)')
        ax.plot(t, obs[:, 3] * 10 - 1.2, color=AQUA, lw=1.4, label='gripper (scaled)')
        if mode == 'timer':
            # find the longest stretch where the arm is still but the label changes
            still = np.abs(np.diff(obs[:, 2])) < 1e-3
            idx = np.where(still)[0]
            if len(idx):
                start = idx[0]
                flips = np.where(np.abs(np.diff(act[start:, 2])) > 1e-3)[0]
                end = start + (flips[0] + 1 if len(flips) else 12)
                ax.axvspan(t[start], t[min(end, len(t) - 1)], color=ORANGE, alpha=0.12, lw=0)
                ax.text(t[start] + 0.05, 0.1, 'same observation, label "hold" ...\nthen "descend": BC averages them',
                        fontsize=8, color=INK2)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('rad')
        ax.set_title(title)
        ax.set_ylim(-1.3, 0.35)
        ax.legend(fontsize=7.5, loc='lower left')
    ax = axes[2]
    vals = []
    for mode in ('timer', 'state'):
        pth = os.path.join(runs_dir, f'bc_fake_arm_{mode}', 'ros_e2e_result.csv')
        ok, n = (int(x) for x in open(pth).read().strip().split(','))
        vals.append(ok / n)
    bars = ax.bar([0, 1], vals, color=[ORANGE, BLUE], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.0%}', ha='center', fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['clock-driven\ndemos', 'state-driven\ndemos'], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis='x', visible=False)
    ax.set_title('policy_runner success\n(20 new objects)')
    fig.tight_layout()
    p = os.path.join(out, 'teleop_consistency.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['filmstrip', 'demos', 'ablations', 'loss', 'teleop'])
    ap.add_argument('input', nargs='?', default='')
    ap.add_argument('--out', default='docs')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()
    if args.what == 'teleop':
        teleop(args.out)
        return
    {'filmstrip': filmstrip, 'demos': demos, 'ablations': ablations, 'loss': loss}[args.what](args.input, args.out)


if __name__ == '__main__':
    main()
