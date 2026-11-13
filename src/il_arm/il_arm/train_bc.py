"""Train a behaviour-cloning policy from HDF5 demos.

  python -m il_arm.train_bc data/fetch_pick.hdf5 --out runs/bc_v1 --chunk 5 --epochs 300
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from il_arm.bc_policy import BCPolicy
from il_arm.dataset import load_demos


def build_chunks(demos, chunk):
    """For each timestep, target = next `chunk` actions (padded with the last action)."""
    X, Y = [], []
    for obs, act in demos:
        T = len(act)
        pad = np.concatenate([act, np.repeat(act[-1:], chunk - 1, axis=0)])
        for t in range(T):
            X.append(obs[t])
            Y.append(pad[t:t + chunk])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--out', default='runs/bc')
    ap.add_argument('--obs-key', default='state')
    ap.add_argument('--chunk', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--gripper-weight', type=float, default=2.0, help='extra loss weight on gripper dim')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-demos', type=int, default=0, help='use only the first N demos (0 = all)')
    ap.add_argument('--threads', type=int, default=0, help='torch CPU threads (0 = default)')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    demos = load_demos(args.data, args.obs_key)
    if args.max_demos:
        demos = demos[:args.max_demos]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(demos))
    n_val = max(1, int(len(demos) * args.val_frac))
    val_d = [demos[i] for i in idx[:n_val]]
    tr_d = [demos[i] for i in idx[n_val:]]
    Xtr, Ytr = build_chunks(tr_d, args.chunk)
    Xva, Yva = build_chunks(val_d, args.chunk)
    print(f'{len(tr_d)} train demos ({len(Xtr)} samples), {len(val_d)} val demos ({len(Xva)} samples)')

    hidden = (512, 512, 512)
    model = BCPolicy(Xtr.shape[1], Ytr.shape[2], args.chunk, hidden=hidden).to(args.device)
    model.set_norm(Xtr.mean(0), Xtr.std(0))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xtr_t, Ytr_t = torch.as_tensor(Xtr, device=args.device), torch.as_tensor(Ytr, device=args.device)
    Xva_t, Yva_t = torch.as_tensor(Xva, device=args.device), torch.as_tensor(Yva, device=args.device)
    w = torch.ones(Ytr.shape[2], device=args.device)
    w[-1] = args.gripper_weight

    best = float('inf')
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=args.device)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            loss = (F.smooth_l1_loss(model(Xtr_t[b]), Ytr_t[b], reduction='none', beta=0.1) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = (F.smooth_l1_loss(model(Xva_t), Yva_t, reduction='none', beta=0.1) * w).mean().item()
        if vl < best:
            best = vl
            model.save(os.path.join(args.out, 'best.pt'), hidden=hidden, val_loss=vl, epoch=ep)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f'epoch {ep:4d}  train {tot / len(Xtr_t):.5f}  val {vl:.5f}  best {best:.5f}', flush=True)
        with open(os.path.join(args.out, 'loss.csv'), 'a' if ep else 'w') as lf:
            lf.write(f'{ep},{tot / len(Xtr_t):.6f},{vl:.6f}\n')
    model.save(os.path.join(args.out, 'last.pt'), hidden=hidden, val_loss=vl, epoch=args.epochs - 1)
    print(f'saved {args.out}/best.pt')


if __name__ == '__main__':
    main()
