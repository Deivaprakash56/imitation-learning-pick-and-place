"""HDF5 demo storage shared by the sim collector and the real-robot recorder.

Layout (robomimic-style, so the data also loads in robomimic / converts to LeRobot):
  /data.attrs['env'], ['obs_keys'], ['action_dim']
  /data/demo_<i>/actions             (T, action_dim) float32
  /data/demo_<i>/obs/<key>           (T, ...) e.g. 'state' (T, D), 'image' (T, H, W, 3) uint8
  /data/demo_<i>.attrs['success'], ['num_samples']

The file is opened only while a demo is being appended, so it is always complete on disk:
training can read it while the recorder is still running, and a recorder crash loses at most
the episode in progress.
"""
import os

import h5py
import numpy as np


class DemoWriter:
    def __init__(self, path, env_name, mode='a'):
        self.path = path
        if mode == 'w' and os.path.exists(path):
            os.remove(path)
        with h5py.File(path, 'a') as f:
            grp = f.require_group('data')
            grp.attrs['env'] = env_name
            self.n = len([k for k in grp.keys() if k.startswith('demo_')])

    def add(self, obs: dict, actions: np.ndarray, success: bool, **attrs):
        with h5py.File(self.path, 'a') as f:
            grp = f['data']
            g = grp.create_group(f'demo_{self.n}')
            g.create_dataset('actions', data=np.asarray(actions, np.float32))
            og = g.create_group('obs')
            for k, v in obs.items():
                v = np.asarray(v)
                og.create_dataset(k, data=v, compression='gzip' if v.dtype == np.uint8 else None)
            g.attrs['success'] = bool(success)
            g.attrs['num_samples'] = len(actions)
            for k, v in attrs.items():
                g.attrs[k] = v
            grp.attrs['obs_keys'] = list(obs.keys())
            grp.attrs['action_dim'] = int(np.asarray(actions).shape[1])
        self.n += 1

    def close(self):
        pass   # nothing held open


def load_demos(path, obs_key='state', only_success=True):
    """Return list of (obs (T,D), actions (T,A)) tuples."""
    out = []
    with h5py.File(path, 'r') as f:
        for k in sorted(f['data'].keys(), key=lambda s: int(s.split('_')[1])):
            g = f['data'][k]
            if only_success and not g.attrs.get('success', True):
                continue
            out.append((g['obs'][obs_key][()].astype(np.float32), g['actions'][()].astype(np.float32)))
    return out
