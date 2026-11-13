import numpy as np
import torch

from il_arm.bc_policy import BCPolicy, ChunkExecutor
from il_arm.dataset import DemoWriter, load_demos
from il_arm.train_bc import build_chunks


def test_roundtrip(tmp_path):
    p = tmp_path / 'd.hdf5'
    w = DemoWriter(str(p), 'test', mode='w')
    w.add({'state': np.zeros((10, 3))}, np.ones((10, 2)), True)
    w.add({'state': np.zeros((7, 3))}, np.ones((7, 2)), False)
    w.close()
    assert len(load_demos(str(p))) == 1
    assert len(load_demos(str(p), only_success=False)) == 2


def test_chunks_padding():
    act = np.arange(4, dtype=np.float32)[:, None]
    X, Y = build_chunks([(np.zeros((4, 1), np.float32), act)], 3)
    assert Y.shape == (4, 3, 1)
    assert Y[3, :, 0].tolist() == [3, 3, 3] and Y[1, :, 0].tolist() == [1, 2, 3]


def test_executor_and_save(tmp_path):
    m = BCPolicy(5, 4, chunk=3, hidden=(16,)).eval()  # eval: no dropout, so outputs are comparable
    m.save(str(tmp_path / 'm.pt'), hidden=(16,))
    m2 = BCPolicy.load(str(tmp_path / 'm.pt'))
    ex = ChunkExecutor(m2)
    for _ in range(5):
        a = ex.act(np.zeros(5, np.float32))
        assert a.shape == (4,) and np.all(np.abs(a) <= 1)
    with torch.no_grad():
        assert torch.allclose(m(torch.zeros(1, 5)), m2(torch.zeros(1, 5)))
