import numpy as np
import pytest
import torch
from suite2p.registration import bidiphase
from suite2p.registration.nonrigid import transform_data


def test_positive_bidiphase_shift_shifts_every_other_line():
    orig = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])
    expected = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5, 6, 7]]
    ])

    shifted = orig.copy()
    bidiphase.shift(shifted, 2)
    assert np.allclose(shifted, expected)


def test_negative_bidiphase_shift_shifts_every_other_line():
    orig = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])
    expected = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [3, 4, 5, 6, 7, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [3, 4, 5, 6, 7, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])

    shifted = orig.copy()
    bidiphase.shift(shifted, -2)
    assert np.allclose(shifted, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_transform_data_mps_cpu_consistency():
    """Test that MPS and CPU code paths in transform_data produce similar results."""
    from suite2p.registration.nonrigid import make_blocks

    np.random.seed(42)
    torch.manual_seed(42)
    Ly, Lx, n_frames = 128, 128, 2
    yblock, xblock, nblocks, *_ = make_blocks(Ly, Lx, (32, 32))
    data_np = np.random.rand(n_frames, Ly, Lx).astype(np.float32) * 100
    ymax1 = torch.randn(nblocks[0] * nblocks[1], n_frames) * 2
    xmax1 = torch.randn(nblocks[0] * nblocks[1], n_frames) * 2

    result_cpu = transform_data(
        torch.from_numpy(data_np), nblocks, xblock, yblock, ymax1.clone(), xmax1.clone()
    )
    result_mps = transform_data(
        torch.from_numpy(data_np).to("mps"), nblocks, xblock, yblock,
        ymax1.clone().to("mps"), xmax1.clone().to("mps")
    )

    cpu_np = result_cpu.numpy().astype(np.float32)
    mps_np = result_mps.cpu().numpy().astype(np.float32)
    correlation = np.corrcoef(cpu_np.flatten(), mps_np.flatten())[0, 1]
    max_diff = np.abs(cpu_np - mps_np).max()

    assert correlation > 0.99, f"Correlation: {correlation}"
    assert max_diff < 2, f"Max diff: {max_diff}"