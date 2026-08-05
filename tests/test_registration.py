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


@pytest.mark.parametrize("align_by_chan2", [False, True])
def test_reg_tif_channels_are_written_to_their_own_directories(tmp_path, align_by_chan2):
    """Each channel's registered tiffs go to its own directory (issue #1208)."""
    import tifffile
    from suite2p.parameters import default_settings
    from suite2p.registration.register import registration_wrapper

    np.random.seed(0)
    n_frames, Ly, Lx = 8, 64, 64
    # the two channels are separated by an order of magnitude so the mean of a
    # written tiff identifies which channel produced it
    f_reg = (np.random.rand(n_frames, Ly, Lx) * 20 + 100).astype("int16")
    f_reg_chan2 = (np.random.rand(n_frames, Ly, Lx) * 20 + 1000).astype("int16")

    settings = default_settings()["registration"]
    settings["reg_tif"] = True
    settings["reg_tif_chan2"] = True
    settings["batch_size"] = n_frames
    settings["nonrigid"] = False

    registration_wrapper(f_reg, f_reg_chan2=f_reg_chan2, align_by_chan2=align_by_chan2,
                         save_path=str(tmp_path), settings=settings,
                         device=torch.device("cpu"))

    chan1_tifs = sorted((tmp_path / "reg_tif").glob("*.tif"))
    chan2_tifs = sorted((tmp_path / "reg_tif_chan2").glob("*.tif"))
    assert chan1_tifs, "no tiffs written to reg_tif"
    assert chan2_tifs, "no tiffs written to reg_tif_chan2"

    assert tifffile.imread(chan1_tifs[0]).mean() < 500, "reg_tif holds channel 2 data"
    assert tifffile.imread(chan2_tifs[0]).mean() > 500, "reg_tif_chan2 holds channel 1 data"


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