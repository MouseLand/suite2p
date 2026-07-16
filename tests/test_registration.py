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

def test_convolve_matches_full_spectrum_reference():
    """convolve uses a real-input FFT; check it against a full complex-FFT reference.

    The phase-correlation spectrum of two real images is Hermitian, so the half
    spectrum carries the same information. This pins that equivalence.
    """
    from suite2p.registration.utils import convolve, ref_smooth_fft

    np.random.seed(0)
    Ly, Lx = 64, 64
    mov = torch.from_numpy(np.random.rand(4, Ly, Lx).astype(np.float32))
    ref = torch.from_numpy(np.random.rand(Ly, Lx).astype(np.float32))

    got = convolve(mov.clone(), ref_smooth_fft(ref, smooth_sigma=1.15))

    # reference: full complex spectrum, as suite2p computed it before
    cf_full = torch.conj(torch.fft.fft2(ref))
    cf_full /= (1e-5 + torch.abs(cf_full))
    from suite2p.registration.utils import gaussian_fft
    cf_full *= gaussian_fft(1.15, Ly, Lx)
    m = torch.fft.fft2(mov.clone().type(torch.complex64))
    m /= (1e-5 + torch.abs(m))
    m *= cf_full.type(torch.complex64)
    expected = torch.real(torch.fft.ifft2(m))

    assert got.shape == expected.shape
    assert got.dtype == torch.float32
    assert torch.allclose(got, expected, atol=1e-5)


def test_ref_smooth_fft_returns_half_spectrum():
    from suite2p.registration.utils import ref_smooth_fft

    ref = torch.from_numpy(np.random.rand(64, 48).astype(np.float32))
    cf = ref_smooth_fft(ref, smooth_sigma=1.15)
    assert cf.shape == (64, 48 // 2 + 1)
    assert cf.dtype == torch.complex64


def test_rigid_phasecorr_recovers_known_shifts():
    from suite2p.registration import rigid

    np.random.seed(0)
    Ly = Lx = 128
    ref = np.random.rand(Ly, Lx).astype(np.float32) * 500
    shifts = [(0, 0), (3, -2), (-5, 4)]
    frames = np.stack([np.roll(ref, s, (0, 1)) for s in shifts])

    maskMul, maskOffset, cfRefImg = rigid.compute_masks_ref_smooth_fft(
        torch.from_numpy(ref), maskSlope=3.45, smooth_sigma=1.15)
    ymax, xmax, cmax, _ = rigid.phasecorr(
        torch.from_numpy(frames), cfRefImg, maskMul, maskOffset,
        maxregshift=0.1, smooth_sigma_time=0)

    assert [(int(y), int(x)) for y, x in zip(ymax, xmax)] == shifts


def test_spatial_taper_is_float32():
    """maskMul must stay float32: a float64 mask would upcast the FFT to complex128."""
    from suite2p.registration.utils import spatial_taper

    assert spatial_taper(3.45, 64, 64).dtype == torch.float32
