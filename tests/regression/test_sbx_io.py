import os
import numpy as np
import pytest
import scipy.io as sio

sbxreader = pytest.importorskip("sbxreader")

import suite2p
from suite2p import default_db, default_settings
from suite2p.io import init_dbs, sbx_to_binary
from suite2p.io.sbx import sbx_get_shape

def _write_synthetic_sbx(
    tmp_path,
    stem="test_rec",
    nframes=25,
    nplanes=1,
    nchannels=1,
    Ly=32,
    Lx=48,
    scanmode=1,
    dead_cols=0,
    gt_signal=None,
):
    """Create a synthetic Scanbox .sbx + .mat file pair with known pixel values."""
    sbx_path = tmp_path / f"{stem}.sbx"
    mat_path = tmp_path / f"{stem}.mat"

    if gt_signal is None:
        rng = np.random.default_rng(42)
        gt_signal = rng.integers(
            100, 30000, size=(nframes, nplanes, nchannels, Ly, Lx), dtype=np.uint16
        )

    channels_code = 1 if nchannels == 2 else 2  # 1 = both PMTs, 2 = PMT0 only
    info = {
        "sz": np.array([Ly, Lx], dtype=np.int32),
        "channels": channels_code,
        "scanmode": scanmode,
        "volscan": 1 if nplanes > 1 else 0,
        "recordsPerBuffer": Ly,
        "resfreq": 7920,
        "scanbox_version": 2.0,
        "config": {
            "frames": nframes * nplanes,
            "magnification": 1,
            "magnification_list": np.array(["1.0"], dtype=object),
            "lines": Ly,
            "coord_rel": np.zeros(3, dtype=np.float64),
        },
    }
    if nplanes > 1:
        info["otwave"] = np.arange(nplanes, dtype=np.int32)

    sio.savemat(mat_path, {"info": info})

    # Invert (x << 1) so `sbx_to_binary` (`(~raw) >> 1`) recovers `gt_signal`
    raw_vals = np.invert(gt_signal.astype(np.uint16) << 1)
    if dead_cols > 0:
        raw_vals[..., :dead_cols] = 0xFFFF

    # Scanbox disk layout: (nframes, nplanes, Ly, Lx, nchannels) in C-order
    disk_order = np.transpose(raw_vals, (0, 1, 3, 4, 2))
    disk_order.ravel(order="C").tofile(sbx_path)

    return str(sbx_path), gt_signal

def _run_conversion(
    sbx_path,
    tmp_path,
    nplanes_init=1,
    nchannels_init=1,
    functional_chan=1,
    ndeadcols=-1,
    ndeadrows=0,
):
    db = default_db()
    db["data_path"] = [os.path.dirname(sbx_path)]
    db["save_path0"] = str(tmp_path / "out")
    db["input_format"] = "sbx"
    db["file_list"] = [sbx_path]
    db["nplanes"] = nplanes_init
    db["nchannels"] = nchannels_init

    settings = default_settings()
    settings["io"]["batch_size"] = 8
    settings["io"]["functional_chan"] = functional_chan

    dbs = init_dbs(db)
    reg_file = [open(d["reg_file"], "wb") for d in dbs]
    reg_file_chan2 = (
        [open(d["reg_file_chan2"], "wb") for d in dbs]
        if dbs[0]["nchannels"] > 1
        else None
    )
    try:
        dbs = sbx_to_binary(
            dbs,
            settings,
            reg_file,
            reg_file_chan2,
            ndeadcols=ndeadcols,
            ndeadrows=ndeadrows,
        )
    finally:
        for f in reg_file:
            f.close()
        if reg_file_chan2:
            for f in reg_file_chan2:
                f.close()

    return dbs


@pytest.mark.parametrize(
    "nplanes,nchannels,functional_chan",
    [
        (1, 1, 1),  # Single-plane, 1-channel
        (1, 2, 1),  # Single-plane, 2-channel (functional = PMT0)
        (1, 2, 2),  # Single-plane, 2-channel (functional = PMT1)
        (3, 1, 1),  # Multi-plane ETL, 1-channel
        (3, 2, 1),  # Multi-plane ETL, 2-channel (Issue #1109 regression)
    ],
)
def test_sbx_universal_shapes_and_auto_detection(tmp_path, nplanes, nchannels, functional_chan):
    sbx_path, gt_signal = _write_synthetic_sbx(
        tmp_path,
        nframes=23,
        nplanes=nplanes,
        nchannels=nchannels,
        Ly=24,
        Lx=32,
        scanmode=1,
    )

    assert sbx_get_shape(sbx_path) == (23, nplanes, nchannels, 24, 32)

    dbs = _run_conversion(
        sbx_path,
        tmp_path,
        nplanes_init=nplanes,
        nchannels_init=nchannels,
        functional_chan=functional_chan,
    )

    assert len(dbs) == nplanes
    nfunc = (functional_chan - 1) if nchannels > 1 else 0
    nchan2 = 1 - nfunc

    for p in range(nplanes):
        assert dbs[p]["nframes"] == 23
        assert dbs[p]["Ly"] == 24
        assert dbs[p]["Lx"] == 32

        bin1 = np.memmap(dbs[p]["reg_file"], dtype=np.int16, mode="r", shape=(23, 24, 32))
        expected1 = gt_signal[:, p, nfunc].astype(np.int16)
        np.testing.assert_array_equal(bin1, expected1)
        np.testing.assert_allclose(dbs[p]["meanImg"], expected1.mean(axis=0), rtol=1e-5)

        if nchannels > 1:
            bin2 = np.memmap(dbs[p]["reg_file_chan2"], dtype=np.int16, mode="r", shape=(23, 24, 32))
            expected2 = gt_signal[:, p, nchan2].astype(np.int16)
            np.testing.assert_array_equal(bin2, expected2)
            np.testing.assert_allclose(dbs[p]["meanImg_chan2"], expected2.mean(axis=0), rtol=1e-5)


def test_sbx_bidirectional_dead_cols_and_flyback_crop(tmp_path):
    sbx_path, gt_signal = _write_synthetic_sbx(
        tmp_path,
        nframes=12,
        nplanes=2,
        nchannels=1,
        Ly=20,
        Lx=30,
        scanmode=0,
        dead_cols=4,
    )

    dbs = _run_conversion(
        sbx_path,
        tmp_path,
        nplanes_init=2,
        nchannels_init=1,
        ndeadcols=-1,
        ndeadrows=2,
    )

    for p in range(2):
        assert dbs[p]["Ly"] == 18
        assert dbs[p]["Lx"] == 26
        bin1 = np.memmap(dbs[p]["reg_file"], dtype=np.int16, mode="r", shape=(12, 18, 26))
        np.testing.assert_array_equal(bin1, gt_signal[:, p, 0, 2:, 4:].astype(np.int16))


def test_sbx_end_to_end_run_s2p(tmp_path):
    """Run full Suite2p pipeline (I/O -> registration -> detection -> extraction) offline."""
    nframes, Ly, Lx = 210, 128, 128
    rng = np.random.default_rng(123)
    movie = rng.integers(50, 150, size=(nframes, 1, 1, Ly, Lx), dtype=np.uint16)

    # Add two synthetic active neurons with calcium transients so detection finds ROIs
    yy, xx = np.ogrid[:Ly, :Lx]
    cell1 = (yy - 40) ** 2 + (xx - 40) ** 2 <= 5**2
    cell2 = (yy - 88) ** 2 + (xx - 88) ** 2 <= 5**2
    for t in range(15, nframes, 30):
        decay = (2000 * np.exp(-np.arange(min(15, nframes - t)) / 4.0)).astype(np.uint16)
        for dt, val in enumerate(decay):
            movie[t + dt, 0, 0][cell1] += val
            movie[t + dt, 0, 0][cell2] += val

    sbx_dir = tmp_path / "sbx_in"
    sbx_dir.mkdir()
    _write_synthetic_sbx(
        sbx_dir,
        stem="synth_000_000",
        nframes=nframes,
        nplanes=1,
        nchannels=1,
        Ly=Ly,
        Lx=Lx,
        gt_signal=movie,
    )

    db = default_db()
    db["data_path"] = [str(sbx_dir)]
    db["save_path0"] = str(tmp_path / "sbx_out")
    db["input_format"] = "sbx"

    settings = default_settings()
    settings["fs"] = 30.0
    settings["tau"] = 1.0
    settings["diameter"] = 10
    settings["registration"]["nonrigid"] = False

    suite2p.run_s2p(db=db, settings=settings)

    plane0 = tmp_path / "sbx_out" / "suite2p" / "plane0"
    for fname in ["data.bin", "F.npy", "Fneu.npy", "spks.npy", "stat.npy", "iscell.npy"]:
        assert (plane0 / fname).exists(), f"Missing pipeline output: {fname}"