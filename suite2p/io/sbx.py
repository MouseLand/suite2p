"""
Copyright © 2026 Purdue University, Authored by Jacob Gilpin.
"""
import logging
import numpy as np

try:
    from sbxreader import sbx_memmap
    HAS_SBX = True
except ImportError:
    sbx_memmap = None
    HAS_SBX = False

logger = logging.getLogger(__name__)


def _require_sbxreader():
    """Raise an informative ImportError if sbxreader is not installed."""
    if not HAS_SBX:
        raise ImportError(
            "sbxreader is required to read Scanbox (.sbx) files. "
            "Install it with: pip install 'suite2p[io]' (or pip install sbxreader)"
        )


def _close_memmap(mm):
    """Safely release underlying mmap handles (prevents file-lock errors on Windows)."""
    if mm is None:
        return
    base = getattr(mm, "base", None)
    if hasattr(mm, "_mmap") and mm._mmap is not None:
        mm._mmap.close()
    elif hasattr(base, "_mmap") and base._mmap is not None:
        base._mmap.close()


def sbx_get_shape(sbxfile):
    """Return (nframes, nplanes, nchannels, Ly, Lx) for a Scanbox .sbx file.

    Parameters
    ----------
    sbxfile : str or Path
        Path to the .sbx file.

    Returns
    -------
    shape : tuple of int
        (nframes, nplanes, nchannels, Ly, Lx)
    """
    _require_sbxreader()
    mm = sbx_memmap(sbxfile)
    try:
        return tuple(int(dim) for dim in mm.shape)
    finally:
        _close_memmap(mm)


def sbx_get_info(sbxfile):
    """Return the parsed Scanbox metadata object via sbxreader.

    Parameters
    ----------
    sbxfile : str or Path
        Path to the .sbx file.

    Returns
    -------
    info : object
        Metadata object parsed from the companion .mat file.
    """
    _require_sbxreader()
    mm = sbx_memmap(sbxfile)
    try:
        return mm.metadata
    finally:
        _close_memmap(mm)


def _estimate_dead_cols(mm, scanmode=0, n_sample_frames=20):
    """Estimate leading dead columns for bidirectional Scanbox acquisitions.

    Averages across up to `n_sample_frames` frames rather than frame 0 alone
    to avoid resonant-scanner startup artifacts.
    """
    if scanmode != 0 or mm.shape[0] == 0:
        return 0
    n_avail = min(int(mm.shape[0]), n_sample_frames)
    sample = np.asarray(mm[:n_avail, 0, 0])
    # Sum along frames and Y in float64 to avoid uint16 overflow
    col_profile = np.sum(sample, axis=(0, 1), dtype=np.float64)
    zero_cols = np.flatnonzero(col_profile == 0)
    if zero_cols.size > 0:
        return int(zero_cols[-1]) + 1
    return 0


def sbx_to_binary(dbs, settings, reg_file, reg_file_chan2, ndeadcols=-1, ndeadrows=0):
    """Convert Scanbox (.sbx) files into Suite2p binary files.

    Parameters
    ----------
    dbs : list of dict
        Per-plane database dictionaries initialized by `suite2p.io.init_dbs`.
    settings : dict
        Suite2p pipeline settings dictionary.
    reg_file : list of file-like
        Pre-opened writable binary file handles for the functional channel (one per plane).
    reg_file_chan2 : list of file-like or None
        Pre-opened writable binary file handles for the second channel (one per plane),
        or None/empty if `nchannels == 1`.
    ndeadcols : int, optional (default -1)
        Number of leading columns to crop. -1 auto-detects for bidirectional scans.
    ndeadrows : int, optional (default 0)
        Number of top rows to crop (e.g., ETL flyback lines in multi-plane scans).

    Returns
    -------
    dbs : list of dict
        Updated per-plane database dictionaries with cropped Ly, Lx, nframes, and mean images.
    """
    _require_sbxreader()

    if not dbs:
        raise ValueError("`dbs` must contain at least one plane dictionary.")

    db0 = dbs[0]
    nplanes = int(db0["nplanes"])
    nchannels = int(db0["nchannels"])
    io_settings = settings.get("io", {})
    batch_size = int(io_settings.get("batch_size", 500))
    functional_chan = int(io_settings.get("functional_chan", 1))
    nfunc = (functional_chan - 1) if nchannels > 1 else 0

    if nfunc < 0 or nfunc >= nchannels:
        raise ValueError(
            f"Invalid functional_chan={functional_chan} for nchannels={nchannels}."
        )

    file_list = db0.get("file_list", [])
    if len(file_list) == 0:
        raise FileNotFoundError(f"No .sbx files found in {db0.get('data_path')}")

    # Inspect first file to validate geometry and resolve dead-pixel cropping
    first_mm = sbx_memmap(file_list[0])
    try:
        _, file_planes, file_chans, raw_Ly, raw_Lx = (int(d) for d in first_mm.shape)
        if file_planes != nplanes or len(dbs) != file_planes:
            raise ValueError(
                f"Plane count mismatch: .sbx file has {file_planes} planes, "
                f"but db/dbs initialized with {nplanes} ({len(dbs)} dicts)."
            )
        if file_chans != nchannels:
            raise ValueError(
                f"Channel count mismatch: .sbx file has {file_chans} channels, "
                f"but db initialized with nchannels={nchannels}."
            )
        if nchannels > 1 and (not reg_file_chan2 or len(reg_file_chan2) < nplanes):
            raise ValueError(
                f"Expected {nplanes} channel-2 binary handles in `reg_file_chan2` "
                f"for nchannels={nchannels}."
            )

        meta = getattr(first_mm, "metadata", {})
        if isinstance(meta, dict):
            scanmode = 0 if str(meta.get("scanning_mode", "")).lower().startswith("bi") else 1
        else:
            scanmode = int(getattr(meta, "scanmode", 1))
        if ndeadcols == -1:
            ndeadcols = _estimate_dead_cols(first_mm, scanmode=scanmode)
        ndeadcols = max(0, int(ndeadcols))
        ndeadrows = max(0, int(ndeadrows))
    finally:
        _close_memmap(first_mm)

    Ly = raw_Ly - ndeadrows
    Lx = raw_Lx - ndeadcols
    if Ly <= 0 or Lx <= 0:
        raise ValueError(
            f"Invalid cropped frame shape ({Ly}, {Lx}) from raw ({raw_Ly}, {raw_Lx}) "
            f"with ndeadrows={ndeadrows}, ndeadcols={ndeadcols}."
        )

    # Initialize per-plane metadata and float64 mean accumulators
    mean_acc = [np.zeros((Ly, Lx), dtype=np.float64) for _ in range(nplanes)]
    mean_acc_chan2 = (
        [np.zeros((Ly, Lx), dtype=np.float64) for _ in range(nplanes)]
        if nchannels > 1
        else None
    )

    for j in range(nplanes):
        dbs[j]["Ly"] = Ly
        dbs[j]["Lx"] = Lx
        dbs[j]["nframes"] = 0
        dbs[j]["nframes_per_folder"] = np.zeros(len(file_list), dtype=np.int32)

        if hasattr(reg_file[j], "Ly"):
            reg_file[j].Ly = Ly
            reg_file[j].Lx = Lx
        if nchannels > 1 and hasattr(reg_file_chan2[j], "Ly"):
            reg_file_chan2[j].Ly = Ly
            reg_file_chan2[j].Lx = Lx

    # Number of volumes per chunk, scaled so total frames per chunk stays near batch_size
    frames_per_batch = max(1, batch_size // max(1, nplanes))
    # Single reusable contiguous buffer to avoid per-batch allocations
    work_buf = np.empty((frames_per_batch, Ly, Lx), dtype=np.uint16)

    for ifile, sbxfile in enumerate(file_list):
        logger.info("Converting %s to binary...", sbxfile)
        mm = sbx_memmap(sbxfile)
        try:
            nframes_file, f_planes, f_chans, f_Ly, f_Lx = (int(d) for d in mm.shape)
            if (f_planes, f_chans, f_Ly, f_Lx) != (nplanes, nchannels, raw_Ly, raw_Lx):
                raise ValueError(
                    f"Inconsistent shape in {sbxfile}: expected "
                    f"({nplanes}, {nchannels}, {raw_Ly}, {raw_Lx}), "
                    f"got ({f_planes}, {f_chans}, {f_Ly}, {f_Lx})."
                )
            if nframes_file == 0:
                logger.warning("Skipping empty .sbx file (0 frames): %s", sbxfile)
                continue

            for i in range(0, nframes_file, frames_per_batch):
                iend = min(i + frames_per_batch, nframes_file)
                n_chunk = iend - i

                # Single sequential disk read for the entire batch
                raw_batch = mm[i:iend, :, :, ndeadrows:, ndeadcols:]

                for j in range(nplanes):
                    for ichan in range(nchannels):
                        # Scanbox stores raw PMT values inverted in uint16 (65535 = dark).
                        # Invert and right-shift in-place into contiguous scratch buffer,
                        # then zero-copy view as int16 (MSB is guaranteed 0 after >> 1).
                        work = work_buf[:n_chunk]
                        np.invert(raw_batch[:, j, ichan], out=work)
                        np.right_shift(work, 1, out=work)
                        im2write = work.view(np.int16)

                        if ichan == nfunc:
                            reg_file[j].write(memoryview(im2write))
                            mean_acc[j] += np.sum(im2write, axis=0, dtype=np.float64)
                        else:
                            reg_file_chan2[j].write(memoryview(im2write))
                            mean_acc_chan2[j] += np.sum(im2write, axis=0, dtype=np.float64)

                    # Increment frame counters once per plane per batch (outside channel loop)
                    dbs[j]["nframes"] += n_chunk
                    dbs[j]["nframes_per_folder"][ifile] += n_chunk
        finally:
            _close_memmap(mm)

    # Finalize mean images and persist per-plane metadata
    for j in range(nplanes):
        n_tot = dbs[j]["nframes"]
        if n_tot > 0:
            dbs[j]["meanImg"] = (mean_acc[j] / n_tot).astype(np.float32)
            if nchannels > 1:
                dbs[j]["meanImg_chan2"] = (mean_acc_chan2[j] / n_tot).astype(np.float32)
        else:
            dbs[j]["meanImg"] = np.zeros((Ly, Lx), dtype=np.float32)
            if nchannels > 1:
                dbs[j]["meanImg_chan2"] = np.zeros((Ly, Lx), dtype=np.float32)

        if "settings_path" in dbs[j]:
            np.save(dbs[j]["settings_path"], settings)
        if "db_path" in dbs[j]:
            np.save(dbs[j]["db_path"], dbs[j])
        if "ops_path" in dbs[j]:
            np.save(dbs[j]["ops_path"], dbs[j])

    return dbs