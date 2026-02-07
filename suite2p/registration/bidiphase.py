"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from numpy import fft


def compute(frames: np.ndarray) -> int:
    """
    Compute the bidirectional phase offset between odd and even scan lines.

    Estimates the pixel offset between alternating lines that can occur in
    bidirectional line scanning, using phase correlation along the x-axis.

    Parameters
    ----------
    frames : np.ndarray
        Random subsample of frames of shape (n_frames, Ly, Lx).

    Returns
    -------
    bidiphase : int
        Bidirectional phase offset in pixels.
    """

    _, Ly, Lx = frames.shape

    # compute phase-correlation between lines in x-direction
    d1 = fft.fft(frames[:, 1::2, :], axis=2)
    d1 /= np.abs(d1) + 1e-5

    d2 = np.conj(fft.fft(frames[:, ::2, :], axis=2))
    d2 /= np.abs(d2) + 1e-5
    d2 = d2[:, :d1.shape[1], :]

    cc = np.real(fft.ifft(d1 * d2, axis=2))
    cc = cc.mean(axis=1).mean(axis=0)
    cc = fft.fftshift(cc)

    bidiphase = -(np.argmax(cc[-10 + Lx // 2:11 + Lx // 2]) - 10)
    return bidiphase


def shift(frames: np.ndarray, bidiphase: int) -> None:
    """
    Shift odd scan lines by the bidirectional phase offset.

    Corrects bidirectional scanning artifacts by shifting every other row
    (odd lines) along the x-axis by the given pixel offset.

    Parameters
    ----------
    frames : np.ndarray
        Frames of shape (n_frames, Ly, Lx). Modified in-place.
    bidiphase : int
        Bidirectional phase offset in pixels.

    Returns
    -------
    frames : np.ndarray
        The input frames with odd lines shifted.
    """
    if bidiphase > 0:
        frames[:, 1::2, bidiphase:] = frames[:, 1::2, :-bidiphase]
    else:
        frames[:, 1::2, :bidiphase] = frames[:, 1::2, -bidiphase:]
    return frames
