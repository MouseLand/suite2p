import numpy as np
from numpy import fft


def compute(frames: np.ndarray) -> int:
    """
    Returns the bidirectional phase offset, the offset between lines that sometimes occurs in line scanning.

    Parameters
    ----------
    frames : frames x Ly x Lx
        random subsample of frames in binary (frames x Ly x Lx)

    Returns
    -------
    bidiphase : int
        bidirectional phase offset in pixels

    """

    _, Ly, Lx = frames.shape

    # compute phase-correlation between lines in x-direction
    d1 = fft.fft(frames[:, 1::2, :], axis=2)
    d1 /= np.abs(d1) + 1e-5

    d2 = np.conj(fft.fft(frames[:, ::2, :], axis=2))
    d2 /= np.abs(d2) + 1e-5
    d2 = d2[:,:d1.shape[1],:]

    cc = np.real(fft.ifft(d1 * d2, axis=2))
    cc = cc.mean(axis=1).mean(axis=0)
    cc = fft.fftshift(cc)

    bidiphase = -(np.argmax(cc[-10 + Lx // 2 : 11 + Lx // 2]) - 10)
    return bidiphase


def shift(frames: np.ndarray, bidiphase: int) -> None:
    """
    Shift last axis of 'frames' by bidirectional phase offset in-place, bidiphase.

    Parameters
    ----------
    frames : frames x Ly x Lx
    bidiphase : int
        bidirectional phase offset in pixels
    """
    if bidiphase > 0:
        frames[:, 1::2, bidiphase:] = frames[:, 1::2, :-bidiphase]
    else:
        frames[:, 1::2, :bidiphase] = frames[:, 1::2, -bidiphase:]