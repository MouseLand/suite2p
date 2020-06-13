from typing import Optional, Tuple

import numpy as np


class BinaryFile:

    def __init__(self, Ly: int, Lx: int, nframes: int, reg_file: str, raw_file: str):
        self.Ly = Ly
        self.Lx = Lx
        self.nframes = nframes
        self.reg_file = open(reg_file, mode='wb' if raw_file else 'r+b')
        self.raw_file = open(raw_file, 'rb') if raw_file else None

        self._nfr = 0
        self._index = 0
        self._can_read = True

    @property
    def nbytesread(self) -> int:
        return 2 * self.Ly * self.Lx

    def close(self) -> None:
        self.reg_file.close()
        if self.raw_file:
            self.raw_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def iter_frames(self, batch_size=1, dtype=np.float32):
        while True:
            data = self.read(batch_size=batch_size, dtype=dtype)
            if data is None:
                break
            yield data

    def read(self, batch_size=1, dtype=np.float32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._can_read:
            raise IOError("BinaryFile needs to write before it can read again.")
        nbytes = self.nbytesread * batch_size
        buff = self.raw_file.read(nbytes) if self.raw_file else self.reg_file.read(nbytes)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).reshape(-1, self.Ly, self.Lx).astype(dtype)
        if (data.size == 0) | (self._nfr >= self.nframes):
            return None
        self._nfr += data.size
        indices = np.arange(self._index, self._index + data.shape[0])
        self._index += data.shape[0]
        self._can_read = False
        return indices, data

    def write(self, data: np.ndarray) -> None:
        if self._can_read:
            raise IOError("BinaryFile needs to read before it can write again.")

        if not self.raw_file:
            self.reg_file.seek(-2 * data.size, 1)
        self.reg_file.write(bytearray(np.minimum(data, 2 ** 15 - 2).astype('int16')))
        self._can_read = True


def get_frames(Lx, Ly, xrange, yrange, ix, bin_file, crop=False):
    """ get frames ix from bin_file
        frames are cropped by ops['yrange'] and ops['xrange']

    Parameters
    ----------
    ops : dict
        requires 'Ly', 'Lx'
    ix : int, array
        frames to take
    bin_file : str
        location of binary file to read (frames x Ly x Lx)
    crop : bool
        whether or not to crop by 'yrange' and 'xrange' - if True, needed in ops

    Returns
    -------
        mov : int16, array
            frames x Ly x Lx
    """
    nbytesread =  np.int64(Ly*Lx*2)
    Lyc = yrange[-1] - yrange[0]
    Lxc = xrange[-1] - xrange[0]

    mov = np.zeros((len(ix), Lyc, Lxc), np.int16) if crop else np.zeros((len(ix), Ly, Lx), np.int16)
    # load and bin data
    with open(bin_file, 'rb') as bfile:
        for mov_i, ixx in zip(mov, ix):
            bfile.seek(nbytesread * ixx, 0)
            buff = bfile.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            data = np.reshape(data, (Ly, Lx))
            mov_i[:, :] = data[yrange[0]:yrange[-1], xrange[0]:xrange[-1]] if crop else data
    return mov