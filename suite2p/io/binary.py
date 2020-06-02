from typing import Optional

import numpy as np


class BinaryFile:

    def __init__(self, nbatch: int, Ly: int, Lx: int, nframes: int, reg_file: str, raw_file: str):
        self.nbatch = nbatch
        self.Ly = Ly
        self.Lx = Lx
        self.nframes = nframes
        self.reg_file = open(reg_file, mode='wb' if raw_file else 'r+b')
        self.raw_file = open(raw_file, 'rb') if raw_file else None

        self._nfr = 0
        self._can_read = True

    @property
    def nbytesread(self) -> int:
        return 2 * self.Ly * self.Lx * self.nbatch

    def close(self) -> None:
        self.reg_file.close()
        if self.raw_file:
            self.raw_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        data = self.read()
        if data is None:
            raise StopIteration
        return data

    def read(self, dtype=np.float32) -> Optional[np.ndarray]:
        if not self._can_read:
            raise IOError("BinaryFile needs to write before it can read again.")
        buff = self.raw_file.read(self.nbytesread) if self.raw_file else self.reg_file.read(self.nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).reshape(-1, self.Ly, self.Lx).astype(dtype)
        if (data.size == 0) | (self._nfr >= self.nframes):
            return None
        self._nfr += data.size
        self._can_read = False
        return data

    def write(self, data: np.ndarray) -> None:
        if self._can_read:
            raise IOError("BinaryFile needs to read before it can write again.")

        if not self.raw_file:
            self.reg_file.seek(-2 * data.size, 1)
        self.reg_file.write(bytearray(np.minimum(data, 2 ** 15 - 2).astype('int16')))
        self._can_read = True
