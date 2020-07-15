from typing import Optional, Tuple, Sequence

import numpy as np


def from_slice(s: slice) -> np.ndarray:
    if s.start == None and s.stop == None and s.step == None:
        return None
    else:
        return np.arange(s.start, s.stop, s.step)


class BinaryFile:

    def __init__(self, Ly: int, Lx: int, read_file: str, write_file: Optional[str] = None):
        self.Ly = Ly
        self.Lx = Lx
        # bytes per frame (FIXED for given file)
        self.nbytesread = np.int64(2 * self.Ly * self.Lx)
        if read_file == write_file:
            self.read_file = open(read_file, mode='r+b')
            self.write_file = self.read_file
        elif read_file and not write_file:
            self.read_file = open(read_file, mode='rb')
            self.write_file = write_file
        elif read_file and write_file and read_file != write_file:
            self.read_file = open(read_file, mode='rb')
            self.write_file = open(write_file, mode='wb')
        else:
            raise IOError("Invalid combination of read_file and write_file")

        self._index = 0
        self._can_read = True

    @property
    def n_frames(self) -> int:
        return int(self.nbytesread / self.Ly / self.Lx)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.n_frames, self.Ly, self.Lx

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    def close(self) -> None:
        self.read_file.close()
        if self.write_file:
            self.write_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, *items):
        frame_indices, *crop = items
        if isinstance(frame_indices, int):
            frames = self.ix(indices=[frame_indices])
        elif isinstance(frame_indices, slice):
            frames = self.ix(indices=from_slice(frame_indices))
        else:
            frames = self.ix(indices=frame_indices)
        return frames[(slice(None),) + crop] if crop else frames

    def sampled_mean(self):
        n_frames = self.n_frames
        nsamps = min(n_frames, 1000)
        inds = np.linspace(0, n_frames, 1+nsamps).astype(np.int64)[:-1]
        frames = self.ix(indices=inds).astype(np.float32)
        return frames.mean(axis=0)

    def iter_frames(self, batch_size=1, dtype=np.float32):
        while True:
            results = self.read(batch_size=batch_size, dtype=dtype)
            if results is None:
                break
            indices, data = results
            yield indices, data

    def ix(self, indices: Sequence[int]):
        frames = np.empty((len(indices), self.Ly, self.Lx), np.int16)
        # load and bin data
        orig_ptr = self.read_file.tell()
        for frame, ixx in zip(frames, indices):
            self.read_file.seek(self.nbytesread * ixx)
            buff = self.read_file.read(self.nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            frame[:] = np.reshape(data, (self.Ly, self.Lx))
        self.read_file.seek(orig_ptr)
        return frames

    def read(self, batch_size=1, dtype=np.float32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._can_read:
            raise IOError("BinaryFile needs to write before it can read again.")
        nbytes = self.nbytesread * batch_size
        buff = self.read_file.read(nbytes)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).reshape(-1, self.Ly, self.Lx).astype(dtype)
        if data.size == 0:
            return None
        indices = np.arange(self._index, self._index + data.shape[0])
        self._index += data.shape[0]
        if self.read_file is self.write_file:
            self._can_read = False
        return indices, data

    def write(self, data: np.ndarray) -> None:
        if self._can_read and self.read_file is self.write_file:
            raise IOError("BinaryFile needs to read before it can write again.")
        if not self.write_file:
            raise IOError("No write_file specified, writing not possible.")
        if self.read_file is self.write_file:
            self.write_file.seek(-2 * data.size, 1)
            self._can_read = True
        self.write_file.write(bytearray(np.minimum(data, 2 ** 15 - 2).astype('int16')))
