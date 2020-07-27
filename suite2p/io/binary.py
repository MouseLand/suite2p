from typing import Optional, Tuple, Sequence

import numpy as np


class BinaryFile:

    def __init__(self, Ly: int, Lx: int, read_filename: str, write_filename: Optional[str] = None):
        self.Ly = Ly
        self.Lx = Lx
        self.read_filename = read_filename
        self.write_filename = write_filename

        if read_filename == write_filename:
            self.read_file = open(read_filename, mode='r+b')
            self.write_file = self.read_file
        elif read_filename and not write_filename:
            self.read_file = open(read_filename, mode='rb')
            self.write_file = write_filename
        elif read_filename and write_filename and read_filename != write_filename:
            self.read_file = open(read_filename, mode='rb')
            self.write_file = open(write_filename, mode='wb')
        else:
            raise IOError("Invalid combination of read_file and write_file")

        self._index = 0
        self._can_read = True

    @property
    def nbytesread(self):
        """number of bytes per frame (FIXED for given file)"""
        return np.int64(2 * self.Ly * self.Lx)

    @property
    def nbytes(self):
        """total number of bytes in the read_file."""
        currpos = self.read_file.tell()
        self.read_file.seek(0, 2)
        size = self.read_file.tell()
        print(size)
        self.read_file.seek(currpos)
        return size

    @property
    def n_frames(self) -> int:
        """total number of fraames in the read_file."""
        return int(self.nbytes // self.nbytesread)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.n_frames, self.Ly, self.Lx

    @property
    def size(self):
        return np.prod(np.array(self.shape).astype(np.int64))

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

    def bin_movie(self, bin_size: int, x_range: Optional[Tuple[int, int]] = None, y_range: Optional[Tuple[int, int]] = None,
                  bad_frames: Optional[np.ndarray] = None, reject_threshold: float = 0.5) -> np.ndarray:
        """Returns binned movie that rejects bad_frames (bool array) and crops to (y_range, x_range)."""

        good_frames = ~bad_frames if bad_frames is not None else np.ones(self.n_frames, dtype=bool)

        batch_size = min(np.sum(good_frames), 500)
        batches = []
        for indices, data in self.iter_frames(batch_size=batch_size):
            if len(data) != batch_size:
                break

            if x_range is not None and y_range is not None:
                data = data[:, slice(*y_range), slice(*x_range)]  # crop

            good_indices = good_frames[indices]
            if np.mean(good_indices) > reject_threshold:
                data = data[good_indices]

            if data.shape[0] > bin_size:
                data = binned_mean(mov=data, bin_size=bin_size)
                batches.extend(data)

        mov = np.stack(batches)
        return mov


def from_slice(s: slice) -> Optional[np.ndarray]:
    return np.arange(s.start, s.stop, s.step) if any([s.start, s.stop, s.step]) else None


def binned_mean(mov: np.ndarray, bin_size) -> np.ndarray:
    """Returns an array with the mean of each time bin (of size 'bin_size')."""
    n_frames, Ly, Lx = mov.shape
    mov = mov[:(n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, Ly, Lx).mean(axis=1)
