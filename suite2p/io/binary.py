from typing import Optional, Tuple, Sequence
from contextlib import contextmanager
import os

import numpy as np

class BinaryRWFile:
    def __init__(self, Ly: int, Lx: int, filename: str):
        """
        Creates/Opens a Suite2p BinaryFile for reading and/or writing image data that acts like numpy array

        Parameters
        ----------
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        filename: str
            The filename of the file to read from or write to
        """
        self.Ly = Ly
        self.Lx = Lx
        self.filename = filename
        if not os.path.exists(filename):
            self.file = open(filename, mode='w+b')
        else:
            self.file = open(filename, mode='r+b')
        self._index = 0
        self._can_read = True

    @staticmethod
    def convert_numpy_file_to_suite2p_binary(from_filename: str, to_filename: str) -> None:
        """
        Works with npz files, pickled npy files, etc.

        Parameters
        ----------
        from_filename: str
            The npy file to convert
        to_filename: str
            The binary file that will be created
        """
        np.load(from_filename).tofile(to_filename)

    @property
    def nbytesread(self):
        """number of bytes per frame (FIXED for given file)"""
        return np.int64(2 * self.Ly * self.Lx)

    @property
    def nbytes(self):
        """total number of bytes in the file."""
        with temporary_pointer(self.file) as f:
            f.seek(0, 2)
            return f.tell()

    @property
    def n_frames(self) -> int:
        """total number of frames in the file."""
        return int(self.nbytes // self.nbytesread)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        The dimensions of the data in the file

        Returns
        -------
        n_frames: int
            The number of frames
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        """
        return self.n_frames, self.Ly, self.Lx

    @property
    def size(self) -> int:
        """
        Returns the total number of pixels

        Returns
        -------
        size: int
        """
        return np.prod(np.array(self.shape).astype(np.int64))

    def close(self) -> None:
        """
        Closes the file.
        """
        self.file.close()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __setitem__(self, *items):
        frame_indices, data = items
        self.ix_write(data=data, indices=from_slice(frame_indices))
        
    def __getitem__(self, *items):
        frame_indices, *crop = items
        if isinstance(frame_indices, int):
            frames = self.ix(indices=[frame_indices], is_slice=False)
        elif isinstance(frame_indices, slice):
            frames = self.ix(indices=from_slice(frame_indices), is_slice=True)
        else:
            frames = self.ix(indices=frame_indices, is_slice=False)
        return frames[(slice(None),) + crop] if crop else frames

    def sampled_mean(self) -> float:
        """
        Returns the sampled mean.
        """
        n_frames = self.n_frames
        nsamps = min(n_frames, 1000)
        inds = np.linspace(0, n_frames, 1+nsamps).astype(np.int64)[:-1]
        frames = self.ix(indices=inds).astype(np.float32)
        return frames.mean(axis=0)

    def ix_write(self, data, indices: Sequence[int]):
        """
        Writes the frames at index values "indices".

        Parameters
        ----------
        indices: int array
            The frame indices to get, must be a slice

        """
        i0 = indices[0]
        batch_size = len(indices)
        if self._index != i0:
            self.file.seek(self.nbytesread * (i0 - self._index), 1)
        self._index = i0 + batch_size
        self.write(data)  

    def ix(self, indices: Sequence[int], is_slice=False):
        """
        Returns the frames at index values "indices".

        Parameters
        ----------
        indices: int array
            The frame indices to get

        is_slice: bool, default False
            if indices are slice, read slice with "read" function and return

        Returns
        -------
        frames: len(indices) x Ly x Lx
            The requested frames
        """
        if not is_slice:
            frames = np.empty((len(indices), self.Ly, self.Lx), np.int16)
            # load and bin data
            with temporary_pointer(self.file) as f:
                for frame, ixx in zip(frames, indices):
                    if ixx!=self._index:
                        f.seek(self.nbytesread * ixx)
                    buff = f.read(self.nbytesread)
                    data = np.frombuffer(buff, dtype=np.int16, offset=0)
                    frame[:] = np.reshape(data, (self.Ly, self.Lx))
                    #self._index = ixx+1
        else:
            i0 = indices[0]
            batch_size = len(indices)
            if self._index != i0:
                self.file.seek(self.nbytesread * i0)
            _, frames = self.read(batch_size=batch_size, dtype=np.int16)
            self._index = i0 + batch_size
        
        return frames

    @property
    def data(self) -> np.ndarray:
        """
        Returns all the frames in the file.

        Returns
        -------
        frames: nImg x Ly x Lx
            The frame data
        """
        with temporary_pointer(self.file) as f:
            return np.fromfile(f, np.int16).reshape(-1, self.Ly, self.Lx)

    def read(self, batch_size=1, dtype=np.float32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the next frame(s) in the file and its associated indices.

        Parameters
        ----------
        batch_size: int
            The number of frames to read at once.
        frames: batch_size x Ly x Lx
            The frame data
        """
        if not self._can_read:
            raise IOError("BinaryFile needs to write before it can read again.")
        nbytes = self.nbytesread * batch_size
        buff = self.file.read(nbytes)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).reshape(-1, self.Ly, self.Lx).astype(dtype)
        if data.size == 0:
            return None
        indices = np.arange(self._index, self._index + data.shape[0])
        self._index += data.shape[0]
        return indices, data

    def write(self, data: np.ndarray) -> None:
        """
        Writes frame(s) to the file.

        Parameters
        ----------
        data: 2D or 3D array
            The frame(s) to write.  Should be the same width and height as the other frames in the file.
        """
        self.file.write(bytearray(np.minimum(data, 2 ** 15 - 2).astype('int16')))

class BinaryFile:

    def __init__(self, Ly: int, Lx: int, read_filename: str, write_filename: Optional[str] = None):
        """
        Creates/Opens a Suite2p BinaryFile for reading and writing image data

        Parameters
        ----------
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        read_filename: str
            The filename of the file to read from
        write_filename: str
            The filename to write to, if different from the read_filename (optional)
        """
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

    @staticmethod
    def convert_numpy_file_to_suite2p_binary(from_filename: str, to_filename: str) -> None:
        """
        Works with npz files, pickled npy files, etc.

        Parameters
        ----------
        from_filename: str
            The npy file to convert
        to_filename: str
            The binary file that will be created
        """
        np.load(from_filename).tofile(to_filename)

    @property
    def nbytesread(self):
        """number of bytes per frame (FIXED for given file)"""
        return np.int64(2 * self.Ly * self.Lx)

    @property
    def nbytes(self):
        """total number of bytes in the read_file."""
        with temporary_pointer(self.read_file) as f:
            f.seek(0, 2)
            return f.tell()

    @property
    def n_frames(self) -> int:
        """total number of frames in the read_file."""
        return int(self.nbytes // self.nbytesread)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        The dimensions of the data in the file

        Returns
        -------
        n_frames: int
            The number of frames
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        """
        return self.n_frames, self.Ly, self.Lx

    @property
    def size(self) -> int:
        """
        Returns the total number of pixels

        Returns
        -------
        size: int
        """
        return np.prod(np.array(self.shape).astype(np.int64))

    def close(self) -> None:
        """
        Closes the file.
        """
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
            frames = self.ix(indices=[frame_indices], is_slice=False)
        elif isinstance(frame_indices, slice):
            frames = self.ix(indices=from_slice(frame_indices), is_slice=True)
        else:
            frames = self.ix(indices=frame_indices)
        return frames[(slice(None),) + crop] if crop else frames

    def sampled_mean(self) -> float:
        """
        Returns the sampled mean.
        """
        n_frames = self.n_frames
        nsamps = min(n_frames, 1000)
        inds = np.linspace(0, n_frames, 1+nsamps).astype(np.int64)[:-1]
        frames = self.ix(indices=inds).astype(np.float32)
        return frames.mean(axis=0)

    def iter_frames(self, batch_size: int = 1, dtype=np.float32):
        """
        Iterates through each set of frames, depending on batch_size, yielding both the frame index and frame data.

        Parameters
        ---------
        batch_size: int
            The number of frames to get at a time
        dtype: np.dtype
            The nympy data type that the data should return as

        Yields
        ------
        indices: array int
            The frame indices.
        data: batch_size x Ly x Lx
            The frames
        """
        while True:
            results = self.read(batch_size=batch_size, dtype=dtype)
            if results is None:
                break
            indices, data = results
            yield indices, data

    def ix(self, indices: Sequence[int], is_slice=False):
        """
        Returns the frames at index values "indices".

        Parameters
        ----------
        indices: int array
            The frame indices to get

        is_slice: bool, default False
            if indices are slice, read slice with "read" function and return

        Returns
        -------
        frames: len(indices) x Ly x Lx
            The requested frames
        """
        if not is_slice:
            frames = np.empty((len(indices), self.Ly, self.Lx), np.int16)
            # load and bin data
            with temporary_pointer(self.read_file) as f:
                for frame, ixx in zip(frames, indices):
                    if ixx!=self._index:
                        f.seek(self.nbytesread * ixx)
                    buff = f.read(self.nbytesread)
                    data = np.frombuffer(buff, dtype=np.int16, offset=0)
                    frame[:] = np.reshape(data, (self.Ly, self.Lx))
                    self._index = ixx+1
        else:
            i0 = indices[0]
            batch_size = len(indices)
            if self._index != i0:
                self.read_file.seek(self.nbytesread * i0)
            _, frames = self.read(batch_size=batch_size, dtype=np.int16)
            self._index = i0 + batch_size
        
        return frames

    @property
    def data(self) -> np.ndarray:
        """
        Returns all the frames in the file.

        Returns
        -------
        frames: nImg x Ly x Lx
            The frame data
        """
        with temporary_pointer(self.read_file) as f:
            return np.fromfile(f, np.int16).reshape(-1, self.Ly, self.Lx)

    def read(self, batch_size=1, dtype=np.float32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the next frame(s) in the file and its associated indices.

        Parameters
        ----------
        batch_size: int
            The number of frames to read at once.
        frames: batch_size x Ly x Lx
            The frame data
        """
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
        """
        Writes frame(s) to the file.

        Parameters
        ----------
        data: 2D or 3D array
            The frame(s) to write.  Should be the same width and height as the other frames in the file.
        """
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
        """
        Returns binned movie that rejects bad_frames (bool array) and crops to (y_range, x_range).

        Parameters
        ----------
        bin_size: int
            The size of each bin
        x_range: int, int
            Crops the data to a minimum and maximum x range.
        y_range: int, int
            Crops the data to a minimum and maximum y range.
        bad_frames: int array
            The indices to *not* include.
        reject_threshold: float

        Returns
        -------
        frames: nImg x Ly x Lx
            The frames
        """

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
    """Creates an np.arange() array from a Python slice object.  Helps provide numpy-like slicing interfaces."""
    return np.arange(s.start, s.stop, s.step) if any([s.start, s.stop, s.step]) else None


def binned_mean(mov: np.ndarray, bin_size) -> np.ndarray:
    """Returns an array with the mean of each time bin (of size 'bin_size')."""
    n_frames, Ly, Lx = mov.shape
    mov = mov[:(n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, Ly, Lx).astype(np.float32).mean(axis=1)


@contextmanager
def temporary_pointer(file):
    """context manager that resets file pointer location to its original place upon exit."""
    orig_pointer = file.tell()
    yield file
    file.seek(orig_pointer)

class BinaryFileCombined:

    def __init__(self, LY: int, LX: int, Ly: np.ndarray, Lx: np.ndarray, 
                 dy: np.ndarray, dx: np.ndarray, read_filenames: str):
        """
        Creates/Opens a Suite2p BinaryFile for reading image data across planes

        Parameters
        ----------
        LY: int
            The height of full frame
        LX: int
            The width of full frame
        Ly: numpy array of ints
            The heights of each frame
        Lx: numpy array of ints
            The widths of each frame
        dy: numpy array of ints
            The y-positions of each frame
        dx: numpy array of ints
            The x-positions of each frame
        read_filenames: array of str
            The filenames of the files to read from
        """
        self.LY = LY
        self.LX = LX
        self.Ly = Ly
        self.Lx = Lx
        self.dy = dy
        self.dx = dx
        self.read_filenames = read_filenames
        
        self.read_files = [open(read_filename, mode='rb') for read_filename in self.read_filenames]
        self._index = 0
        self._can_read = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """
        Closes the file.
        """
        for n in range(len(self.read_files)):
            self.read_files[n].close()

    @property
    def nbytesread(self):
        """number of bytes per frame (FIXED for given file)"""
        return (2 * self.Ly * self.Lx).astype(np.int64)

    @property
    def nbytes(self):
        """total number of bytes in the read_file."""
        nbytes = np.zeros(len(self.read_files), np.int64)
        for i,read_file in enumerate(self.read_files):
            with temporary_pointer(read_file) as f:
                f.seek(0, 2)
                nbytes[i] = f.tell()
        return nbytes

    @property
    def n_frames(self) -> int:
        """total number of fraames in the read_file."""
        return int(self.nbytes[0] // self.nbytesread[0])


    def read(self, batch_size=1, dtype=np.float32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the next frame(s) in the file and its associated indices.

        Parameters
        ----------
        batch_size: int
            The number of frames to read at once.
        frames: batch_size x Ly x Lx
            The frame data
        """
        if not self._can_read:
            raise IOError("BinaryFile needs to write before it can read again.")

        for n, (nbytesr, read_file) in enumerate(zip(self.nbytesread, self.read_files)):
            nbytes = nbytesr * batch_size
            buff = read_file.read(nbytes)
            data = np.frombuffer(buff, dtype=np.int16, offset=0).reshape(-1, self.Ly[n], self.Lx[n]).astype(dtype)
            if data.size == 0:
                return None
            if n==0:
                data_all = np.zeros((data.shape[0], self.LY, self.LX), dtype=np.int16)
            data_all[:, self.dy[n]:self.dy[n]+self.Ly[n], self.dx[n]:self.dx[n]+self.Lx[n]] = data
            
        indices = np.arange(self._index, self._index + data.shape[0])
        self._index += data.shape[0]
        
        return indices, data_all

    def __getitem__(self, *items):
        frame_indices, *crop = items
        if isinstance(frame_indices, int):
            frames = self.ix(indices=[frame_indices], is_slice=False)
        elif isinstance(frame_indices, slice):
            frames = self.ix(indices=from_slice(frame_indices), is_slice=True)
        else:
            frames = self.ix(indices=frame_indices)
        return frames[(slice(None),) + crop] if crop else frames


    def ix(self, indices: Sequence[int], is_slice=False):
        """
        Returns the frames at index values "indices".

        Parameters
        ----------
        indices: int array
            The frame indices to get

        is_slice: bool, default False
            if indices are slice, read slice with "read" function and return

        Returns
        -------
        frames: len(indices) x Ly x Lx
            The requested frames
        """
        for n, (nbytesr, read_file) in enumerate(zip(self.nbytesread, self.read_files)):
            
            if not is_slice:
                frames = np.empty((len(indices), self.Ly[n], self.Lx[n]), np.int16)
                # load and bin data
                with temporary_pointer(read_file) as f:
                    for frame, ixx in zip(frames, indices):
                        if ixx!=self._index:
                            f.seek(nbytesr * ixx)
                        buff = f.read(nbytesr)
                        data = np.frombuffer(buff, dtype=np.int16, offset=0)
                        frame[:] = np.reshape(data, (self.Ly[n], self.Lx[n]))
                        if n==len(self.Ly)-1:
                            self._index = ixx+1
            else:
                i0 = indices[0]
                batch_size = len(indices)
                if self._index != i0:
                    read_file.seek(nbytesr * i0)
                buff = read_file.read(nbytesr * batch_size)
                data = np.frombuffer(buff, dtype=np.int16, offset=0)
                frames = np.reshape(data, (-1, self.Ly[n], self.Lx[n]))
                if n==len(self.Ly)-1:
                    self._index = i0 + batch_size

            if frames.size == 0:
                return None
            if n==0:
                data_all = np.zeros((frames.shape[0], self.LY, self.LX), dtype=np.int16)
            data_all[:, self.dy[n]:self.dy[n]+self.Ly[n], self.dx[n]:self.dx[n]+self.Lx[n]] = frames

            
        return data_all

    def iter_frames(self, batch_size: int = 1, dtype=np.float32):
        """
        Iterates through each set of frames, depending on batch_size, yielding both the frame index and frame data.

        Parameters
        ---------
        batch_size: int
            The number of frames to get at a time
        dtype: np.dtype
            The nympy data type that the data should return as

        Yields
        ------
        indices: array int
            The frame indices.
        data: batch_size x Ly x Lx
            The frames
        """
        while True:
            results = self.read(batch_size=batch_size, dtype=dtype)
            if results is None:
                break
            indices, data = results
            yield indices, data


