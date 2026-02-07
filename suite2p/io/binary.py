"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from contextlib import contextmanager
from tifffile import TiffWriter
import logging 
logger = logging.getLogger(__name__)

import os

import numpy as np


class BinaryFile:

    def __init__(self, Ly, Lx, filename, n_frames=None,
                 dtype="int16", write=False):
        """
        Open or create a Suite2p binary file backed by a memory-mapped numpy array.

        Parameters
        ----------
        Ly : int
            Height of each frame in pixels.
        Lx : int
            Width of each frame in pixels.
        filename : str
            Path to the binary file to read from or write to.
        n_frames : int, optional
            Number of frames. Required when creating a new file for writing.
            Inferred from file size when reading.
        dtype : str, optional (default "int16")
            Data type of each pixel value.
        write : bool, optional (default False)
            If True, open the file for reading and writing. If False, open read-only.
        """
        self.Ly = Ly
        self.Lx = Lx
        self.filename = filename
        self.dtype = dtype
        self.write = write

        if write and n_frames is None and not os.path.exists(self.filename):
            raise ValueError(
                "need to provide number of frames n_frames when writing file")
        elif not write:
            n_frames = self.n_frames
        shape = (n_frames, self.Ly, self.Lx)
        if write:
            mode = "r+" if os.path.exists(self.filename) else "w+"
        else:
            mode = "r"
        self.file = np.memmap(self.filename, mode=mode, dtype=self.dtype, shape=shape)
        self._index = 0
        self._can_read = True

    @staticmethod
    def convert_numpy_file_to_suite2p_binary(from_filename, to_filename):
        """
        Convert a numpy file (.npy, .npz) to a Suite2p binary file.

        Parameters
        ----------
        from_filename : str
            Path to the source numpy file to convert.
        to_filename : str
            Path to the binary file that will be created.
        """
        np.load(from_filename).tofile(to_filename)

    @property
    def nbytesread(self):
        """Number of bytes per frame (fixed for a given file)."""
        return np.int64(2 * self.Ly * self.Lx)

    @property
    def nbytes(self):
        """Total number of bytes in the file."""
        return os.path.getsize(self.filename)

    @property
    def n_frames(self):
        """Total number of frames in the file."""
        return int(self.nbytes // self.nbytesread)

    @property
    def shape(self):
        """
        Return the dimensions of the data in the file.

        Returns
        -------
        n_frames : int
            Number of frames.
        Ly : int
            Height of each frame in pixels.
        Lx : int
            Width of each frame in pixels.
        """
        return self.n_frames, self.Ly, self.Lx

    @property
    def size(self):
        """
        Return the total number of pixels across all frames.

        Returns
        -------
        size : int
            Product of n_frames * Ly * Lx.
        """
        return np.prod(np.array(self.shape).astype(np.int64))

    def close(self):
        """
        Closes the file.
        """
        self.file._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __setitem__(self, *items):
        indices, data = items
        if data.dtype != "int16":
            self.file[indices] = np.minimum(data, 2**15 - 2).astype("int16")
        else:
            self.file[indices] = data

    def __getitem__(self, *items):
        indices, *crop = items
        return self.file[indices]

    def sampled_mean(self):
        """
        Compute the mean image from up to 1000 evenly spaced frames.

        Returns
        -------
        mean_img : numpy.ndarray
            Mean image of shape (Ly, Lx), averaged over the sampled frames.
        """
        n_frames = self.n_frames
        nsamps = min(n_frames, 1000)
        inds = np.linspace(0, n_frames, 1 + nsamps).astype(np.int64)[:-1]
        frames = self.file[inds].astype(np.float32)
        return frames.mean(axis=0)

    @property
    def data(self):
        """
        Return all frames in the file.

        Returns
        -------
        frames : numpy.ndarray
            All frame data as an array of shape (n_frames, Ly, Lx).
        """
        return self.file[:]

    def write_tiff(self, fname, range_dict={}):
        """
        Write binary file contents to a TIFF file, optionally cropped by frame/y/x ranges.

        Parameters
        ----------
        fname : str
            Output TIFF file path.
        range_dict : dict, optional
            Dictionary with optional keys "frame_range", "y_range", "x_range", each
            a tuple of (start, stop) indices. Defaults to the full extent of each
            dimension.
        """
        n_frames, Ly, Lx = self.shape
        frame_range, y_range, x_range = (0,n_frames), (0, Ly), (0, Lx)
        with TiffWriter(fname, bigtiff=True) as f:
            # Iterate through current data and write each frame to a tiff
            # All ranges should be Tuples(int,int)
            if 'frame_range' in range_dict:
                frame_range = range_dict['frame_range']
            if 'x_range' in range_dict:
                x_range = range_dict['x_range']
            if 'y_range' in range_dict:
                y_range = range_dict['y_range']
            logger.info('Frame Range: {}, y_range: {}, x_range{}'.format(frame_range, y_range, x_range))
            for i in range(frame_range[0], frame_range[1]):
                curr_frame = np.floor(self.file[i, y_range[0]:y_range[1], x_range[0]:x_range[1]]).astype(np.int16)
                f.write(curr_frame, contiguous=True)
        logger.info('Tiff has been saved to {}'.format(fname))


@contextmanager
def temporary_pointer(file):
    """
    Context manager that saves and restores a file's pointer position.

    Parameters
    ----------
    file : file object
        An open file with tell() and seek() methods.
    """
    orig_pointer = file.tell()
    yield file
    file.seek(orig_pointer)


class BinaryFileCombined:

    def __init__(self, LY, LX, Ly, Lx, dy, dx, read_filenames):
        """
        Open multiple Suite2p binary files for combined reading across ROIs/planes.

        Stitches multiple binary files into a single full-frame view by placing each
        ROI at its (dy, dx) offset within a canvas of size (LY, LX).

        Parameters
        ----------
        LY : int
            Height of the full combined frame in pixels.
        LX : int
            Width of the full combined frame in pixels.
        Ly : numpy.ndarray
            Array of per-ROI frame heights.
        Lx : numpy.ndarray
            Array of per-ROI frame widths.
        dy : numpy.ndarray
            Array of y-offsets for placing each ROI in the full frame.
        dx : numpy.ndarray
            Array of x-offsets for placing each ROI in the full frame.
        read_filenames : list of str
            Paths to the binary files to read, one per ROI.
        """
        self.LY = LY
        self.LX = LX
        self.Ly = Ly
        self.Lx = Lx
        self.dy = dy
        self.dx = dx
        self.read_filenames = read_filenames

        self.read_files = [
            BinaryFile(ly, lx, read_filename)
            for (ly, lx, read_filename) in zip(self.Ly, self.Lx, self.read_filenames)
        ]
        n_frames = np.zeros(len(self.read_files))
        for i, rf in enumerate(self.read_files):
            n_frames[i] = rf.n_frames
        assert (n_frames == n_frames[0]).sum() == len(self.read_files)
        self._index = 0
        self._can_read = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all underlying binary files."""
        for n in range(len(self.read_files)):
            self.read_files[n].close()

    @property
    def nbytes(self):
        """Total number of bytes per ROI file, as a numpy array."""
        nbytes = np.zeros(len(self.read_files), np.int64)
        for i, read_file in enumerate(self.read_files):
            nbytes[i] = read_file.nbytes
        return nbytes

    @property
    def n_frames(self):
        """Total number of frames (from the first ROI file)."""
        return self.read_files[0].n_frames

    def __getitem__(self, *items):
        indices, *crop = items
        data0 = self.read_files[0][indices]
        data_all = np.zeros((data0.shape[0], self.LY, self.LX), "int16")
        for n, read_file in enumerate(self.read_files):
            if n > 0:
                data0 = self.read_file[indices]
            data_all[:, self.dy[n]:self.dy[n] + self.Ly[n],
                     self.dx[n]:self.dx[n] + self.Lx[n]] = data0

        return data_all
