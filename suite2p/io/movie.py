try:
    import cv2
    HAS_CV2 = True 
except:
    HAS_CV2 = False 

import numpy as np
import time
from typing import Optional, Tuple, Sequence
import logging 
logger = logging.getLogger(__name__)

from .utils import find_files_open_binaries, init_settings

class VideoReader:
    """ Uses cv2 to read video files """
    def __init__(self, filenames: list):
        """ Uses cv2 to open video files and obtain their details for reading

        Parameters
        ------------
        filenames : int
            list of video files
        """
        cumframes = [0]
        containers = []
        Ly = []
        Lx = []
        for f in filenames:  # for each video in the list
            cap = cv2.VideoCapture(f)
            containers.append(cap)
            Lx.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            Ly.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cumframes.append(cumframes[-1] + int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cumframes = np.array(cumframes).astype(int)
        Ly = np.array(Ly)
        Lx = np.array(Lx)
        if (Ly==Ly[0]).sum() < len(Ly) or (Lx==Lx[0]).sum() < len(Lx):
            raise ValueError("videos are not all the same size in y and x")
        else:
            Ly, Lx = Ly[0], Lx[0]

        self.filenames = filenames
        self.cumframes = cumframes 
        self.n_frames = cumframes[-1]
        self.Ly = Ly
        self.Lx = Lx
        self.containers = containers
        self.fs = containers[0].get(cv2.CAP_PROP_FPS)

    def close(self) -> None:
        """
        Closes the video files
        """
        for i in range(len(self.containers)):  # for each video in the list
            cap = self.containers[i]
            cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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

    def get_frames(self, cframes):
        """
        read frames "cframes" from videos

        Parameters
        ------------
        cframes : np.array
            start and stop of frames to read, or consecutive list of frames to read
        """
        cframes = np.maximum(0, np.minimum(self.n_frames - 1, cframes))
        cframes = np.arange(cframes[0], cframes[-1] + 1).astype(int)
        # find which video the frames exist in (ivids is length of cframes)
        ivids = (cframes[np.newaxis, :] >= self.cumframes[1:, np.newaxis]).sum(axis=0)
        nk = 0
        im = np.zeros((len(cframes), self.Ly, self.Lx), "uint8")
        for n in np.unique(ivids):  # for each video in cumframes
            cfr = cframes[ivids == n]
            start = cfr[0] - self.cumframes[n]
            end = cfr[-1] - self.cumframes[n] + 1
            nt0 = end - start
            capture = self.containers[n]
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != start:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            fc = 0
            ret = True
            while fc < nt0 and ret:
                ret, frame = capture.read()
                if ret:
                    im[nk + fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    logger.info("img load failed, replacing with prev..")
                    im[nk + fc] = im[nk + fc - 1]
                fc += 1
            nk += nt0
        return im

def movie_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """  finds movie files and writes them to binaries

    Parameters
    ----------
    dbs : list of dict
        Database dictionaries for each plane. Must contain keys "file_list",
        "nplanes", "nchannels", "batch_size", "h5py_key", and "functional_chan".
        Updated in-place with "Ly", "Lx", "nframes", "nframes_per_folder",
        "meanImg", and "meanImg_chan2".
    settings : dict
        Suite2p settings dictionary, saved alongside each plane's database.
    reg_file : list of file objects
        Opened binary files for writing each plane's functional channel data.
    reg_file_chan2 : list of file objects
        Opened binary files for writing each plane's second channel data
        (used only when nchannels > 1).

    Returns
    -------
    dbs : list of dict
        Updated database dictionaries with image dimensions, frame counts, and
        mean images populated.
    """
    if not HAS_CV2:
        raise ImportError("cv2 is required for this file type, please 'pip install opencv-python-headless'")

    
    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    flist = dbs[0]["file_list"]

    # todo: implement for multiple movies
    # for j in range(nplanes):
    #    dbs[j]["nframes_per_folder"] = np.zeros(len(flist), np.int32)

    ncp = nplanes * nchannels
    nbatch = ncp * int(np.ceil(dbs[0]["batch_size"] / ncp))
    logger.info(flist)
    t0 = time.time()
    with VideoReader(filenames=flist) as vr:
        if settings["fs"]<=0:
            settings["fs"] = vr.fs
            
        nframes_all = vr.cumframes[-1]
        nbatch = min(nbatch, nframes_all)
        nfunc = settings["functional_chan"] - 1 if nchannels > 1 else 0
        # loop over all video frames
        ik = 0
        while 1:
            irange = np.arange(ik, min(ik + nbatch, nframes_all), 1)
            if irange.size == 0:
                break
            im = vr.get_frames(irange).astype("int16")
            nframes = im.shape[0]
            for j in range(0, nplanes):
                if ik == 0:
                    dbs[j]["meanImg"] = np.zeros((im.shape[1], im.shape[2]),
                                                    np.float32)
                    if nchannels > 1:
                        dbs[j]["meanImg_chan2"] = np.zeros(
                            (im.shape[1], im.shape[2]), np.float32)
                    dbs[j]["nframes"] = 0
                i0 = nchannels * ((j) % nplanes)
                im2write = im[np.arange(int(i0) +
                                        nfunc, nframes, ncp), :, :].astype(
                                            np.int16)
                reg_file[j].write(bytearray(im2write))
                dbs[j]["meanImg"] += im2write.astype(np.float32).sum(axis=0)
                if nchannels > 1:
                    im2write = im[np.arange(int(i0) + 1 -
                                            nfunc, nframes, ncp), :, :].astype(
                                                np.int16)
                    reg_file_chan2[j].write(bytearray(im2write))
                    dbs[j]["meanImg_chan2"] += im2write.astype(
                        np.float32).sum(axis=0)
                dbs[j]["nframes"] += im2write.shape[0]
                
            ik += nframes
            if ik % (nbatch * 4) == 0:
                logger.info("%d frames of binary, time %0.2f sec." %
                      (ik, time.time() - t0))

    # write settings files
    # update dbs with image dimensions and mean images
    do_registration = settings["run"]["do_registration"]
    for db in dbs:
        db["Ly"] = im2write.shape[1]
        db["Lx"] = im2write.shape[2]
        if not do_registration:
            db["yrange"] = np.array([0, db["Ly"]])
            db["xrange"] = np.array([0, db["Lx"]])
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        # Save db and settings to each plane folder
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    return dbs