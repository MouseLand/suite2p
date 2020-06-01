from contextlib import ExitStack
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from . import bidiphase, nonrigid, utils, rigid


#HAS_GPU=False
#try:
#    import cupy as cp
#    from cupyx.scipy.fftpack import fftn, ifftn, get_fft_plan
#    HAS_GPU=True
#except ImportError:
#    HAS_GPU=False


def compute_motion_and_shift(data, bidiphase, bidi_corrected, refAndMasks, maxregshift, nblocks, xblock, yblock,
                             nr_sm, snr_thresh, smooth_sigma_time, maxregshiftNR,
                             is_nonrigid, reg_1p, spatial_hp, pre_smooth,
                             ):
    """ register data matrix to reference image and shift

    need to run ```refAndMasks = register.prepare_refAndMasks(ops)``` to get fft'ed masks;
    if ```ops['nonrigid']``` need to run ```ops = nonrigid.make_blocks(ops)```

    Parameters
    ----------
    data : int16
        array that's frames x Ly x Lx
    refAndMasks : list
        maskMul, maskOffset and cfRefImg (from prepare_refAndMasks)
    ops : dictionary
        requires 'nonrigid', 'bidiphase', '1Preg'

    Returns
    -------
    data : int16 (or float32, if ops['nonrigid'])
        registered frames x Ly x Lx
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame
    cmax : float
        maximum of phase correlation for each frame
    yxnr : list
        ymax, xmax and cmax from the non-rigid registration

    """

    if bidiphase and not bidi_corrected:
        bidiphase.shift(data, bidiphase)

    yxnr = []
    if smooth_sigma_time > 0:
        data_smooth = gaussian_filter1d(data, sigma=smooth_sigma_time, axis=0)

    # rigid registration
    ymax, xmax, cmax = rigid.phasecorr(
        data=data_smooth if smooth_sigma_time > 0 else data,
        refAndMasks=refAndMasks[:3],
        maxregshift=maxregshift,
        reg_1p=reg_1p,
        spatial_hp=spatial_hp,
        pre_smooth=pre_smooth,
        smooth_sigma_time=smooth_sigma_time,
    )
    rigid.shift_data(data, ymax, xmax)

    # non-rigid registration
    if is_nonrigid and len(refAndMasks)>3:
        # preprocessing for 1P recordings
        if reg_1p:
            data = utils.one_photon_preprocess(
                data=data,
                spatial_hp=spatial_hp,
                pre_smooth=pre_smooth,
            )

        ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(
            data=data_smooth if smooth_sigma_time > 0 else data,
            refAndMasks=refAndMasks[3:],
            snr_thresh=snr_thresh,
            NRsm=nr_sm,
            xblock=xblock,
            yblock=yblock,
            maxregshiftNR=maxregshiftNR,
        )
        yxnr = [ymax1, xmax1, cmax1]
        data = nonrigid.transform_data(
            data=data,
            nblocks=nblocks,
            xblock=xblock,
            yblock=yblock,
            ymax1=ymax1,
            xmax1=xmax1
        )
    return data, ymax, xmax, cmax, yxnr

def compute_crop(xoff, yoff, corrXY, th_badframes, badframes, maxregshift, Ly, Lx):
    """ determines how much to crop FOV based on motion
    
    determines badframes which are frames with large outlier shifts
    (threshold of outlier is th_badframes) and
    it excludes these badframes when computing valid ranges
    from registration in y and x
    """
    dx = xoff - medfilt(xoff, 101)
    dy = yoff - medfilt(yoff, 101)
    # offset in x and y (normed by mean offset)
    dxy = (dx**2 + dy**2)**.5
    dxy /= dxy.mean()
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = corrXY / medfilt(corrXY, 101)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    badframes = np.logical_or(px > th_badframes * 100, badframes)
    badframes = np.logical_or(abs(xoff) > (maxregshift * Lx * 0.95), badframes)
    badframes = np.logical_or(abs(yoff) > (maxregshift * Ly * 0.95), badframes)
    ymin = np.ceil(np.abs(yoff[np.logical_not(badframes)]).max())
    ymax = Ly - ymin
    xmin = np.ceil(np.abs(xoff[np.logical_not(badframes)]).max())
    xmax = Lx - xmin
    # ymin = np.maximum(0, np.ceil(np.amax(yoff[np.logical_not(badframes)])))
    # ymax = Ly + np.minimum(0, np.floor(np.amin(yoff)))
    # xmin = np.maximum(0, np.ceil(np.amax(xoff[np.logical_not(badframes)])))
    # xmax = Lx + np.minimum(0, np.floor(np.amin(xoff)))
    yrange = [int(ymin), int(ymax)]
    xrange = [int(xmin), int(xmax)]

    return badframes, yrange, xrange


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
    def nbytesread(self):
        return 2 * self.Ly * self.Lx * self.nbatch

    def close(self):
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

    def write(self, data):
        if self._can_read:
            raise IOError("BinaryFile needs to read before it can write again.")

        if not self.raw_file:
            self.reg_file.seek(-2 * data.size, 1)
        self.reg_file.write(bytearray(np.minimum(data, 2 ** 15 - 2).astype('int16')))
        self._can_read = True


def register_binary_to_ref(nbatch: int, Ly: int, Lx: int, nframes: int, ops, refAndMasks, reg_file_align, raw_file_align):

    with BinaryFile(nbatch=nbatch, Ly=Ly, Lx=Lx, nframes=nframes, reg_file=reg_file_align, raw_file=raw_file_align) as f:
        for data in f:

            data, ymax, xmax, cmax, yxnr = compute_motion_and_shift(
                data=data,
                bidiphase=ops['bidiphase'],
                bidi_corrected=ops['bidi_corrected'],
                refAndMasks=refAndMasks,
                maxregshift=ops['maxregshift'],
                nblocks=ops['nblocks'],
                xblock=ops['xblock'],
                yblock=ops['yblock'],
                nr_sm=ops['NRsm'],
                snr_thresh=ops['snr_thresh'],
                smooth_sigma_time=ops['smooth_sigma_time'],
                maxregshiftNR=ops['maxregshiftNR'],
                is_nonrigid=ops['nonrigid'],
                reg_1p=ops['1Preg'],
                spatial_hp=ops['spatial_hp'],
                pre_smooth=ops['pre_smooth'],
            )

            # output
            rigid_offset = [ymax, xmax, cmax]
            nonrigid_offset = list(yxnr)
            yield rigid_offset, nonrigid_offset, data

            f.write(data)




def apply_shifts_to_binary(batch_size: int, Ly: int, Lx: int, nframes: int,
    is_nonrigid: bool, bidiphase_value: int, bidi_corrected, nblocks, xblock, yblock,
    offsets, reg_file_alt, raw_file_alt):
    """ apply registration shifts computed on one binary file to another
    
    Parameters
    ----------

    offsets : list of arrays
        shifts computed from reg_file_align/raw_file_align, 
        rigid shifts in Y are offsets[0] and in X are offsets[1], 
        nonrigid shifts in Y are offsets[3] and in X are offsets[4]

    reg_file_alt : string
        file to (read if raw_file_align empty, and) write registered binary to

    raw_file_align : string
        file to read raw binary from (if not empty)
    """
    nbytesread = 2 * Ly * Lx * batch_size
    nfr = 0

    raw = len(raw_file_alt) > 0
    with open(reg_file_alt, mode='wb' if raw else 'r+b') as reg_file_alt, ExitStack() as stack:
        if raw:
            raw_file_alt = stack.enter_context(open(raw_file_alt, 'rb'))
        while True:
            data = np.frombuffer(
                raw_file_alt.read(nbytesread) if raw else reg_file_alt.read(nbytesread),
                dtype=np.int16,
                offset=0,
            ).copy()
            if (data.size == 0) | (nfr >= nframes):
                break
            data = np.reshape(data, (-1, Ly, Lx))



            nframes = data.shape[0]
            iframes = nfr + np.arange(0, nframes, 1, int)

            # get shifts
            ymax, xmax = offsets[0][iframes].astype(np.int32), offsets[1][iframes].astype(np.int32)
            ymax1, xmax1 = [], []
            if is_nonrigid:
                ymax1, xmax1 = offsets[3][iframes], offsets[4][iframes]

            # apply shifts
            if bidiphase_value != 0 and not bidi_corrected:
                bidiphase.shift(data, bidiphase_value)
            rigid.shift_data(data, ymax, xmax)
            if is_nonrigid:
                data = nonrigid.transform_data(data, nblocks=nblocks, xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
            # data = np.minimum(data, 2 ** 15 - 2)

            # write to binary
            if not raw:
                reg_file_alt.seek(-2 * data.size, 1)
            reg_file_alt.write(bytearray(np.minimum(data, 2 ** 15 - 2).astype('int16')))

            nfr += nframes

            yield data


def pick_initial_reference(frames):
    """ computes the initial reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    Parameters
    ----------
    frames : 3D array, int16
        size [frames x Ly x Lx], frames from binary

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """
    nimg,Ly,Lx = frames.shape
    frames = np.reshape(frames, (nimg,-1)).astype('float32')
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
    cc = np.matmul(frames, frames.T)
    ndiag = np.sqrt(np.diag(cc))
    cc = cc / np.outer(ndiag, ndiag)
    CCsort = -np.sort(-cc, axis = 1)
    bestCC = np.mean(CCsort[:, 1:20], axis=1);
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    refImg = np.mean(frames[indsort[0:20], :], axis = 0)
    refImg = np.reshape(refImg, (Ly,Lx))
    return refImg

