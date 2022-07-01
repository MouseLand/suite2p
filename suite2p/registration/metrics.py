from multiprocessing import Pool

import numpy as np
from numpy.linalg import norm
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from . import rigid, nonrigid, utils, bidiphase
from .. import io

def pclowhigh(mov, nlowhigh, nPC, random_state):
    """
    Compute mean of top and bottom PC weights for nPC's of mov

    computes nPC PCs of mov and returns average of top and bottom

    Parameters
    ----------
    mov : frames x Ly x Lx
        subsampled frames from movie
    nlowhigh : int
        number of frames to average at top and bottom of each PC
    nPC : int
        number of PCs to compute
    random_state:
        a value that sets the seed for the PCA randomizer.

    Returns
    -------
        pclow : float, array
            average of bottom of spatial PC: nPC x Ly x Lx
        pchigh : float, array
            average of top of spatial PC: nPC x Ly x Lx
        w : float, array
            singular values of decomposition of mov
        v : float, array
            frames x nPC, how the PCs vary across frames
    """
    nframes, Ly, Lx = mov.shape
    mov = mov.reshape((nframes, -1))
    mov = mov.astype(np.float32)
    mimg = mov.mean(axis=0)
    mov -= mimg
    pca = PCA(n_components=nPC, random_state=random_state).fit(mov.T)
    v = pca.components_.T
    w = pca.singular_values_
    mov += mimg
    mov = np.transpose(np.reshape(mov, (-1, Ly, Lx)), (1,2,0))
    pclow  = np.zeros((nPC, Ly, Lx), np.float32)
    pchigh = np.zeros((nPC, Ly, Lx), np.float32)
    isort = np.argsort(v, axis=0)
    for i in range(nPC):
        pclow[i] = mov[:,:,isort[:nlowhigh, i]].mean(axis=-1)
        pchigh[i] = mov[:,:,isort[-nlowhigh:, i]].mean(axis=-1)
    return pclow, pchigh, w, v


def pc_register(pclow, pchigh, bidi_corrected, spatial_hp=None, pre_smooth=None, smooth_sigma=1.15, smooth_sigma_time=0,
                block_size=(128,128), maxregshift=0.1, maxregshiftNR=10, reg_1p=False, snr_thresh=1.25,
                is_nonrigid=True, bidiphase_offset=0, spatial_taper=50.0):
    """
    register top and bottom of PCs to each other

    Parameters
    ----------
    pclow : float, array
        average of bottom of spatial PC: nPC x Ly x Lx
    pchigh : float, array
        average of top of spatial PC: nPC x Ly x Lx
    bidi_corrected: bool
        whether to do bidi correction.
    spatial_hp: int
        high-pass filter window size for the spatial dimensions
    pre_smooth: int
        low-pass filter window size for the spatial dimensions
    smooth_sigma : int
        see registration settings
    smooth_sigma_time: int
        see registration settings
    block_size : int, int
        see registration settings
    maxregshift : float
        see registration settings
    maxregshiftNR : int
        see registration settings
    reg_1p : bool
        see 1Preg settings
    snr_thresh: float
        signal to noise threshold to use.
    is_nonrigid: bool
    bidiphase_offset: int
    spatial_taper: float

    Returns
    -------
        X : float array
            nPC x 3 where X[:,0] is rigid, X[:,1] is average nonrigid, X[:,2] is max nonrigid shifts
    """
    # registration settings
    nPC, Ly, Lx = pclow.shape
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(
        Ly=Ly, Lx=Lx, block_size=np.array(block_size)
    )
    maxregshiftNR = np.array(maxregshiftNR)

    X = np.zeros((nPC,3))
    for i in range(nPC):
        refImg = pclow[i]
        Img = pchigh[i][np.newaxis, :, :]

        if reg_1p:
            data = refImg
            data = data.astype(np.float32)
            if pre_smooth:
                data = utils.spatial_smooth(data, int(pre_smooth))
            refImg = utils.spatial_high_pass(data, int(spatial_hp))

        rmin, rmax = np.int16(np.percentile(refImg,1)), np.int16(np.percentile(refImg,99))
        refImg = np.clip(refImg, rmin, rmax)

        maskMul, maskOffset = rigid.compute_masks(
            refImg=refImg,
            maskSlope=spatial_taper if reg_1p else 3 * smooth_sigma
        )
        cfRefImg = rigid.phasecorr_reference(
            refImg=refImg,
            smooth_sigma=smooth_sigma,
        )

        cfRefImg = cfRefImg[np.newaxis, :, :]
        if is_nonrigid:
            maskSlope = spatial_taper if reg_1p else 3 * smooth_sigma  # slope of taper mask at the edges

            maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
                refImg0=refImg,
                maskSlope=maskSlope,
                smooth_sigma=smooth_sigma,
                yblock=yblock,
                xblock=xblock,
            )



        if bidiphase_offset and not bidi_corrected:
            bidiphase.shift(Img, bidiphase_offset)

        # preprocessing for 1P recordings
        dwrite = Img.astype(np.float32)
        if reg_1p:
            if pre_smooth:
                dwrite = utils.spatial_smooth(dwrite, int(pre_smooth))
            dwrite = utils.spatial_high_pass(dwrite, int(spatial_hp))[np.newaxis, :]
        dwrite = np.clip(dwrite, rmin, rmax)

        # rigid registration
        ymax, xmax, cmax = rigid.phasecorr(
            data=rigid.apply_masks(data=dwrite, maskMul=maskMul, maskOffset=maskOffset),
            cfRefImg=cfRefImg.squeeze(),
            maxregshift=maxregshift,
            smooth_sigma_time=0,
        )
        for frame, dy, dx in zip(Img, ymax.flatten(), xmax.flatten()):
            frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)
        ###

        # non-rigid registration
        if is_nonrigid:

            if smooth_sigma_time > 0:
                dwrite = gaussian_filter1d(dwrite, sigma=smooth_sigma_time, axis=0)

            ymax1, xmax1, cmax1, = nonrigid.phasecorr(
                data=dwrite,
                maskMul=maskMulNR.squeeze(),
                maskOffset=maskOffsetNR.squeeze(),
                cfRefImg=cfRefImgNR.squeeze(),
                snr_thresh=snr_thresh,
                NRsm=NRsm,
                xblock=xblock,
                yblock=yblock,
                maxregshiftNR=maxregshiftNR,
            )

            X[i,1] = np.mean((ymax1**2 + xmax1**2)**.5)
            X[i,0] = np.mean((ymax[0]**2 + xmax[0]**2)**.5)
            X[i,2] = np.amax((ymax1**2 + xmax1**2)**.5)
    return X


def get_pc_metrics(mov, ops, use_red=False):
    """
    Computes registration metrics using top PCs of registered movie

    movie saved as binary file ops['reg_file']
    metrics saved to ops['regPC'] and ops['X']
    'regDX' is nPC x 3 where X[:,0] is rigid, X[:,1] is average nonrigid, X[:,2] is max nonrigid shifts
    'regPC' is average of top and bottom frames for each PC
    'tPC' is PC across time frames

    Parameters
    ----------
    ops : dict
        'nframes', 'Ly', 'Lx', 'reg_file' (if use_red=True, 'reg_file_chan2')
        (optional, 'refImg', 'block_size', 'maxregshiftNR', 'smooth_sigma', 'maxregshift', '1Preg')
    use_red : :obj:`bool`, optional
        default False, whether to use 'reg_file' or 'reg_file_chan2'

    Returns
    -------
    ops : dict
        The same as the ops input, but will now include 'regPC', 'tPC', and 'regDX'.

    """
    random_state = ops['reg_metrics_rs'] if 'reg_metrics_rs' in ops else None
    nPC = ops['reg_metric_n_pc'] if 'reg_metric_n_pc' in ops else 30
    pclow, pchigh, sv, ops['tPC'] = pclowhigh(mov, nlowhigh=np.minimum(300, int(ops['nframes'] / 2)),
                                              nPC=nPC, random_state=random_state
                                    )
    ops['regPC'] = np.concatenate((pclow[np.newaxis, :, :, :], pchigh[np.newaxis, :, :, :]), axis=0)

    ops['regDX'] = pc_register(
        pclow,
        pchigh,
        spatial_hp=ops['spatial_hp_reg'],
        pre_smooth=ops['pre_smooth'],
        bidi_corrected=ops['bidi_corrected'],
        smooth_sigma=ops['smooth_sigma'] if 'smooth_sigma' in ops else 1.15,
        smooth_sigma_time=ops['smooth_sigma_time'],
        block_size=ops['block_size'] if 'block_size' in ops else [128, 128],
        maxregshift=ops['maxregshift'] if 'maxregshift' in ops else 0.1,
        maxregshiftNR=ops['maxregshiftNR'] if 'maxregshiftNR' in ops else 5,
        reg_1p=ops['1Preg'] if '1Preg' in ops else False,
        snr_thresh=ops['snr_thresh'],
        is_nonrigid=ops['nonrigid'],
        bidiphase_offset=ops['bidiphase'],
        spatial_taper=ops['spatial_taper']
    )
    return ops


def filt_worker(inputs):
    X, filt = inputs
    for n in range(X.shape[0]):
        X[n,:,:] = convolve2d(X[n,:,:], filt, 'same')
    return X

def filt_parallel(data, filt, num_cores):
    nimg = data.shape[0]
    nbatch = int(np.ceil(nimg/float(num_cores)))
    inputs = np.arange(0, nimg, nbatch)
    irange = []
    dsplit = []
    for i in inputs:
        ilist = i + np.arange(0,np.minimum(nbatch, nimg-i),1,int)
        irange.append(ilist)
        dsplit.append([data[ilist,:, :], filt])
    if num_cores > 1:
        with Pool(num_cores) as p:
            results = p.map(filt_worker, dsplit)
        results = np.concatenate(results, axis=0 )
    else:
        results = filt_worker(dsplit[0])
    return results

def local_corr(mov, batch_size, num_cores):
    """ computes correlation image on mov (nframes x pixels x pixels) """
    nframes, Ly, Lx = mov.shape

    filt = np.ones((3,3),np.float32)
    filt[1,1] = 0
    filt /= norm(filt)
    ix=0
    k=0
    filtnorm = convolve2d(np.ones((Ly,Lx)),filt,'same')

    img_corr = np.zeros((Ly,Lx), np.float32)
    while ix < nframes:
        ifr = np.arange(ix, min(ix+batch_size, nframes), 1, int)

        X = mov[ifr,:,:]
        X = X.astype(np.float32)
        X -= X.mean(axis=0)
        Xstd = X.std(axis=0)
        Xstd[Xstd==0] = np.inf
        #X /= np.maximum(1, X.std(axis=0))
        X /= Xstd
        #for n in range(X.shape[0]):
        #    X[n,:,:] *= convolve2d(X[n,:,:], filt, 'same')
        X *= filt_parallel(X, filt, num_cores)
        img_corr += X.mean(axis=0)
        ix += batch_size
        k+=1
    img_corr /= filtnorm
    img_corr /= float(k)
    return img_corr

def bin_median(mov, window=10):
    nframes,Ly,Lx = mov.shape
    if nframes < window:
        window = nframes
    mov = np.nanmedian(np.reshape(mov[:int(np.floor(nframes/window)*window),:,:],
                                  (-1,window,Ly,Lx)).mean(axis=1), axis=0)
    return mov

def corr_to_template(mov, tmpl):
    nframes, Ly, Lx = mov.shape
    tmpl_flat = tmpl.flatten()
    tmpl_flat -= tmpl_flat.mean()
    tmpl_std = tmpl_flat.std()

    mov_flat = np.reshape(mov,(nframes,-1)).astype(np.float32)
    mov_flat -= mov_flat.mean(axis=1)[:,np.newaxis]
    mov_std = (mov_flat**2).mean(axis=1) ** 0.5

    correlations = (mov_flat * tmpl_flat).mean(axis=1) / (tmpl_std * mov_std)

    return correlations

def optic_flow(mov, tmpl, nflows):
    """ optic flow computation using farneback """
    window = int(1 / 0.2) # window size
    nframes, Ly, Lx = mov.shape
    mov = mov.astype(np.float32)
    mov = np.reshape(mov[:int(np.floor(nframes/window)*window),:,:],
                                  (-1,window,Ly,Lx)).mean(axis=1)

    mov = mov[np.random.permutation(mov.shape[0])[:min(nflows,mov.shape[0])], :, :]

    pyr_scale=.5
    levels=3
    winsize=100
    iterations=15
    poly_n=5
    poly_sigma=1.2 / 5
    flags=0

    nframes, Ly, Lx = mov.shape
    norms = np.zeros((nframes,))
    flows = np.zeros((nframes,Ly,Lx,2))

    for n in range(nframes):
        flow = cv2.calcOpticalFlowFarneback(
            tmpl, mov[n,:,:], None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

        flows[n,:,:,:] = flow
        norms[n] = norm(flow)

    return flows, norms


def get_flow_metrics(ops):
    """ get farneback optical flow and some other stats from normcorre paper """
    # done in batches for memory reasons
    Ly = ops['Ly']
    Lx = ops['Lx']
    reg_file = open(ops['reg_file'], 'rb')
    nbatch = ops['batch_size']
    nbytesread = 2 * Ly * Lx * nbatch

    Lyc = ops['yrange'][1] - ops['yrange'][0]
    Lxc = ops['xrange'][1] - ops['xrange'][0]
    img_corr = np.zeros((Lyc,Lxc), np.float32)
    img_median = np.zeros((Lyc,Lxc), np.float32)
    correlations = np.zeros((0,), np.float32)
    flows = np.zeros((0,Lyc,Lxc,2), np.float32)
    norms = np.zeros((0,), np.float32)
    smoothness = 0
    smoothness_corr = 0

    nflows = np.minimum(ops['nframes'], int(np.floor(100 / (ops['nframes']/nbatch))))
    ncorrs = np.minimum(ops['nframes'], int(np.floor(1000 / (ops['nframes']/nbatch))))

    k=0
    while True:
        buff = reg_file.read(nbytesread)
        mov = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        if mov.size==0:
            break
        mov = np.reshape(mov, (-1, Ly, Lx))

        mov = mov[np.ix_(np.arange(0, mov.shape[0],1,int),
                  np.arange(ops['yrange'][0],ops['yrange'][1],1,int),
                  np.arange(ops['xrange'][0],ops['xrange'][1],1,int))]

        img_corr += local_corr(mov[:,:,:], 1000, ops['num_workers'])
        img_median += bin_median(mov)
        k+=1

        smoothness += np.sqrt(
               np.sum(np.sum(np.array(np.gradient(np.mean(mov, 0)))**2, 0)))
        smoothness_corr += np.sqrt(
            np.sum(np.sum(np.array(np.gradient(img_corr))**2, 0)))

        tmpl = img_median / k

        correlations0 = corr_to_template(mov, tmpl)
        correlations = np.hstack((correlations, correlations0))
        if HAS_CV2:
            flows0, norms0 = optic_flow(mov, tmpl, nflows)
        else:
            flows0=[]
            norms0=[]
            print('flows not computed, cv2 not installed / did not import correctly')

        flows = np.vstack((flows,flows0))
        norms = np.hstack((norms,norms0))


    img_corr /= float(k)
    img_median /= float(k)

    smoothness /= float(k)
    smoothness_corr /= float(k)

    return tmpl, correlations, flows, norms, smoothness, smoothness_corr, img_corr
