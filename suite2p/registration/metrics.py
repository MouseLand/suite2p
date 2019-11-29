import numpy as np
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from ..utils import get_frames
from . import register, nonrigid

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import time

def pclowhigh(mov, nlowhigh, nPC):
    """ get mean of top and bottom PC weights for nPC's of mov

        computes nPC PCs of mov and returns average of top and bottom

        Parameters
        ----------
        mov : int16, array
            subsampled frames from movie size frames x Ly x Lx
        nlowhigh : int
            number of frames to average at top and bottom of each PC
        nPC : int
            number of PCs to compute

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
    tic=time.time()
    mov = mov.reshape((nframes, -1))
    mov = mov.astype(np.float32)
    mimg = mov.mean(axis=0)
    mov -= mimg
    tic=time.time()
    pca = PCA(n_components=nPC).fit(mov.T)
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

def pc_register(pclow, pchigh, refImg, smooth_sigma=1.15, block_size=(128,128), maxregshift=0.1, maxregshiftNR=10, preg=False):
    """ register top and bottom of PCs to each other

        Parameters
        ----------
        pclow : float, array
            average of bottom of spatial PC: nPC x Ly x Lx
        pchigh : float, array
            average of top of spatial PC: nPC x Ly x Lx
        refImg : int16, array
            reference image from registration
        smooth_sigma : :obj:`int`, optional
            default 1.15, see registration settings
        block_size : :obj:`tuple`, optional
            default (128,128), see registration settings
        maxregshift : :obj:`float`, optional
            default 0.1, see registration settings
        maxregshiftNR : :obj:`int`, optional
            default 10, see registration settings
        1Preg : :obj:`bool`, optional
            default True, see 1Preg settings

        Returns
        -------
            X : float, array
                nPC x 3 where X[:,0] is rigid, X[:,1] is average nonrigid, X[:,2] is max nonrigid shifts
    """
    # registration settings
    ops = {
        'num_workers': -1,
        'snr_thresh': 1.25,
        'nonrigid': True,
        'num_workers': -1,
        'block_size': np.array(block_size),
        'maxregshiftNR': np.array(maxregshiftNR),
        'maxregshift': np.array(maxregshift),
        'subpixel': 10,
        'smooth_sigma': smooth_sigma,
        'smooth_sigma_time': 0,
        '1Preg': preg,
        'pad_fft': False,
        'bidiphase': 0,
        'refImg': refImg,
        'spatial_taper': 50.0,
        'spatial_smooth': 2.0
        }
    nPC, ops['Ly'], ops['Lx'] = pclow.shape
    ops = nonrigid.make_blocks(ops)
    X = np.zeros((nPC,3))
    for i in range(nPC):
        refImg = pclow[i]
        Img = pchigh[i][np.newaxis, :, :]
        refAndMasks = register.prepare_refAndMasks(refImg, ops)
        dwrite, ymax, xmax, cmax, yxnr = register.compute_motion_and_shift(Img, refAndMasks, ops)
        X[i,1] = np.mean((yxnr[0]**2 + yxnr[1]**2)**.5)
        X[i,0] = np.mean((ymax[0]**2 + xmax[0]**2)**.5)
        X[i,2] = np.amax((yxnr[0]**2 + yxnr[1]**2)**.5)
    return X

def get_pc_metrics(ops, use_red=False):
    """ computes registration metrics using top PCs of registered movie

        movie saved as binary file ops['reg_file']
        metrics saved to ops['regPC'] and ops['X']
        'regDX' is nPC x 3 where X[:,0] is rigid, X[:,1] is average nonrigid, X[:,2] is max nonrigid shifts
        'regPC' is average of top and bottom frames for each PC
        'tPC' is PC across time frames

        Parameters
        ----------
        ops : dict
            requires 'nframes', 'Ly', 'Lx', 'reg_file' (if use_red=True, 'reg_file_chan2')
        use_red : :obj:`bool`, optional
            default False, whether to use 'reg_file' or 'reg_file_chan2'

        Returns
        -------
            ops : dict
                adds 'regPC' and 'tPC' and 'regDX'

    """
    nsamp    = min(5000, ops['nframes']) # n frames to pick from full movie
    if ops['nframes'] < 5000:
        nsamp = min(2000, ops['nframes'])
    if ops['Ly'] > 700 or ops['Lx'] > 700:
        nsamp = min(2000, nsamp)
    nPC      = 30 # n PCs to compute motion for
    nlowhigh = np.minimum(300,int(ops['nframes']/2)) # n frames to average at ends of PC coefficient sortings
    ix   = np.linspace(0,ops['nframes']-1,nsamp).astype('int')
    if use_red and 'reg_file_chan2' in ops:
        mov  = get_frames(ops, ix, ops['reg_file_chan2'], crop=True, badframes=True)
    else:
        mov  = get_frames(ops, ix, ops['reg_file'], crop=True, badframes=True)

    pclow, pchigh, sv, v = pclowhigh(mov, nlowhigh, nPC)
    if 'block_size' not in ops:
        ops['block_size']   = [128, 128]
    if 'maxregshiftNR' not in ops:
        ops['maxregshiftNR'] = 5
    if 'smooth_sigma' not in ops:
        ops['smooth_sigma'] = 1.15
    if 'maxregshift' not in ops:
        ops['maxregshift'] = 0.1
    if '1Preg' not in ops:
        ops['1Preg'] = False
    if 'refImg' in ops:
        refImg = ops['refImg']

    else:
        refImg = mov.mean(axis=0)
    X    = pc_register(pclow, pchigh, refImg,
                       ops['smooth_sigma'], ops['block_size'], ops['maxregshift'], ops['maxregshiftNR'], ops['1Preg'])
    ops['regPC'] = np.concatenate((pclow[np.newaxis, :,:,:], pchigh[np.newaxis, :,:,:]), axis=0)
    ops['regDX'] = X
    ops['tPC'] = v

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
    ''' computes correlation image on mov (nframes x pixels x pixels)'''
    nframes, Ly, Lx = mov.shape

    filt = np.ones((3,3),np.float32)
    filt[1,1] = 0
    fnorm = ((filt**2).sum())**0.5
    filt /= fnorm
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
        norms[n] = ((flow**2).sum()) ** 0.5

    return flows, norms


def get_flow_metrics(ops):
    ''' get farneback optical flow and some other stats from normcorre paper'''
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
