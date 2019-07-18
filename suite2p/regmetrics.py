from numpy import fft
import numpy as np
import multiprocessing
from multiprocessing import Pool
import sys
from suite2p import register, utils, nonrigid
from scipy.signal import convolve2d
from scipy.sparse import linalg
from sklearn.decomposition import PCA
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import time

def pclowhigh(mov, nlowhigh, nPC):
    ''' get mean of top and bottom PC weights for nPC's of mov '''
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

def cov_worker(mov):
    return mov @ mov.T

def pc_register(pclow, pchigh, refImg, smooth_sigma=1.15, block_size=(128,128), maxregshift=0.1, maxregshiftNR=5, preg=True):
    ''' register top and bottom of PCs to each other '''
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
        '1Preg': False,
        'pad_fft': False,
        'bidiphase': 0,
        'refImg': refImg
        }
    nPC, ops['Ly'], ops['Lx'] = pclow.shape
    ops = utils.make_blocks(ops)
    X = np.zeros((nPC,3))
    for i in range(nPC):
        refImg = pclow[i]
        Img = pchigh[i][np.newaxis, :, :]
        X[i,:] = pc_register_worker([ops, refImg, Img])

    return X

def pc_register_worker(inputs):
    ops, refImg, Img = inputs
    maskMul, maskOffset, cfRefImg = register.prepare_masks(refImg, ops)
    maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.prepare_masks(refImg, ops)
    refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
    dwrite, ymax, xmax, cmax, yxnr = register.register_and_shift(Img, refAndMasks, ops)
    X = np.zeros((3,))
    X[1] = np.mean((yxnr[0]**2 + yxnr[1]**2)**.5)
    X[0] = np.mean((ymax[0]**2 + xmax[0]**2)**.5)
    X[2] = np.amax((yxnr[0]**2 + yxnr[1]**2)**.5)
    return X

def get_pc_metrics(ops, use_red=False):
    ''' computes registration metrics using top PCs of registered movie
        movie saved as binary file ops['reg_file']
        metrics saved to ops['regPC'] and ops['X']
    '''
    nsamp    = min(5000, ops['nframes']) # n frames to pick from full movie
    if ops['Ly'] > 700 or ops['Lx'] > 700:
        nsamp = min(2500, nsamp)
    nPC      = 30 # n PCs to compute motion for
    nlowhigh = np.minimum(300,int(ops['nframes']/2)) # n frames to average at ends of PC coefficient sortings
    ix   = np.linspace(0,ops['nframes']-1,nsamp).astype('int')
    if use_red and 'reg_file_chan2' in ops:
        mov  = utils.sample_frames(ops, ix, ops['reg_file_chan2'])
    else:
        mov  = utils.sample_frames(ops, ix, ops['reg_file'])

    pclow, pchigh, sv, v = pclowhigh(mov, nlowhigh, nPC)
    if 'block_size' not in ops:
        ops['block_size']   = [128, 128]
    if 'maxregshiftNR' not in ops:
        ops['maxregshiftNR'] = 5
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
