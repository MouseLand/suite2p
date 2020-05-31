import numpy as np
from sklearn.decomposition import PCA

from . import nonrigid, register, utils


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
    mov = mov.reshape((nframes, -1))
    mov = mov.astype(np.float32)
    mimg = mov.mean(axis=0)
    mov -= mimg
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
        dwrite, ymax, xmax, cmax, yxnr = register.compute_motion_and_shift(
            data=Img,
            refAndMasks=refAndMasks,
            nblocks=ops['nblocks'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            nr_sm=ops['NRsm'],
            snr_thresh=ops['snr_thresh'],
            smooth_sigma_time=ops['smooth_sigma_time'],
            maxregshiftNR=ops['maxregshiftNR'],
            ops=ops,
        )
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
        ops : dictionary
            'nframes', 'Ly', 'Lx', 'reg_file' (if use_red=True, 'reg_file_chan2')
            (optional, 'refImg', 'block_size', 'maxregshiftNR', 'smooth_sigma', 'maxregshift', '1Preg')
        use_red : :obj:`bool`, optional
            default False, whether to use 'reg_file' or 'reg_file_chan2'

        Returns
        -------
            ops : dictionary
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
        mov  = utils.get_frames(ops, ix, ops['reg_file_chan2'], crop=True, badframes=True)
    else:
        mov  = utils.get_frames(ops, ix, ops['reg_file'], crop=True, badframes=True)

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