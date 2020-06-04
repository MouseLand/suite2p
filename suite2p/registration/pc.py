import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


from . import rigid, nonrigid, register, utils


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


def pc_register(pclow, pchigh, spatial_hp, pre_smooth, bidi_corrected, smooth_sigma=1.15, smooth_sigma_time=0,
                block_size=(128,128), maxregshift=0.1, maxregshiftNR=10, reg_1p=False, snr_thresh=1.25,
                is_nonrigid=True, pad_fft=False, bidiphase=0, spatial_taper=50.0):
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
    block_size = np.array(block_size)
    maxregshiftNR = np.array(maxregshiftNR)
    maxregshift = np.array(maxregshift)

    nPC, Ly, Lx = pclow.shape
    yblock, xblock, nblocks, maxregshiftNR, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, maxregshiftNR=maxregshiftNR, block_size=block_size)

    X = np.zeros((nPC,3))
    for i in range(nPC):
        refImg = pclow[i]
        Img = pchigh[i][np.newaxis, :, :]

        maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(
            refImg0=refImg,
            spatial_taper=spatial_taper,
            smooth_sigma=smooth_sigma,
            pad_fft=pad_fft,
            reg_1p=reg_1p,
            spatial_hp=spatial_hp,
            pre_smooth=pre_smooth,
        )
        if is_nonrigid:
            maskSlope = spatial_taper if reg_1p else 3 * smooth_sigma  # slope of taper mask at the edges
            # pre filtering for one-photon data
            if reg_1p:
                data = refImg[np.newaxis, :, :]
                if pre_smooth and pre_smooth % 2:
                    raise ValueError("if set, pre_smooth must be a positive even integer.")
                if spatial_hp % 2:
                    raise ValueError("spatial_hp must be a positive even integer.")
                data = data.astype(np.float32)
                if pre_smooth:
                    data = utils.spatial_smooth(data, int(pre_smooth))
                data = utils.spatial_high_pass(data, int(spatial_hp))
                refImg = data


            maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
                refImg0=refImg,
                maskSlope=maskSlope,
                smooth_sigma=smooth_sigma,
                yblock=yblock,
                xblock=xblock,
                pad_fft=pad_fft,
            )

        if bidiphase and not bidi_corrected:
            bidiphase.shift(Img, bidiphase)

        dwrite, ymax, xmax, cmax = register.compute_motion_and_shift(
            data=Img,
            maskMul=maskMul,
            maskOffset=maskOffset,
            cfRefImg=cfRefImg,
            maxregshift=maxregshift,
            smooth_sigma_time=smooth_sigma_time,
            reg_1p=reg_1p,
            spatial_hp=spatial_hp,
            pre_smooth=pre_smooth,
        )

        # non-rigid registration
        if is_nonrigid:

            if smooth_sigma_time > 0:
                data_smooth = gaussian_filter1d(dwrite, sigma=smooth_sigma_time, axis=0)

            ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(
                data=data_smooth if smooth_sigma_time > 0 else dwrite,
                refAndMasks=[maskMulNR, maskOffsetNR, cfRefImgNR],
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


