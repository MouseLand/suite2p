import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

from . import rigid, nonrigid, utils
from .. import io


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


def pc_register(pclow, pchigh, bidi_corrected, spatial_hp=None, pre_smooth=None, smooth_sigma=1.15, smooth_sigma_time=0,
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
    nPC, Ly, Lx = pclow.shape
    yblock, xblock, nblocks, maxregshiftNR, block_size, NRsm = nonrigid.make_blocks(
        Ly=Ly, Lx=Lx, maxregshiftNR=np.array(maxregshiftNR), block_size=np.array(block_size)
    )

    X = np.zeros((nPC,3))
    for i in range(nPC):
        refImg = pclow[i]
        Img = pchigh[i][np.newaxis, :, :]

        if reg_1p:
            data = refImg
            if pre_smooth and pre_smooth % 2:
                raise ValueError("if set, pre_smooth must be a positive even integer.")
            if spatial_hp % 2:
                raise ValueError("spatial_hp must be a positive even integer.")
            data = data.astype(np.float32)
            data = data[np.newaxis, :, :]
            if pre_smooth:
                data = utils.spatial_smooth(data, int(pre_smooth))
            data = utils.spatial_high_pass(data, int(spatial_hp))
            refImg = data.squeeze()

        maskMul, maskOffset = rigid.compute_masks(
            refImg=refImg,
            maskSlope=spatial_taper if reg_1p else 3 * smooth_sigma
        )
        cfRefImg = rigid.phasecorr_reference(
            refImg=refImg,
            smooth_sigma=smooth_sigma,
            pad_fft=pad_fft,
        )
        cfRefImg = cfRefImg[np.newaxis, :, :]
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

        ###
        dwrite = Img
        if smooth_sigma_time > 0:
            dwrite = gaussian_filter1d(dwrite, sigma=smooth_sigma_time, axis=0)
            dwrite = dwrite.astype(np.float32)

        # preprocessing for 1P recordings
        if reg_1p:
            if pre_smooth and pre_smooth % 2:
                raise ValueError("if set, pre_smooth must be a positive even integer.")
            if spatial_hp % 2:
                raise ValueError("spatial_hp must be a positive even integer.")
            Img = Img.astype(np.float32)

            if pre_smooth:
                dwrite = utils.spatial_smooth(dwrite, int(pre_smooth))
            dwrite = utils.spatial_high_pass(dwrite, int(spatial_hp))

        # rigid registration
        ymax, xmax, cmax = rigid.phasecorr(
            data=rigid.apply_masks(data=dwrite, maskMul=maskMul, maskOffset=maskOffset),
            cfRefImg=cfRefImg.squeeze(),
            maxregshift=maxregshift,
            smooth_sigma_time=smooth_sigma_time,
        )
        for frame, dy, dx in zip(Img, ymax.flatten(), xmax.flatten()):
            frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)
        ###

        # non-rigid registration
        if is_nonrigid:

            if smooth_sigma_time > 0:
                dwrite = gaussian_filter1d(dwrite, sigma=smooth_sigma_time, axis=0)

            ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(
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


def get_pc_metrics(ops, use_red=False, nPC=30):
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
        nPC : int
            # n PCs to compute motion for

        Returns
        -------
            ops : dictionary
                adds 'regPC' and 'tPC' and 'regDX'

    """
    # n frames to pick from full movie
    nsamp = min(2000 if ops['nframes'] < 5000 or ops['Ly'] > 700 or ops['Lx'] > 700 else 5000, ops['nframes'])

    mov = io.get_frames(
        Lx=ops['Lx'],
        Ly=ops['Ly'],
        xrange=ops['xrange'],
        yrange=ops['yrange'],
        ix=np.linspace(0, ops['nframes'] - 1, nsamp).astype('int'),
        bin_file=ops['reg_file_chan2'] if use_red and 'reg_file_chan2' in ops else ops['reg_file'],
        crop=True,
    )

    pclow, pchigh, sv, ops['tPC'] = pclowhigh(mov, nlowhigh=np.minimum(300, int(ops['nframes'] / 2)), nPC=nPC)
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
        pad_fft=ops['pad_fft'],
        bidiphase=ops['bidiphase'],
        spatial_taper=ops['spatial_taper']
    )
    return ops