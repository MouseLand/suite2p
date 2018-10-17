from skimage import io
import glob, h5py, time, os, shutil
import numpy as np
from numpy import fft
from numpy import random as rnd
import multiprocessing
from multiprocessing import Pool
import math
from suite2p import nonrigid, utils

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

eps0 = 1e-5;
sigL = 0.85 # smoothing width for up-sampling kernels, keep it between 0.5 and 1.0...
lpad = 3   # upsample from a square +/- lpad

# smoothing kernel
def kernelD(a, b):
    dxs = np.reshape(a[0], (-1,1)) - np.reshape(b[0], (1,-1))
    dys = np.reshape(a[1], (-1,1)) - np.reshape(b[1], (1,-1))
    ds = np.square(dxs) + np.square(dys)
    K = np.exp(-ds/(2*np.square(sigL)))
    return K

def mat_upsample(lpad, ops):
    lar    = np.arange(-lpad, lpad+1)
    larUP  = np.arange(-lpad, lpad+.001, 1./ops['subpixel'])
    x, y   = np.meshgrid(lar, lar)
    xU, yU = np.meshgrid(larUP, larUP)
    Kx = kernelD((x,y),(x,y))
    Kx = np.linalg.inv(Kx)
    Kg = kernelD((x,y),(xU,yU))
    Kmat = np.dot(Kx, Kg)
    nup = larUP.shape[0]
    return Kmat, nup

def prepare_masks(refImg, ops):
    maskSlope   = 3 * ops['smooth_sigma'] # slope of taper mask at the edges
    Ly,Lx = refImg.shape
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    mY = y.max() - 2*maskSlope
    mX = x.max() - 2*maskSlope
    maskY = 1./(1.+np.exp((yy-mY)/maskSlope))
    maskX = 1./(1.+np.exp((xx-mX)/maskSlope))
    maskMul = maskY * maskX
    maskOffset = refImg.mean() * (1. - maskMul);

    hgx = np.exp(-np.square(xx/ops['smooth_sigma']))
    hgy = np.exp(-np.square(yy/ops['smooth_sigma']))
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft.fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain

    cfRefImg   = np.conj(fft.fft2(refImg));
    if ops['do_phasecorr']:
        absRef     = np.absolute(cfRefImg);
        cfRefImg   = cfRefImg / (eps0 + absRef)
    cfRefImg *= fhg

    maskMul = maskMul.astype('float32')
    maskOffset = maskOffset.astype('float32')
    cfRefImg = cfRefImg.astype('complex64')
    return maskMul, maskOffset, cfRefImg

def correlation_map(data, refAndMasks, do_phasecorr):
    maskMul    = refAndMasks[0]
    maskOffset = refAndMasks[1]
    cfRefImg   = refAndMasks[2]
    nimg, Ly, Lx = data.shape
    cfRefImg = np.reshape(cfRefImg, (1, Ly, Lx))
    data = data.astype('float32') * maskMul + maskOffset
    X = fft.fft2(data)
    if do_phasecorr:
        J = X / (eps0 + np.absolute(X))
    else:
        J = X
    J = J * cfRefImg
    cc = np.real(fft.ifft2(J))
    cc = fft.fftshift(cc, axes=(1,2))
    return cc

def getSNR(cc, Ls, ops):
    (lcorr, lpad, Lyhalf, Lxhalf) = Ls
    nimg = cc.shape[0]
    cc0 = cc[:, (Lyhalf-lcorr):(Lyhalf+lcorr+1), (Lxhalf-lcorr):(Lxhalf+lcorr+1)]
    cc0 = np.reshape(cc0, (nimg, -1))
    X1max  = np.amax(cc0, axis = 1)
    ix  = np.argmax(cc0, axis = 1)
    ymax, xmax = np.unravel_index(ix, (2*lcorr+1,2*lcorr+1))
    # set to 0 all pts +-lpad from ymax,xmax
    cc0 = cc[:, (Lyhalf-lcorr-lpad):(Lyhalf+lcorr+1+lpad), (Lxhalf-lcorr-lpad):(Lxhalf+lcorr+1+lpad)]
    for j in range(nimg):
        cc0[j,ymax[j]:ymax[j]+2*lpad, xmax[j]:xmax[j]+2*lpad] = 0
    cc0 = np.reshape(cc0, (nimg, -1))
    Xmax  = np.maximum(0, np.amax(cc0, axis = 1))
    snr = X1max / Xmax # computes snr
    return snr

def getXYup(cc, Ls, ops):
    (lcorr, lpad, Lyhalf, Lxhalf) = Ls
    nimg = cc.shape[0]
    cc0 = cc[:, (Lyhalf-lcorr):(Lyhalf+lcorr+1), (Lxhalf-lcorr):(Lxhalf+lcorr+1)]
    cc0 = np.reshape(cc0, (nimg, -1))
    ix  = np.argmax(cc0, axis = 1)
    ymax, xmax = np.unravel_index(ix, (2*lcorr+1,2*lcorr+1))
    X1max  = np.amax(cc0, axis = 1)
    # set to 0 all pts +-lpad from ymax,xmax
    cc0 = cc[:, (Lyhalf-lcorr-lpad):(Lyhalf+lcorr+1+lpad), (Lxhalf-lcorr-lpad):(Lxhalf+lcorr+1+lpad)].copy()
    for j in range(nimg):
        cc0[j,ymax[j]:ymax[j]+2*lpad, xmax[j]:xmax[j]+2*lpad] = 0
    cc0 = np.reshape(cc0, (nimg, -1))
    Xmax  = np.maximum(0, np.amax(cc0, axis = 1))
    snr = X1max / Xmax # computes snr
    mxpt = [ymax+Lyhalf-lcorr, xmax + Lxhalf-lcorr]
    ccmat = np.zeros((nimg, 2*lpad+1, 2*lpad+1))
    for j in range(0, nimg):
        ccmat[j,:,:] = cc[j, (mxpt[0][j] -lpad):(mxpt[0][j] +lpad+1), (mxpt[1][j] -lpad):(mxpt[1][j] +lpad+1)]
    ccmat = np.reshape(ccmat, (nimg,-1))
    Kmat, nup = mat_upsample(lpad, ops)
    ccb = np.dot(ccmat, Kmat)
    imax = np.argmax(ccb, axis=1)
    cmax = np.amax(ccb, axis=1)
    ymax, xmax = np.unravel_index(imax, (nup,nup))
    mdpt = np.floor(nup/2)
    ymax,xmax = (ymax-mdpt)/ops['subpixel'], (xmax-mdpt)/ops['subpixel']
    ymax, xmax = ymax + mxpt[0] - Lyhalf, xmax + mxpt[1] - Lxhalf
    return ymax, xmax, cmax, snr

def getXYup2(cc, Ls, ops):
    (lcorr, lpad, Lyhalf, Lxhalf) = Ls
    nimg = cc.shape[0]
    cc0 = cc[:, (Lyhalf-lcorr):(Lyhalf+lcorr+1), (Lxhalf-lcorr):(Lxhalf+lcorr+1)]
    cc0 = np.reshape(cc0, (nimg, -1))
    ix  = np.argmax(cc0, axis = 1)
    ymax, xmax = np.unravel_index(ix, (2*lcorr+1,2*lcorr+1))

    mxpt = [ymax+Lyhalf-lcorr, xmax + Lxhalf-lcorr]
    ccmat = np.zeros((nimg, 2*lpad+1, 2*lpad+1))
    for j in range(0, nimg):
        ccmat[j,:,:] = cc[j, (mxpt[0][j] -lpad):(mxpt[0][j] +lpad+1), (mxpt[1][j] -lpad):(mxpt[1][j] +lpad+1)]
    ccmat = np.reshape(ccmat, (nimg,-1))
    Kmat, nup = mat_upsample(lpad, ops)
    ccb = np.dot(ccmat, Kmat)
    imax = np.argmax(ccb, axis=1)
    cmax = np.amax(ccb, axis=1)
    ymax, xmax = np.unravel_index(imax, (nup,nup))
    mdpt = np.floor(nup/2)
    ymax,xmax = (ymax-mdpt)/ops['subpixel'], (xmax-mdpt)/ops['subpixel']
    ymax, xmax = ymax + mxpt[0] - Lyhalf, xmax + mxpt[1] - Lxhalf
    return ymax, xmax, cmax

def shift_data(inputs):
    X, ymax, xmax = inputs
    ymax = ymax.flatten()
    xmax = xmax.flatten()
    if X.ndim<3:
        X = X[np.newaxis,:,:]
    X = fft.fft2(X.astype('float32'))
    nimg, Ly, Lx = X.shape
    Ny = fft.ifftshift(np.arange(-np.fix(Ly/2), np.ceil(Ly/2)))
    Nx = fft.ifftshift(np.arange(-np.fix(Lx/2), np.ceil(Lx/2)))
    [Nx,Ny] = np.meshgrid(Nx,Ny)
    Nx = Nx.astype('float32') / Lx
    Ny = Ny.astype('float32') / Ly
    dph = Nx * np.reshape(xmax, (-1,1,1)) + Ny * np.reshape(ymax, (-1,1,1))
    Y = np.real(fft.ifft2(X * np.exp((2j * np.pi) * dph)))
    return Y

def phasecorr_worker(inputs):
    data, refAndMasks, ops = inputs
    if ops['nonrigid'] and len(refAndMasks)>3:
        refAndMasksNR = refAndMasks[3:]
        refAndMasks = refAndMasks[:3]
        nr = True
    else:
        nr = False
    nimg, Ly, Lx = data.shape
    Lyhalf = int(np.floor(Ly/2))
    Lxhalf = int(np.floor(Lx/2))
    maxregshift = np.round(ops['maxregshift'] *np.maximum(Ly, Lx))
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(Ly,Lx)/2.)-lpad))
    cc = correlation_map(data, refAndMasks, ops['do_phasecorr'])
    ymax, xmax, cmax,snr = getXYup(cc, (lcorr,lpad, Lyhalf, Lxhalf), ops)
    Y = shift_data((data, ymax, xmax))
    yxnr = []
    if nr:
        Y, ymax1, xmax1, cmax1, snr1 = nonrigid.phasecorr_worker((Y, refAndMasksNR, ops))
        yxnr = [ymax1,xmax1,cmax1, snr1]
    return Y, ymax, xmax, cmax, yxnr

def phasecorr(data, refAndMasks, ops):
    nr=False
    yxnr = []
    if ops['nonrigid'] and len(refAndMasks)>3:
        nb = ops['nblocks'][0] * ops['nblocks'][1]
        nr=True
    if ops['num_workers']<0:
        Y, ymax, xmax, cmax, yxnr = phasecorr_worker((data, refAndMasks, ops))
    else:
        nimg = data.shape[0]
        if ops['num_workers']<1:
            ops['num_workers'] = int(multiprocessing.cpu_count()/2)
        num_cores = ops['num_workers']

        nbatch = int(np.ceil(nimg/float(num_cores)))
        #nbatch = 50
        inputs = np.arange(0, nimg, nbatch)
        irange = []
        dsplit = []
        for i in inputs:
            ilist = i + np.arange(0,np.minimum(nbatch, nimg-i))
            irange.append(ilist)
            dsplit.append([data[ilist,:, :], refAndMasks, ops])
        with Pool(num_cores) as p:
            results = p.map(phasecorr_worker, dsplit)
        Y = np.zeros_like(data)
        ymax = np.zeros((nimg,))
        xmax = np.zeros((nimg,))
        cmax = np.zeros((nimg,))
        if nr:
            ymax1 = np.zeros((nimg,nb))
            xmax1 = np.zeros((nimg,nb))
            cmax1 = np.zeros((nimg,nb))
        for i in range(0,len(results)):
            Y[irange[i], :, :] = results[i][0]
            ymax[irange[i]] = results[i][1]
            xmax[irange[i]] = results[i][2]
            cmax[irange[i]] = results[i][3]
            if nr:
                ymax1[irange[i],:] = results[i][4][0]
                xmax1[irange[i],:] = results[i][4][1]
                cmax1[irange[i],:] = results[i][4][2]
        if nr:
            yxnr = [ymax1,xmax1,cmax1]
    #if nr:
        #print(yxnr[0].max(),yxnr[1].max())
    return Y, ymax, xmax, cmax, yxnr

def get_nFrames(ops):
    nbytes = os.path.getsize(ops['reg_file'])
    nFrames = int(nbytes/(2* ops['Ly'] *  ops['Lx']))
    return nFrames

def register_myshifts(ops, data, ymax, xmax):
    if ops['num_workers']<0:
        dreg = shift_data((data, ymax, xmax))
    else:
        if ops['num_workers']<1:
            ops['num_workers'] = int(multiprocessing.cpu_count()/2)
        num_cores = ops['num_workers']
        nimg = data.shape[0]
        nbatch = int(np.ceil(nimg/float(num_cores)))
        #nbatch = 50
        inputs = np.arange(0, nimg, nbatch)
        irange = []
        dsplit = []
        for i in inputs:
            ilist = i + np.arange(0,np.minimum(nbatch, nimg-i))
            irange.append(i + np.arange(0,np.minimum(nbatch, nimg-i)))
            dsplit.append([data[ilist,:, :], ymax[ilist], xmax[ilist]])
        with Pool(num_cores) as p:
            results = p.map(shift_data, dsplit)

        dreg = np.zeros_like(data)
        for i in range(0,len(results)):
            dreg[irange[i], :, :] = results[i]
    return dreg

def subsample_frames(ops, nsamps):
    nFrames = get_nFrames(ops)
    Ly = ops['Ly']
    Lx = ops['Lx']
    frames = np.zeros((nsamps, Ly, Lx), dtype='int16')
    nbytesread = 2 * Ly * Lx
    istart = np.linspace(0, nFrames, 1+nsamps).astype('int64')
    if ops['nchannels']>1:
        if ops['functional_chan'] == ops['align_by_chan']:
            reg_file = open(ops['reg_file'], 'rb')
        else:
            reg_file = open(ops['reg_file_chan2'], 'rb')
    else:
        reg_file = open(ops['reg_file'], 'rb')
    for j in range(0,nsamps):
        reg_file.seek(nbytesread * istart[j], 0)
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        frames[j,:,:] = np.reshape(data, (Ly, Lx))
    reg_file.close()
    return frames

def pick_init_init(ops, frames):
    nimg = frames.shape[0]
    frames = np.reshape(frames, (nimg,-1)).astype('float32')
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
    cc = frames @ np.transpose(frames)
    ndiag = np.sqrt(np.diag(cc))
    cc = cc / np.outer(ndiag, ndiag)
    CCsort = -np.sort(-cc, axis = 1)
    bestCC = np.mean(CCsort[:, 1:20], axis=1);
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    refImg = np.mean(frames[indsort[0:20], :], axis = 0)
    refImg = np.reshape(refImg, (ops['Ly'], ops['Lx']))
    return refImg

def refine_init_init(ops, frames, refImg):
    niter = 8
    nmax  = np.minimum(100, int(frames.shape[0]/2))
    for iter in range(0,niter):
        maskMul, maskOffset, cfRefImg = prepare_masks(refImg, ops)
        freg, ymax, xmax, cmax, yxnr = phasecorr(frames, [maskMul, maskOffset, cfRefImg], ops)
        isort = np.argsort(-cmax)
        nmax = int(frames.shape[0] * (1.+iter)/(2*niter))
        refImg = freg[isort[1:nmax], :, :].mean(axis=0).squeeze()
        dy, dx = -np.mean(ymax[isort[1:nmax]]), -np.mean(xmax[isort[1:nmax]])
        refImg = shift_data((refImg, dy, dx)).squeeze()
        ymax, xmax = ymax+dy, xmax+dx
    return refImg

def pick_init(ops):
    nbytes = os.path.getsize(ops['reg_file'])
    Ly = ops['Ly']
    Lx = ops['Lx']
    nFrames = int(nbytes/(2*Ly*Lx))
    nFramesInit = np.minimum(ops['nimg_init'], nFrames)
    frames = subsample_frames(ops, nFramesInit)
    refImg = pick_init_init(ops, frames)
    refImg = refine_init_init(ops, frames, refImg)
    return refImg

def get_metrics(ops):
    nsamp    = min(10000, ops['nframes']) # n frames to pick from full movie
    nPC      = 50 # n PCs to compute motion for
    nlowhigh = 500 # n frames to average at ends of PC coefficient sortings
    ix   = np.linspace(0,ops['nframes']-1,nsamp).astype('int')
    mov  = utils.sample_frames(ops, ix)
    #mov = mov[:, ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]]

    pclow, pchigh,sv = utils.pclowhigh(mov, nlowhigh, nPC)
    if 'block_size' not in ops:
        ops['block_size']   = [128, 128]
    if 'maxregshiftNR' not in ops:
        ops['maxregshiftNR'] = 5
    X    = utils.metric_register(pclow, pchigh, ops['do_phasecorr'], ops['smooth_sigma'], ops['block_size'], ops['maxregshift'], ops['maxregshiftNR'])
    ops['regPC'] = np.concatenate((pclow[np.newaxis, :,:,:], pchigh[np.newaxis, :,:,:]), axis=0)
    ops['regDX'] = X
    return ops

def register_binary(ops):
    # if ops is a list of dictionaries, each will be registered separately
    if (type(ops) is list) or (type(ops) is np.ndarray):
        for op in ops:
            op = register_binary(op)
        return ops
    if ops['nonrigid']:
        ops = utils.make_blocks(ops)
    if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
        ops['reg_file_raw'] = os.path.splitext(ops['reg_file'])[0]+'_raw.bin'
        print(ops['reg_file_raw'])
        shutil.copyfile(ops['reg_file'], ops['reg_file_raw'])

    Ly = ops['Ly']
    Lx = ops['Lx']
    ops['nframes'] = get_nFrames(ops)
    if ops['nframes']<50:
        raise Exception('the total number of frames should be at least 50 ')
    if ops['nframes']<200:
        print('number of frames is below 200, unpredictable behaviors may occur')
    refImg = pick_init(ops)
    ops['refImg'] = refImg
    print('computed reference frame for registration')
    nbatch = ops['batch_size']
    nbytesread = 2 * Ly * Lx * nbatch
    if ops['nchannels']>1:
        if ops['functional_chan'] == ops['align_by_chan']:
            reg_file_align = open(ops['reg_file'], 'r+b')
            reg_file_alt = open(ops['reg_file_chan2'], 'r+b')
        else:
            reg_file_align = open(ops['reg_file_chan2'], 'r+b')
            reg_file_alt = open(ops['reg_file'], 'r+b')
    else:
        reg_file_align = open(ops['reg_file'], 'r+b')
    meanImg = np.zeros((Ly, Lx))
    k = 0
    nfr = 0
    k0 = tic()
    maskMul, maskOffset, cfRefImg = prepare_masks(refImg, ops)
    yoff = np.zeros((0,),np.float32)
    xoff = np.zeros((0,),np.float32)
    corrXY = np.zeros((0,),np.float32)
    if ops['nonrigid']:
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.prepare_masks(refImg, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
        nb = ops['nblocks'][0] * ops['nblocks'][1]
        yoff1 = np.zeros((0,nb),np.float32)
        xoff1 = np.zeros((0,nb),np.float32)
        corrXY1 = np.zeros((0,nb),np.float32)
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]
    while True:
        buff = reg_file_align.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        if data.size==0:
            break
        data = np.reshape(data, (-1, Ly, Lx))
        dwrite, ymax, xmax, cmax, yxnr = phasecorr(data, refAndMasks, ops)
        dwrite = dwrite.astype('int16')
        reg_file_align.seek(-2*dwrite.size,1)
        reg_file_align.write(bytearray(dwrite))
        meanImg += dwrite.sum(axis=0)
        yoff = np.hstack((yoff, ymax))
        xoff = np.hstack((xoff, xmax))
        corrXY = np.hstack((corrXY, cmax))
        if ops['nonrigid']:
            yoff1 = np.vstack((yoff1, yxnr[0]))
            xoff1 = np.vstack((xoff1, yxnr[1]))
            corrXY1 = np.vstack((corrXY1, yxnr[2]))
        if ops['reg_tif']:
            if k==0:
                if ops['functional_chan']==ops['align_by_chan']:
                    tifroot = os.path.join(ops['save_path'], 'reg_tif')
                else:
                    tifroot = os.path.join(ops['save_path'], 'reg_tif_chan2')
                if not os.path.isdir(tifroot):
                    os.makedirs(tifroot)
                print(tifroot)
            fname = 'file_chan%0.3d.tif'%k
            io.imsave(os.path.join(tifroot, fname), dwrite)
        nfr += dwrite.shape[0]
        k += 1
        if k%5==0:
            print('registered %d/%d frames in time %4.2f'%(nfr, ops['nframes'], toc(k0)))
    reg_file_align.close()
    ops['yoff'] = yoff
    ops['xoff'] = xoff
    ymin = np.maximum(0, np.ceil(np.amax(yoff)))
    ymax = Ly + np.minimum(0, np.floor(np.amin(yoff)))
    ops['yrange'] = [int(ymin), int(ymax)]
    xmin = np.maximum(0, np.ceil(np.amax(xoff)))
    xmax = Lx + np.minimum(0, np.floor(np.amin(xoff)))
    ops['xrange'] = [int(xmin), int(xmax)]
    ops['corrXY'] = corrXY
    if ops['nonrigid']:
        ops['yoff1'] = yoff1
        ops['xoff1'] = xoff1
        ops['corrXY1'] = corrXY1
    if ops['nchannels']==1 or ops['functional_chan']==ops['align_by_chan']:
        ops['meanImg'] = meanImg/ops['nframes']
    else:
        ops['meanImg_chan2'] = meanImg/ops['nframes']
    k=0
    if ops['nchannels']>1:
        ix = 0
        meanImg = np.zeros((Ly, Lx))
        while True:
            buff = reg_file_alt.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            buff = []
            if data.size==0:
                break
            data = np.reshape(data, (-1, Ly, Lx))
            nframes = data.shape[0]
            # register by pre-determined amount
            dwrite = register_myshifts(ops, data, yoff[ix + np.arange(0,nframes)], xoff[ix + np.arange(0,nframes)])
            if ops['nonrigid']==True:
                dwrite = nonrigid.register_myshifts(ops, dwrite, yoff1[ix + np.arange(0,nframes),:], xoff1[ix + np.arange(0,nframes),:])
            ix += nframes
            dwrite = dwrite.astype('int16')
            reg_file_alt.seek(-2*dwrite.size,1)
            reg_file_alt.write(bytearray(dwrite))
            meanImg += dwrite.sum(axis=0)
            if ops['reg_tif_chan2']:
                if k==0:
                    if ops['functional_chan']!=ops['align_by_chan']:
                        tifroot = os.path.join(ops['save_path'], 'reg_tif')
                    else:
                        tifroot = os.path.join(ops['save_path'], 'reg_tif_chan2')
                    print(tifroot)
                    if not os.path.isdir(tifroot):
                        os.makedirs(tifroot)
                fname = 'file_chan%0.3d.tif'%k
                io.imsave(os.path.join(tifroot, fname), dwrite)
            k+=1
        if ops['functional_chan']!=ops['align_by_chan']:
            ops['meanImg'] = meanImg/ops['nframes']
        else:
            ops['meanImg_chan2'] = meanImg/ops['nframes']
        print('registered second channel in time %4.2f'%(toc(k0)))

    # compute metrics for registration
    ops = get_metrics(ops)
    print('computed registration metrics in time %4.2f'%(toc(k0)))

    np.save(ops['ops_path'], ops)
    return ops
