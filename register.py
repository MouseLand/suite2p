from skimage import io
import glob 
import numpy as np
import time, os
import matplotlib.pyplot as plt
from numpy import fft
from numpy import random as rnd
import multiprocessing
from multiprocessing import Pool

eps0 = 1e-5;
sigL = 0.85 # smoothing width for up-sampling kernels, keep it between 0.5 and 1.0...
lpad = 3   # upsample from a square +/- lpad 
smoothSigma = 1.15 # smoothing constant
maskSlope   = 2. # slope of taper mask at the edges
    
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

def prepareMasks(refImg):    
    i0,Ly,Lx = refImg.shape    
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    mY = y.max() - 4.
    mX = x.max() - 4.    
    maskY = 1./(1.+np.exp((yy-mY)/maskSlope))
    maskX = 1./(1.+np.exp((xx-mX)/maskSlope))
    maskMul = maskY * maskX
    maskOffset = refImg.mean() * (1. - maskMul);
    hgx = np.exp(-np.square(xx/smoothSigma))
    hgy = np.exp(-np.square(yy/smoothSigma))
    hgg = hgy * hgx
    hgg = hgg/hgg.sum()
    fhg = np.real(fft.fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain
    cfRefImg   = np.conj(fft.fft2(refImg));
    absRef     = np.absolute(cfRefImg);
    cfRefImg   = cfRefImg / (eps0 + absRef) * fhg;    
    
    maskMul = maskMul.astype('float32')
    maskOffset = maskOffset.astype('float32')
    cfRefImg = cfRefImg.astype('complex64')
    return maskMul, maskOffset, cfRefImg

def correlation_map(data, refImg):
    (maskMul, maskOffset, cfRefImg) = prepareMasks(refImg)    
    data = data.astype('float32') * maskMul + maskOffset    
    X = fft.fft2(data)
    J = X / (eps0 + np.absolute(X))
    J = J * cfRefImg
    cc = np.real(fft.ifft2(J))
    cc = fft.fftshift(cc, axes=(1,2))
    return cc

def getXYup(cc, Ls, ops):
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
    data, refImg, ops, flag_zeroed = inputs
    nimg, Ly, Lx = data.shape        
    refImg = np.reshape(refImg, (1, Ly, Lx))    
    Lyhalf = int(np.floor(Ly/2))
    Lxhalf = int(np.floor(Lx/2))    
    if ops['maxregshift']>0:
        maxregshift = ops['maxregshift'] 
    else:
        maxregshift = np.round(.1*np.maximum(Ly, Lx))
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(Ly,Lx)/2.)-lpad))        
    cc = correlation_map(data, refImg)
    ymax, xmax, cmax = getXYup(cc, (lcorr,lpad, Lyhalf, Lxhalf), ops)        
    if flag_zeroed:
        xmax = xmax - xmax.mean()
        ymax = ymax - ymax.mean()        
    Y = shift_data((data, ymax,xmax))    
    return Y, ymax, xmax, cmax

def phasecorr(data, refImg, ops, flag_zeroed):        
    if ops['num_workers']<0:
        Y, ymax, xmax, cmax = phasecorr_worker((data, refImg, ops, flag_zeroed))
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
            dsplit.append([data[ilist,:, :], refImg, ops, flag_zeroed])            
        with Pool(num_cores) as p:
            results = p.map(phasecorr_worker, dsplit)
            
        Y = np.zeros_like(data)
        ymax = np.zeros((nimg,))
        xmax = np.zeros((nimg,))        
        cmax = np.zeros((nimg,))        
        for i in range(0,len(results)):
            Y[irange[i], :, :] = results[i][0]
            ymax[irange[i]] = results[i][1]
            xmax[irange[i]] = results[i][2]            
            cmax[irange[i]] = results[i][3]                
    return Y, ymax, xmax, cmax

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

def register_binary(ops):  
    # if ops is a list of dictionaries, each will be registered separately
    if type(ops) is list:
        for op in ops:
            op = register_binary(op)
        return ops    
    Ly = ops['Ly']
    Lx = ops['Lx']    
    ops['nframes'] = get_nFrames(ops) 
    refImg = pick_init(ops)   
    print('computed reference frame for registration')
    nbatch = ops['batch_size']
    nbytesread = 2 * Ly * Lx * nbatch    
    if ops['nchannels']>1 and ops['align_by_chan']>1:
        reg_file = open(ops['reg_file_chan2'], 'r+b')    
    else: 
        reg_file = open(ops['reg_file'], 'r+b')    
    yoff = []
    xoff = []
    corrXY = []    
    meanImg = np.zeros((Ly, Lx))    
    k = 0
    nfr = 0
    while True:
        buff = reg_file.read(nbytesread)    
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        if data.size==0:
            break            
        data = np.reshape(data, (-1, Ly, Lx))
        dwrite, ymax, xmax, cmax = phasecorr(data, refImg, ops, False)
        dwrite = dwrite.astype('int16')
        reg_file.seek(-2*dwrite.size,1)    
        reg_file.write(bytearray(dwrite))        
        meanImg += dwrite.sum(axis=0)
        yoff = np.append(yoff, ymax)
        xoff = np.append(xoff, xmax)
        corrXY = np.append(corrXY, cmax)
        if ops['reg_tif']:
            if k==0:
                tifroot = os.path.join(ops['save_path'], 'reg_tif')
                if not os.path.isdir(tifroot):
                    os.makedirs(tifroot)
            fname = 'file_chan%0.3d.tif'%k
            io.imsave(os.path.join(tifroot, fname), dwrite)            
        nfr += dwrite.shape[0]
        if k%10==0:
            print('registered %d/%d frames'%(nfr, ops['nframes']))
        k += 1
    ops['yoff'] = yoff
    ops['xoff'] = xoff
    ops['corrXY'] = corrXY
    ops['refImg'] = refImg    
    if ops['align_by_chan']>1:
        ops['meanImg_chan2'] = meanImg/ops['nframes']
    else:
        ops['meanImg'] = meanImg/ops['nframes']    
    if ops['nchannels']>1:
        ix = 0
        meanImg = np.zeros((Ly, Lx))   
        if ops['align_by_chan']>1:
            reg_file = open(ops['reg_file_chan2'], 'r+b')    
        else:
            reg_file = open(ops['reg_file'], 'r+b')            
        while True:
            buff = reg_file.read(nbytesread)    
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            if data.size==0:
                break            
            data = np.reshape(data, (-1, Ly, Lx))
            nframes = data.shape[0]
            # register by pre-determined amount
            dwrite = register_myshifts(ops, data, yoff[ix + np.arange(0,nframes)], xoff[ix + np.arange(0,nframes)])
            ix += nframes            
            dwrite = dwrite.astype('int16')
            reg_file.seek(-2*dwrite.size,1)    
            reg_file.write(bytearray(dwrite))        
            meanImg += dwrite.sum(axis=0)
            yoff = np.append(yoff, ymax)
            xoff = np.append(xoff, xmax)
            corrXY = np.append(corrXY, cmax)                
            if ops['align_by_chan']>1:
                ops['meanImg'] = meanImg/ops['nframes']
            else:
                ops['meanImg_chan2'] = meanImg/ops['nframes']   
    ymin = np.maximum(0, np.ceil(np.amax(yoff)))
    ymax = Ly + np.minimum(0, np.floor(np.amin(yoff)))
    ops['yrange'] = [int(ymin), int(ymax)]
    xmin = np.maximum(0, np.ceil(np.amax(xoff)))
    xmax = Lx + np.minimum(0, np.floor(np.amin(xoff)))
    ops['xrange'] = [int(xmin), int(xmax)]
    
    np.save(ops['ops_path'], ops)
    
    return ops

def subsample_frames(ops, nsamps):    
    nFrames = get_nFrames(ops) 
    Ly = ops['Ly']
    Lx = ops['Lx']    
    frames = np.zeros((nsamps, Ly, Lx), dtype='int16')    
    nbytesread = 2 * Ly * Lx    
    istart = np.linspace(0, nFrames, 1+nsamps).astype('int64')    
    reg_file = open(ops['reg_file'], 'rb')        
    for j in range(0,nsamps):        
        reg_file.seek(nbytesread * istart[j], 0)  
        buff = reg_file.read(nbytesread)        
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
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
        freg, ymax, xmax, cmax = phasecorr(frames, refImg, ops, True)
        isort = np.argsort(-cmax)
        if iter>=niter/2:
            nmax = int(frames.shape[0]/2)
        refImg = np.mean(freg[isort[1:nmax], :, :], axis=0)    
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

def tiff_to_binary(ops):    
    nplanes = ops['nplanes']
    nchannels = ops['nchannels']
    ops1 = []    
    # open all binary files for writing
    reg_file = []
    if nchannels>1:
        reg_file_chan2 = []            
    for j in range(0,nplanes):
        fpath = os.path.join(ops['data_path'][0], 'suite2p', 'plane%d'%j)
        ops['save_path'] = fpath
        ops['ops_path'] = os.path.join(fpath,'ops.npy')        
        ops['reg_file'] = os.path.join(fpath, 'data.bin')        
        if nchannels>1:
            ops['reg_file_chan2'] = os.path.join(fpath, 'data_chan2.bin')
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        ops1.append(ops.copy())                   
        reg_file.append(open(ops['reg_file'], 'wb'))
        if nchannels>1:
            reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))
    fs = get_tif_list(ops) # look for tiffs in all requested folders
    # keep track of the plane identity of the first frame (channel identity is assumed always 0)
    iplane = 0
    # loop over all tiffs
    for file in fs:
        im = io.imread(file)
        nframes = im.shape[0]
        for j in range(0,nplanes):    
            i0 = nchannels * ((iplane+j)%nplanes)
            im2write = im[np.arange(int(i0), nframes, nplanes*nchannels),:,:]
            reg_file[j].write(bytearray(im2write))
            if nchannels>1:
                im2write = im[np.arange(i0+1, nframes, nplanes*nchannels),:,:]
                reg_file_chan2[j].write(bytearray(im2write))            
        iplane = (iplane+nframes/nchannels)%nplanes    
    # write ops files
    for ops in ops1:        
        ops['Ly'] = im.shape[1]
        ops['Lx'] = im.shape[2]        
        np.save(ops['ops_path'], ops)    
    # close all binary files and write ops files
    for j in range(0,nplanes):
        reg_file[j].close()        
        if nchannels>1:
            reg_file_chan2[j].close()
    return ops1

def list_tifs(froot, look_one_level_down):      
    lpath = os.path.join(froot, "*.tif")
    fs  = sorted(glob.glob(lpath)) 
    lpath = os.path.join(froot, "*.tiff")
    fs2 = sorted(glob.glob(lpath))
    fs.extend(fs2)    
    if look_one_level_down:
        fdir = glob.glob(os.path.join(froot, "*", ""))
        for folder_down in fdir:            
            lpath = os.path.join(froot, folder_down, "*.tif")
            fs3 = sorted(glob.glob(lpath))
            lpath = os.path.join(froot, folder_down, "*.tiff")
            fs4 = sorted(glob.glob(lpath))
            fs.extend(fs3)
            fs.extend(fs4)            
    return fs

def get_tif_list(ops):
    froot = ops['data_path']
    
    if len(froot)==1:            
        if len(ops['subfolders'])==0:        
            fold_list = ops['data_path']
        else:
            fold_list = []
            for folder_down in ops['subfolders']:
                fold = os.path.join(froot[0], folder_down)
                fold_list.append(fold)            
    else:
        fold_list = froot
    fs = []
    for fld in fold_list:        
        fs.extend(list_tifs(fld, ops['look_one_level_down']))
        
    if len(fs)==0:
        raise Exception('Could not find any tifs')
    else:
        print('Found %d tifs'%(len(fs)))
    return fs

def get_tif_list_old(ops):
    froot = ops['data_path']
    if len(ops['subfolders'])==0:        
        fs = list_tifs(ops, froot)
    else:
        fs = []
        for folder_down in ops['subfolders']:
            fold = os.path.join(froot, folder_down)
            fs.extend(list_tifs(ops, fold))            
    if fs is None:
        raise Exception('Could not find any tifs')
    