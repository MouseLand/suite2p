import numpy as np
import math, time
import glob, h5py, os
from scipy import signal
from suite2p import celldetect2 as celldetect2
from scipy import stats, io, signal

from skimage import io
def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

def fitMVGaus(y,x,lam,thres=2.5):
    ''' computes 2D gaussian fit to data and returns ellipse of radius thres standard deviations
    inputs:
        y, x, lam, thres
            y,x: pixel locations
            lam: pixel weights
        thres: number of standard deviations at which to draw ellipse
    outputs:
        mu, cov, ellipse, area
            mu: mean of gaussian fit
            cov: covariance of gaussian fit
            radii: half of major and minor axis lengths of elliptical fit
            ellipse: coordinates of elliptical fit
            area: area of ellipse
    '''
    # normalize pixel weights
    lam = lam / lam.sum()
    # mean of gaussian
    yx = np.stack((y,x))
    mu  = (lam*yx).sum(axis=-1)
    yx = yx - np.expand_dims(mu, axis=1)
    yx = yx * lam**.5
    #yx  = np.concatenate((y*lam**0.5, x*lam**0.5),axis=0)
    cov = yx @ yx.transpose()
    # radii of major and minor axes
    radii,evec  = np.linalg.eig(cov)
    radii = np.maximum(0, np.real(radii))
    radii       = thres * radii**.5
    # compute pts of ellipse
    npts = 100
    p = np.expand_dims(np.linspace(0, 2*math.pi, npts),axis=1)
    p = np.concatenate((np.cos(p), np.sin(p)),axis=1)
    ellipse = (p * radii) @ evec.transpose() + mu
    area = (radii[0] * radii[1])**0.5 * math.pi
    radii  = np.sort(radii)[::-1]
    return mu, cov, radii, ellipse, area

def enhanced_mean_image(ops):
    if 1:
        I = ops['meanImg']
        Imed = signal.medfilt2d(I, 4*ops['diameter']+1)
        I = I - Imed
        Idiv = signal.medfilt2d(np.absolute(I), 4*ops['diameter']+1)
        I = I / (1e-10 + Idiv)
        mimg1 = -6
        mimg99 = 6
        mimg0 = I
    else:
        mimg0 = ops['meanImg']
        mimg0 = mimg0 - gaussian_filter(filters.minimum_filter(mimg0,50,mode='mirror'),
                                          10,mode='mirror')
        mimg0 = mimg0 / gaussian_filter(filters.maximum_filter(mimg0,50,mode='mirror'),
                                      10,mode='mirror')
        mimg1 = np.percentile(mimg0,1)
        mimg99 = np.percentile(mimg0,99)
    mimg0 = mimg0[ops['yrange'][0]:ops['yrange'][1], ops['xrange'][0]:ops['xrange'][1]]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0,np.minimum(1,mimg0))
    mimg = mimg0.min() * np.ones((ops['Ly'],ops['Lx']),np.float32)
    mimg[ops['yrange'][0]:ops['yrange'][1],
        ops['xrange'][0]:ops['xrange'][1]] = mimg0
    ops['meanImgE'] = mimg
    return ops

def h5py_to_binary(ops):
    nplanes = ops['nplanes']
    nchannels = ops['nchannels']
    ops1 = []
    # open all binary files for writing
    reg_file = []
    if nchannels>1:
        reg_file_chan2 = []
    for j in range(0,nplanes):
        ops['save_path'] = os.path.join(ops['save_path0'], 'suite2p', 'plane%d'%j)
        if ('fast_disk' not in ops) or len(ops['fast_disk'])>0:
            ops['fast_disk'] = ops['save_path0']
        ops['fast_disk'] = os.path.join(ops['fast_disk'], 'suite2p', 'plane%d'%j)
        ops['ops_path'] = os.path.join(ops['save_path'],'ops.npy')
        ops['reg_file'] = os.path.join(ops['fast_disk'], 'data.bin')
        if nchannels>1:
            ops['reg_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2.bin')
        if not os.path.isdir(ops['fast_disk']):
            os.makedirs(ops['fast_disk'])
        if not os.path.isdir(ops['save_path']):
            os.makedirs(ops['save_path'])
        ops1.append(ops.copy())
        reg_file.append(open(ops['reg_file'], 'wb'))
        if nchannels>1:
            reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))

    # open h5py file for reading
    key = ops['h5py_key']

    with h5py.File(ops['h5py'], 'r') as f:
        # keep track of the plane identity of the first frame (channel identity is assumed always 0)
        nbatch = nplanes*nchannels*math.ceil(ops['batch_size']/(nplanes*nchannels))
        nframes_all = f[key].shape[0]
        # loop over all tiffs
        i0 = 0
        while 1:
            irange = np.arange(i0, min(i0+nbatch, nframes_all), 1)
            if irange.size==0:
                break
            im = f[key][irange, :, :]
            if i0==0:
                ops1[j]['meanImg'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                if nchannels>1:
                    ops1[j]['meanImg_chan2'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                ops1[j]['nframes'] = 0
            nframes = im.shape[0]
            for j in range(0,nplanes):
                im2write = im[np.arange(j, nframes, nplanes*nchannels),:,:]
                reg_file[j].write(bytearray(im2write))
                ops1[j]['meanImg'] += im2write.astype(np.float32).sum(axis=0)
                if nchannels>1:
                    im2write = im[np.arange(j+1, nframes, nplanes*nchannels),:,:]
                    reg_file_chan2[j].write(bytearray(im2write))
                    ops1[j]['meanImg_chan2'] += im2write.astype(np.float32).sum(axis=0)
                ops1[j]['nframes'] += im2write.shape[0]
            i0 += nframes
    # write ops files
    do_registration = ops['do_registration']
    for ops in ops1:
        ops['Ly'] = im2write.shape[1]
        ops['Lx'] = im2write.shape[2]
        if not do_registration:
            ops['yrange'] = np.array([0,ops['Ly']])
            ops['xrange'] = np.array([0,ops['Lx']])
        ops['meanImg'] /= ops['nframes']
        if nchannels>1:
            ops['meanImg_chan2'] /= ops['nframes']
        np.save(ops['ops_path'], ops)
    # close all binary files and write ops files
    for j in range(0,nplanes):
        reg_file[j].close()
        if nchannels>1:
            reg_file_chan2[j].close()
    return ops1

def tiff_to_binary(ops):
    nplanes = ops['nplanes']
    nchannels = ops['nchannels']
    ops1 = []
    # open all binary files for writing
    reg_file = []
    if nchannels>1:
        reg_file_chan2 = []
    for j in range(0,nplanes):
        fpath = os.path.join(ops['save_path0'], 'suite2p', 'plane%d'%j)
        ops['save_path'] = fpath
        if ('fast_disk' not in ops) or len(ops['fast_disk'])>0:
            ops['fast_disk'] = ops['save_path0']
        ops['fast_disk'] = os.path.join(ops['fast_disk'], 'suite2p', 'plane%d'%j)
        ops['ops_path'] = os.path.join(ops['save_path'],'ops.npy')
        ops['reg_file'] = os.path.join(ops['fast_disk'], 'data.bin')
        if nchannels>1:
            ops['reg_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2.bin')
        if not os.path.isdir(ops['fast_disk']):
            os.makedirs(ops['fast_disk'])
        if not os.path.isdir(ops['save_path']):
            os.makedirs(ops['save_path'])
        ops1.append(ops.copy())
        reg_file.append(open(ops['reg_file'], 'wb'))
        if nchannels>1:
            reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))
    fs, ops = get_tif_list(ops) # look for tiffs in all requested folders
    # loop over all tiffs

    for ik, file in enumerate(fs):
        # keep track of the plane identity of the first frame (channel identity is assumed always 0)
        if ops['first_tiffs'][ik]:
            iplane = 0
        im = io.imread(file)
        if len(im.shape)<3:
            im = np.expand_dims(im, axis=0)
        nframes = im.shape[0]
        for j in range(0,nplanes):
            if ik==0:
                ops1[j]['meanImg'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                if nchannels>1:
                    ops1[j]['meanImg_chan2'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                ops1[j]['nframes'] = 0
            i0 = nchannels * ((iplane+j)%nplanes)
            im2write = im[np.arange(int(i0), nframes, nplanes*nchannels),:,:]
            ops1[j]['meanImg'] += im2write.astype(np.float32).sum(axis=0)
            reg_file[j].write(bytearray(im2write))
            if nchannels>1:
                im2write = im[np.arange(int(i0)+1, nframes, nplanes*nchannels),:,:]
                reg_file_chan2[j].write(bytearray(im2write))
                ops1[j]['meanImg_chan2'] += im2write.astype(np.float32).sum(axis=0)
            ops1[j]['nframes']+= im2write.shape[0]
        iplane = (iplane+nframes/nchannels)%nplanes
    # write ops files
    do_registration = ops['do_registration']
    for ops in ops1:
        ops['Ly'] = im.shape[1]
        ops['Lx'] = im.shape[2]
        if not do_registration:
            ops['yrange'] = np.array([0,ops['Ly']])
            ops['xrange'] = np.array([0,ops['Lx']])
        ops['meanImg'] /= ops['nframes']
        if nchannels>1:
            ops['meanImg_chan2'] /= ops['nframes']
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
    nfs = 0
    ix = np.zeros((len(fold_list), 1), 'int32')
    for k,fld in enumerate(fold_list):
        ix[k] = len(fs)
        fs.extend(list_tifs(fld, ops['look_one_level_down']))
    ops['first_tiffs'] = np.zeros(len(fs), 'bool_')
    ops['first_tiffs'][ix] = True
    if len(fs)==0:
        raise Exception('Could not find any tifs')
    else:
        print('Found %d tifs'%(len(fs)))
    return fs, ops

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


def get_cells(ops):
    i0 = tic()
    if (type(ops['diameter']) is int) or len(ops['diameter'])<2:
        ops['diameter'] = int(np.array(ops['diameter']))
        ops['diameter'] = np.array((ops['diameter'], ops['diameter']))
    ops['diameter'] = np.array(ops['diameter']).astype('int32')
    print(ops['diameter'])
    ops, stat = celldetect2.sourcery(ops)
    print('time %4.4f. Found %d ROIs'%(toc(i0), len(stat)))
    # extract fluorescence and neuropil
    F, Fneu, ops = celldetect2.extractF(ops, stat)
    print('time %4.4f. Extracted fluorescence from %d ROIs'%(toc(i0), len(stat)))
    # subtract neuropil
    dF = F - ops['neucoeff'] * Fneu
    # compute activity statistics for classifier
    sk = stats.skew(dF, axis=1)
    sd = np.std(dF, axis=1)
    npix = np.array([stat[n]['npix'] for n in range(len(stat))]).astype('float32')
    npix /= np.mean(npix)
    for k in range(F.shape[0]):
        stat[k]['skew'] = sk[k]
        stat[k]['std']  = sd[k]
        stat[k]['npix_norm'] = npix[k]
    # add enhanced mean image
    ops = enhanced_mean_image(ops)
    # save ops
    np.save(ops['ops_path'], ops)
    # save results
    fpath = ops['save_path']
    np.save(os.path.join(fpath,'F.npy'), F)
    np.save(os.path.join(fpath,'Fneu.npy'), Fneu)
    np.save(os.path.join(fpath,'stat.npy'), stat)
    iscell = np.ones((len(stat),2))
    np.save(os.path.join(fpath, 'iscell.npy'), iscell)
    print('results saved to %s'%ops['save_path'])
    return ops

def combined(ops1):
    '''
    Combines all the entries in ops1 into a single result file.
    Multi-plane recordings are arranged to best tile a square.
    Multi-roi recordings are arranged by their dx,dy physical localization.
    '''
    ops = ops1[0]
    if ('dx' not in ops) or ('dy' not in ops):
        Lx = ops['Lx']
        Ly = ops['Ly']
        nX = np.ceil(np.sqrt(ops['Ly'] * ops['Lx'] * len(ops1))/ops['Lx'])
        nX = int(nX)
        nY = int(np.ceil(len(ops1)/nX))
        for j in range(len(ops1)):
            ops1[j]['dx'] = (j%nX) * Lx
            ops1[j]['dy'] = int(j/nX) * Ly
    LY = int(np.amax(np.array([ops['Ly']+ops['dy'] for ops in ops1])))
    LX = int(np.amax(np.array([ops['Lx']+ops['dx'] for ops in ops1])))
    meanImg = np.zeros((LY, LX))
    meanImgE = np.zeros((LY, LX))
    Vcorr = np.zeros((LY, LX))
    Nfr = np.amax(np.array([ops['nframes'] for ops in ops1]))
    for k,ops in enumerate(ops1):
        fpath = ops['save_path']
        stat0 = np.load(os.path.join(fpath,'stat.npy'))
        xrange = np.arange(ops['dx'],ops['dx']+ops['Lx'])
        yrange = np.arange(ops['dy'],ops['dy']+ops['Ly'])
        meanImg[np.ix_(yrange, xrange)] = ops['meanImg']
        meanImgE[np.ix_(yrange, xrange)] = ops['meanImgE']
        xrange = np.arange(ops['dx']+ops['xrange'][0],ops['dx']+ops['xrange'][-1])
        yrange = np.arange(ops['dy']+ops['yrange'][0],ops['dy']+ops['yrange'][-1])
        Vcorr[np.ix_(yrange, xrange)] = ops['Vcorr']
        for j in range(len(stat0)):
            stat0[j]['xpix'] += ops['dx']
            stat0[j]['ypix'] += ops['dy']
            stat0[j]['med'][0] += ops['dy']
            stat0[j]['med'][1] += ops['dx']
        F0    = np.load(os.path.join(fpath,'F.npy'))
        Fneu0 = np.load(os.path.join(fpath,'Fneu.npy'))
        spks0 = np.load(os.path.join(fpath,'spks.npy'))
        iscell0 = np.load(os.path.join(fpath,'iscell.npy'))
        nn,nt = F0.shape
        if nt<Nfr:
            fcat    = np.zeros((nn,Nfr-nt), 'float32')
            print(F0.shape)
            print(fcat.shape)
            F0      = np.concatenate((F0, fcat), axis=1)
            spks0   = np.concatenate((spks0, fcat), axis=1)
            Fneu0   = np.concatenate((Fneu0, fcat), axis=1)
        if k==0:
            F, Fneu, spks,stat,iscell = F0, Fneu0, spks0,stat0, iscell0
        else:
            F    = np.concatenate((F, F0))
            Fneu = np.concatenate((Fneu, Fneu0))
            spks = np.concatenate((spks, spks0))
            stat = np.concatenate((stat,stat0))
            iscell = np.concatenate((iscell,iscell0))
    ops['meanImg']  = meanImg
    ops['meanImgE'] = meanImgE
    ops['Vcorr'] = Vcorr
    ops['Ly'] = LY
    ops['Lx'] = LX
    ops['xrange'] = [0, ops['Lx']]
    ops['yrange'] = [0, ops['Ly']]
    fpath = os.path.join(ops['save_path0'], 'suite2p', 'combined')
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    ops['save_path'] = fpath
    np.save(os.path.join(fpath, 'F.npy'), F)
    np.save(os.path.join(fpath, 'Fneu.npy'), Fneu)
    np.save(os.path.join(fpath, 'spks.npy'), spks)
    np.save(os.path.join(fpath, 'ops.npy'), ops)
    np.save(os.path.join(fpath, 'stat.npy'), stat)
    np.save(os.path.join(fpath, 'iscell.npy'), iscell)

    # save as matlab file
    if ('save_mat' in ops) and ops['save_mat']:
        matpath = os.path.join(ops['save_path'],'Fall.mat')
        scipy.io.savemat(matpath, {'stat': stat,
                                   'ops': ops,
                                   'F': F,
                                   'Fneu': Fneu,
                                   'spks': spks,
                                   'iscell': iscell})
    return ops
