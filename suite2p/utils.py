import numpy as np
from natsort import natsorted
import math, time
import glob, h5py, os, json
from scipy import signal
from suite2p import celldetect2 as celldetect2
from suite2p import utils, register, nonrigid
from scipy import stats, signal
from scipy.sparse import linalg
import scipy.io
from skimage.external.tifffile import imread, TiffFile
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

def init_ops(ops):
    if 'lines' in ops:
        lines = ops['lines']
        ops['nplanes'] = len(ops['lines'])
    nplanes = ops['nplanes']
    nchannels = ops['nchannels']
    ops1 = []
    if ('fast_disk' not in ops) or len(ops['fast_disk'])==0:
        ops['fast_disk'] = ops['save_path0']
    fast_disk = ops['fast_disk']
    # for mesoscope recording FOV locations
    if 'dy' in ops and ops['dy']!='':
        dy = ops['dy']
        dx = ops['dx']
    # compile ops into list across planes
    for j in range(0,nplanes):
        ops['save_path'] = os.path.join(ops['save_path0'], 'suite2p', 'plane%d'%j)
        if ('fast_disk' not in ops) or len(ops['fast_disk'])==0:
            ops['fast_disk'] = ops['save_path0']
        ops['fast_disk'] = os.path.join(fast_disk, 'suite2p', 'plane%d'%j)
        ops['ops_path'] = os.path.join(ops['save_path'],'ops.npy')
        ops['reg_file'] = os.path.join(ops['fast_disk'], 'data.bin')
        if 'lines' in ops:
            ops['lines'] = lines[j]
        if nchannels>1:
            ops['reg_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2.bin')
        if 'dy' in ops and ops['dy']!='':
            ops['dy'] = dy[j]
            ops['dx'] = dx[j]
        if not os.path.isdir(ops['fast_disk']):
            os.makedirs(ops['fast_disk'])
        if not os.path.isdir(ops['save_path']):
            os.makedirs(ops['save_path'])
        ops1.append(ops.copy())
    return ops1

def list_h5(ops):
    froot = os.path.dirname(ops['h5py'])
    lpath = os.path.join(froot, "*.h5")
    fs = natsorted(glob.glob(lpath))
    return fs

def h5py_to_binary(ops):
    # copy ops to list where each element is ops for each plane
    ops1 = utils.init_ops(ops)

    nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']
    # open all binary files for writing
    reg_file = []
    reg_file_chan2=[]
    for ops in ops1:
        reg_file.append(open(ops['reg_file'], 'wb'))
        if nchannels>1:
            reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))
    # open h5py file for reading
    key = ops1[0]['h5py_key']
    if ops1[0]['look_one_level_down']:
        h5list = list_h5(ops1[0])
        print('using a list of h5 files:')
        print(h5list)
    else:
        h5list = [ops1[0]['h5py']]
    iall = 0
    for h5 in h5list:
        with h5py.File(h5, 'r') as f:
            # if h5py data is 4D instead of 3D, assume that
            # data = nframes x nplanes x pixels x pixels
            hdims = f[key].ndim
            # keep track of the plane identity of the first frame (channel identity is assumed always 0)
            nbatch = nplanes*nchannels*math.ceil(ops1[0]['batch_size']/(nplanes*nchannels))
            if hdims==3:
                nframes_all = f[key].shape[0]
            else:
                nframes_all = f[key].shape[0] * f[key].shape[1]
            nbatch = min(nbatch, nframes_all)
            if nchannels>1:
                nfunc = ops['functional_chan'] - 1
            else:
                nfunc = 0
            # loop over all tiffs
            ik = 0
            while 1:
                if hdims==3:
                    irange = np.arange(ik, min(ik+nbatch, nframes_all), 1)
                    if irange.size==0:
                        break
                    im = f[key][irange, :, :]
                else:
                    irange = np.arange(ik/nplanes, min(ik/nplanes+nbatch/nplanes, nframes_all/nplanes), 1)
                    if irange.size==0:
                        break
                    im = f[key][irange,:,:,:]
                    im = np.reshape(im, (im.shape[0]*nplanes,im.shape[2],im.shape[3]))
                nframes = im.shape[0]
                for j in range(0,nplanes):
                    if iall==0:
                        ops1[j]['meanImg'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                        if nchannels>1:
                            ops1[j]['meanImg_chan2'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                        ops1[j]['nframes'] = 0
                    i0 = nchannels * ((j)%nplanes)
                    im2write = im[np.arange(int(i0)+nfunc, nframes, nplanes*nchannels),:,:].astype(np.int16)
                    reg_file[j].write(bytearray(im2write))
                    ops1[j]['meanImg'] += im2write.astype(np.float32).sum(axis=0)
                    if nchannels>1:
                        im2write = im[np.arange(int(i0)+1-nfunc, nframes, nplanes*nchannels),:,:].astype(np.int16)
                        reg_file_chan2[j].write(bytearray(im2write))
                        ops1[j]['meanImg_chan2'] += im2write.astype(np.float32).sum(axis=0)
                    ops1[j]['nframes'] += im2write.shape[0]
                ik += nframes
                iall += nframes
        # write ops files
    do_registration = ops1[0]['do_registration']
    do_nonrigid = ops1[0]['nonrigid']
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
    # copy ops to list where each element is ops for each plane
    ops1 = utils.init_ops(ops)
    nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']
    # open all binary files for writing
    reg_file = []
    reg_file_chan2=[]
    for ops in ops1:
        reg_file.append(open(ops['reg_file'], 'wb'))
        if nchannels>1:
            reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))
    fs, ops = get_tif_list(ops1[0]) # look for tiffs in all requested folders
    batch_size = 2000
    batch_size = nplanes*nchannels*math.ceil(batch_size/(nplanes*nchannels))
    # loop over all tiffs
    for ik, file in enumerate(fs):
        # size of tiff
        tif = TiffFile(file)
        Ltif = len(tif)
        # keep track of the plane identity of the first frame (channel identity is assumed always 0)
        if ops['first_tiffs'][ik]:
            iplane = 0
        ix = 0
        while 1:
            if ix >= Ltif:
                break
            nfr = min(Ltif - ix, batch_size)
            im = imread(file, pages = range(ix, ix + nfr))
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=0)
            if im.shape[0] > nfr:
                im = im[:nfr, :, :]
            nframes = im.shape[0]
            for j in range(0,nplanes):
                if ik==0 and ix==0:
                    ops1[j]['meanImg'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                    if nchannels>1:
                        ops1[j]['meanImg_chan2'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                    ops1[j]['nframes'] = 0
                i0 = nchannels * ((iplane+j)%nplanes)
                if nchannels>1:
                    nfunc = ops['functional_chan']-1
                else:
                    nfunc = 0
                im2write = im[np.arange(int(i0)+nfunc, nframes, nplanes*nchannels),:,:].astype(np.int16)
                ops1[j]['meanImg'] += im2write.astype(np.float32).sum(axis=0)
                reg_file[j].write(bytearray(im2write))
                ops1[j]['nframes'] += im2write.shape[0]
                if nchannels>1:
                    im2write = im[np.arange(int(i0)+1-nfunc, nframes, nplanes*nchannels),:,:].astype(np.int16)
                    reg_file_chan2[j].write(bytearray(im2write))
                    ops1[j]['meanImg_chan2'] += im2write.astype(np.float32).sum(axis=0)
            iplane = (iplane - nframes/nchannels) % nplanes
            ix+=nframes
    print(ops1[0]['nframes'])
    # write ops files
    do_registration = ops['do_registration']
    do_nonrigid = ops1[0]['nonrigid']
    for ops in ops1:
        ops['Ly'],ops['Lx'] = ops['meanImg'].shape
        ops['filelist'] = fs
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

def split_multiops(ops1):
    for j in range(len(ops1)):
        if 'dx' in ops1[j] and np.size(ops1[j]['dx'])>1:
            ops1[j]['dx'] = ops1[j]['dx'][j]
            ops1[j]['dy'] = ops1[j]['dy'][j]
    return ops1
def mesoscan_to_binary(ops):
    # copy ops to list where each element is ops for each plane
    # load json file with line start stops
    if 'lines' not in ops:
        fpath = os.path.join(ops['data_path'][0], '*json')
        fs = glob.glob(fpath)
        with open(fs[0], 'r') as f:
            opsj = json.load(f)
        ops['nplanes'] = len(opsj)
    else:
        ops['nplanes'] = len(ops['lines'])
    ops1 = utils.init_ops(ops)
    if 'lines' not in ops:
        for j in range(len(ops1)):
            ops1[j] = {**ops1[j], **opsj[j]}.copy()
    nplanes = ops['nplanes']
    print(nplanes)
    nchannels = ops1[0]['nchannels']
    # open all binary files for writing
    reg_file = []
    reg_file_chan2=[]
    for ops in ops1:
        reg_file.append(open(ops['reg_file'], 'wb'))
        if nchannels>1:
            reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))
    fs, ops = get_tif_list(ops1[0]) # look for tiffs in all requested folders
    if nchannels>1:
        nfunc = ops['functional_chan']-1
    else:
        nfunc = 0
    batch_size = 500
    # loop over all tiffs
    for ik, file in enumerate(fs):
        ix = 0
        while 1:
            im = imread(file, pages = range(ix,ix+batch_size))
            if im.size==0:
                break
            #im = io.imread(file)
            if len(im.shape)<3:
                im = np.expand_dims(im, axis=0)
            nframes = im.shape[0]
            for j in range(0,nplanes):
                if ik==0:
                    ops1[j]['meanImg'] = np.zeros((len(ops1[j]['lines']),im.shape[2]),np.float32)
                    if nchannels>1:
                        ops1[j]['meanImg_chan2'] = np.zeros((len(ops1[j]['lines']),im.shape[2]),np.float32)
                    ops1[j]['nframes'] = 0
                im2write = im[:,ops1[j]['lines'],:].astype(np.int16)
                ops1[j]['meanImg'] += im2write.astype(np.float32).sum(axis=0)
                reg_file[j].write(bytearray(im2write))
                if nchannels>1:
                    #im2write = im[np.arange(1-nfunc, nframes, nplanes*nchannels),:,:].astype(np.int16)
                    reg_file_chan2[j].write(bytearray(im2write))
                    ops1[j]['meanImg_chan2'] += im2write.astype(np.float32).sum(axis=0)
                ops1[j]['nframes']+= im2write.shape[0]
            ix+=nframes
    # write ops files
    do_registration = ops['do_registration']
    do_nonrigid = ops1[0]['nonrigid']
    for ops in ops1:
        ops['Ly'],ops['Lx'] = ops['meanImg'].shape
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
    fs  = natsorted(glob.glob(lpath))
    lpath = os.path.join(froot, "*.tiff")
    fs2 = natsorted(glob.glob(lpath))
    fs.extend(fs2)
    if look_one_level_down:
        fdir = glob.glob(os.path.join(froot, "*", ""))
        for folder_down in fdir:
            lpath = os.path.join(folder_down, "*.tif")
            fs3 = natsorted(glob.glob(lpath))
            lpath = os.path.join(folder_down, "*.tiff")
            fs4 = natsorted(glob.glob(lpath))
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
    if len(fs)==0:
        print('Could not find any tiffs')
        raise Exception('no tiffs')
    else:
        ops['first_tiffs'] = np.zeros(len(fs), 'bool_')
        ops['first_tiffs'][ix] = True
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
    ops['diameter'] = np.array(ops['diameter'])
    if ops['diameter'].size<2:
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
    if ops['nchannels']>1:
        meanImg_chan2 = np.zeros((LY, LX))
    Vcorr = np.zeros((LY, LX))
    Nfr = np.amax(np.array([ops['nframes'] for ops in ops1]))
    for k,ops in enumerate(ops1):
        fpath = ops['save_path']
        stat0 = np.load(os.path.join(fpath,'stat.npy'))
        xrange = np.arange(ops['dx'],ops['dx']+ops['Lx'])
        yrange = np.arange(ops['dy'],ops['dy']+ops['Ly'])
        meanImg[np.ix_(yrange, xrange)] = ops['meanImg']
        meanImgE[np.ix_(yrange, xrange)] = ops['meanImgE']
        if ops['nchannels']>1:
            meanImg_chan2[np.ix_(yrange, xrange)] = ops['meanImg_chan2']
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
            #print(F0.shape)
            #print(fcat.shape)
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
    if ops['nchannels']>1:
        ops['meanImg_chan2'] = meanImg_chan2
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

def make_blocks(ops):
    ## split FOV into blocks to register separately
    Ly = ops['Ly']
    Lx = ops['Lx']
    if 'maxregshiftNR' not in ops:
        ops['maxregshiftNR'] = 5
    if 'block_size' not in ops:
        ops['block_size'] = [128, 128]

    ny = int(np.ceil(1.5 * float(Ly) / ops['block_size'][0]))
    nx = int(np.ceil(1.5 * float(Lx) / ops['block_size'][1]))

    if ops['block_size'][0]>=Ly:
        ops['block_size'][0] = Ly
        ny = 1
    if ops['block_size'][1]>=Lx:
        ops['block_size'][1] = Lx
        nx = 1

    ystart = np.linspace(0, Ly - ops['block_size'][0], ny).astype('int')
    xstart = np.linspace(0, Lx - ops['block_size'][1], nx).astype('int')
    ops['yblock'] = []
    ops['xblock'] = []
    for iy in range(ny):
        for ix in range(nx):
            yind = np.array([ystart[iy], ystart[iy]+ops['block_size'][0]])
            xind = np.array([xstart[ix], xstart[ix]+ops['block_size'][1]])
            ops['yblock'].append(yind)
            ops['xblock'].append(xind)
    ops['nblocks'] = [ny, nx]

    ys, xs = np.meshgrid(np.arange(nx), np.arange(ny))
    ys = ys.flatten()
    xs = xs.flatten()
    ds = (ys - ys[:,np.newaxis])**2 + (xs - xs[:,np.newaxis])**2
    R = np.exp(-ds)
    R = R / np.sum(R,axis=0)
    ops['NRsm'] = R.T

    return ops

def sample_frames(ops, ix):
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread =  np.int64(Ly*Lx*2)
    Lyc = ops['yrange'][-1] - ops['yrange'][0]
    Lxc = ops['xrange'][-1] - ops['xrange'][0]
    mov = np.zeros((len(ix), Lyc, Lxc), np.int16)
    # load and bin data
    with open(ops['reg_file'], 'rb') as reg_file:
        for i in range(len(ix)):
            reg_file.seek(nbytesread*ix[i], 0)
            buff = reg_file.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            data = np.reshape(data, (Ly, Lx))
            mov[i,:,:] = data[ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]]
    return mov

def pclowhigh(mov, nlowhigh, nPC):
    nframes, Ly, Lx = mov.shape
    mov = mov.reshape((nframes, -1))
    mov = mov.astype('float32')
    mimg = np.mean(mov, axis=0)
    mov -= mimg
    COV = mov @ mov.T
    w,v = linalg.eigsh(COV, k = nPC)
    v = v[:, ::-1]
    mov += mimg
    mov = mov.reshape((nframes, Ly, Lx))
    pclow  = np.zeros((nPC, Ly, Lx), 'float32')
    pchigh = np.zeros((nPC, Ly, Lx), 'float32')
    for i in range(nPC):
        isort = np.argsort(v[:,i])
        pclow[i,:,:] = np.mean(mov[isort[:nlowhigh],:,:], axis=0)
        pchigh[i,:,:] = np.mean(mov[isort[-nlowhigh:],:,:], axis=0)
    return pclow, pchigh, w

def metric_register(pclow, pchigh, do_phasecorr=True, smooth_sigma=1.15, block_size=(128,128), maxregshift=0.1, maxregshiftNR=5):
    ops = {
        'num_workers': -1,
        'snr_thresh': 1.25,
        'nonrigid': True,
        'num_workers': -1,
        'block_size': np.array(block_size),
        'maxregshiftNR': np.array(maxregshiftNR),
        'maxregshift': np.array(maxregshift),
        'subpixel': 10,
        'do_phasecorr': do_phasecorr,
        'smooth_sigma': smooth_sigma
        }
    nPC, ops['Ly'], ops['Lx'] = pclow.shape

    X = np.zeros((nPC,3))
    ops = make_blocks(ops)
    for i in range(nPC):
        refImg = pclow[i]
        Img = pchigh[i][np.newaxis, :, :]
        #Img = np.tile(Img, (1,1,1))
        maskMul, maskOffset, cfRefImg = register.prepare_masks(refImg, ops)
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.prepare_masks(refImg, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
        dwrite, ymax, xmax, cmax, yxnr = register.phasecorr(Img, refAndMasks, ops)
        X[i,1] = np.mean((yxnr[0]**2 + yxnr[1]**2)**.5)
        X[i,0] = np.mean((ymax[0]**2 + xmax[0]**2)**.5)
        X[i,2] = np.amax((yxnr[0]**2 + yxnr[1]**2)**.5)

    return X
