import gc
import numpy as np
from natsort import natsorted
import math, time
import glob, h5py, os, json
from scipy import signal
from suite2p import register, nonrigid, chan2detect, sparsedetect, roiextract
from scipy import stats, signal
from scipy.sparse import linalg
import scipy.io
#from skimage import io
from ScanImageTiffReader import ScanImageTiffReader
from skimage.external.tifffile import imread, TiffFile

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
    lam /= lam.sum()
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
    ''' computes enhanced mean image for GUI '''
    if 1:
        I = ops['meanImg'].astype(np.float32)
        if 'spatscale_pix' not in ops:
            if isinstance(ops['diameter'], int):
                diameter = np.array([ops['diameter'], ops['diameter']])
            else:
                diameter = np.array(ops['diameter'])
            ops['spatscale_pix'] = diameter[1]
            ops['aspect'] = diameter[0]/diameter[1]

        diameter = 4*np.array([ops['spatscale_pix'] * ops['aspect'], ops['spatscale_pix']]) + 1
        diameter = diameter.flatten().astype(np.int64)
        Imed = signal.medfilt2d(I, [diameter[0], diameter[1]])
        I = I - Imed
        Idiv = signal.medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
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
    nplanes = ops['nplanes']
    nchannels = ops['nchannels']
    if 'lines' in ops:
        lines = ops['lines']
    if 'iplane' in ops:
        iplane = ops['iplane']
        #ops['nplanes'] = len(ops['lines'])
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
        if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
            ops['raw_file'] = os.path.join(ops['fast_disk'], 'data_raw.bin')
        if 'lines' in ops:
            ops['lines'] = lines[j]
        if 'iplane' in ops:
            ops['iplane'] = iplane[j]
        if nchannels>1:
            ops['reg_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2.bin')
            if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
                ops['raw_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2_raw.bin')
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

def find_files_open_binaries(ops1, ish5):
    ''' find tiffs or h5 files, and open binaries to write to
        inputs ops1 (list over planes), ish5 (whether or not h5)
        returns ops1, fs (filelist), reg_file, reg_file_chan2 '''
    reg_file = []
    reg_file_chan2=[]


    for ops in ops1:
        nchannels = ops['nchannels']
        if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
            reg_file.append(open(ops['raw_file'], 'wb'))
            if nchannels>1:
                reg_file_chan2.append(open(ops['raw_file_chan2'], 'wb'))
        else:
            reg_file.append(open(ops['reg_file'], 'wb'))
            if nchannels>1:
                reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))

    if ish5:
        # find h5's
        if ops1[0]['look_one_level_down']:
            fs = list_h5(ops1[0])
            print('NOTE: using a list of h5 files:')
            print(fs)
        else:
            fs = [ops1[0]['h5py']]
    else:
        # find tiffs
        fs, ops2 = get_tif_list(ops1[0])
        for ops in ops1:
            ops['first_tiffs'] = ops2['first_tiffs']
            ops['frames_per_folder'] = np.zeros((ops2['first_tiffs'].sum(),), np.int32)
            ops['filelist'] = fs
    return ops1, fs, reg_file, reg_file_chan2

def h5py_to_binary(ops):
    # copy ops to list where each element is ops for each plane
    ops1 = init_ops(ops)

    nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']

    # open all binary files for writing
    ops1, h5list, reg_file, reg_file_chan2 = find_files_open_binaries(ops1, True)

    key = ops1[0]['h5py_key']
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
                if type(im[0,0,0]) == np.uint16:
                    im = im / 2
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

def open_tiff(file, sktiff):
    ''' opens tiff with either ScanImageTiffReader or skimage
    returns tiff and its length '''
    if sktiff:
        tif = TiffFile(file, fastij = False)
        Ltif = len(tif)
    else:
        tif = ScanImageTiffReader(file)
        tsize = tif.shape()
        if len(tsize) < 3:
            # single page tiffs
            Ltif = 1
        else:
            Ltif = tif.shape()[0]
    return tif, Ltif

def choose_tiff_reader(fs0, ops):
    try:
        tif = ScanImageTiffReader(fs0)
        tsize = tif.shape()
        if len(tsize) < 3:
            # single page tiffs
            im = tif.data()
        else:
            im = tif.data(beg=0, end=np.minimum(ops['batch_size'], tif.shape()[0]-1))
        tif.close()
        sktiff=False
    except:
        sktiff = True
        print('NOTE: ScanImageTiffReader not working for this tiff type, using scikit-image')
    if 'force_sktiff' in ops and ops['force_sktiff']:
        sktiff=True
        print('NOTE: user chose scikit-image for tiff reading')
    return sktiff

def tiff_to_binary(ops):
    ''' converts tiff to *.bin file '''
    ''' requires ops keys: nplanes, nchannels, data_path, look_one_level_down, reg_file '''
    ''' assigns ops keys: tiffreader, first_tiffs, frames_per_folder, nframes, meanImg, meanImg_chan2'''
    t0=tic()
    # copy ops to list where each element is ops for each plane
    ops1 = init_ops(ops)
    nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']

    # open all binary files for writing
    # look for tiffs in all requested folders
    ops1, fs, reg_file, reg_file_chan2 = find_files_open_binaries(ops1, False)
    ops = ops1[0]
    # try tiff readers
    sktiff = choose_tiff_reader(fs[0], ops1[0])

    batch_size = ops['batch_size']
    batch_size = nplanes*nchannels*math.ceil(batch_size/(nplanes*nchannels))

    # loop over all tiffs
    which_folder = -1
    ntotal=0
    for ik, file in enumerate(fs):
        # open tiff
        tif, Ltif = open_tiff(file, sktiff)
        # keep track of the plane identity of the first frame (channel identity is assumed always 0)
        if ops['first_tiffs'][ik]:
            which_folder += 1
            iplane = 0
        ix = 0

        while 1:
            if ix >= Ltif:
                break
            nfr = min(Ltif - ix, batch_size)
            # tiff reading
            if sktiff:
                im = imread(file, pages = range(ix, ix + nfr), fastij = False)
            else:
                if Ltif==1:
                    im = tif.data()
                else:
                    im = tif.data(beg=ix, end=ix+nfr)

            # for single-page tiffs, add 1st dim
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=0)

            # check if uint16
            if type(im[0,0,0]) == np.uint16:
                im = im // 2
                im = im.astype(np.int16)
            if type(im[0,0,0]) == np.uint8:
                im = im.astype(np.int16)

            if im.shape[0] > nfr:
                im = im[:nfr, :, :]
            nframes = im.shape[0]
            for j in range(0,nplanes):
                if ik==0 and ix==0:
                    ops1[j]['nframes'] = 0
                    ops1[j]['meanImg'] = np.zeros((im.shape[1], im.shape[2]), np.float32)
                    if nchannels>1:
                        ops1[j]['meanImg_chan2'] = np.zeros((im.shape[1], im.shape[2]), np.float32)
                i0 = nchannels * ((iplane+j)%nplanes)
                if nchannels>1:
                    nfunc = ops['functional_chan']-1
                else:
                    nfunc = 0
                im2write = im[int(i0)+nfunc:nframes:nplanes*nchannels]

                reg_file[j].write(bytearray(im2write))
                ops1[j]['nframes'] += im2write.shape[0]
                ops1[j]['frames_per_folder'][which_folder] += im2write.shape[0]
                #print(ops1[j]['frames_per_folder'][which_folder])
                if nchannels>1:
                    im2write = im[int(i0)+1-nfunc:nframes:nplanes*nchannels]
                    reg_file_chan2[j].write(bytearray(im2write))

            iplane = (iplane-nframes/nchannels)%nplanes
            ix+=nframes
            ntotal+=nframes
            if ntotal%(batch_size*4)==0:
                print('%d frames of binary, time %0.2f sec.'%(ntotal,toc(t0)))
        gc.collect()
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

def split_multiops(ops1):
    for j in range(len(ops1)):
        if 'dx' in ops1[j] and np.size(ops1[j]['dx'])>1:
            ops1[j]['dx'] = ops1[j]['dx'][j]
            ops1[j]['dy'] = ops1[j]['dy'][j]
    return ops1

def mesoscan_to_binary(ops):
    # copy ops to list where each element is ops for each plane
    # load json file with line start stops
    t0 = tic()
    if 'lines' not in ops:
        fpath = os.path.join(ops['data_path'][0], '*json')
        fs = glob.glob(fpath)
        with open(fs[0], 'r') as f:
            opsj = json.load(f)
        if 'nrois' in opsj:
            ops['nrois'] = opsj['nrois']
            ops['nplanes'] = opsj['nplanes']
            ops['dy'] = opsj['dy']
            ops['dx'] = opsj['dx']
            ops['fs'] = opsj['fs']
        elif 'nplanes' in opsj and 'lines' in opsj:
            ops['nrois'] = opsj['nplanes']
            ops['nplanes'] = 1
        else:
            ops['nplanes'] = len(opsj)
        ops['lines'] = opsj['lines']
    else:
        ops['nrois'] = len(ops['lines'])
    nplanes = ops['nplanes']

    print("NOTE: nplanes %d nrois %d => ops['nplanes'] = %d"%(nplanes,ops['nrois'],ops['nrois']*nplanes))
    # multiply lines across planes
    lines = ops['lines'].copy()
    dy = ops['dy'].copy()
    dx = ops['dx'].copy()
    ops['lines'] = [None] * nplanes * ops['nrois']
    ops['dy'] = [None] * nplanes * ops['nrois']
    ops['dx'] = [None] * nplanes * ops['nrois']
    ops['iplane'] = np.zeros((nplanes * ops['nrois'],), np.int32)
    for n in range(ops['nrois']):
        ops['lines'][n::ops['nrois']] = [lines[n]] * nplanes
        ops['dy'][n::ops['nrois']] = [dy[n]] * nplanes
        ops['dx'][n::ops['nrois']] = [dx[n]] * nplanes
        ops['iplane'][n::ops['nrois']] = np.arange(0, nplanes, 1, int)
    ops['nplanes'] = nplanes * ops['nrois']
    ops1 = init_ops(ops)

    # this shouldn't make it here
    if 'lines' not in ops:
        for j in range(len(ops1)):
            ops1[j] = {**ops1[j], **opsj[j]}.copy()

    # open all binary files for writing
    # look for tiffs in all requested folders
    ops1, fs, reg_file, reg_file_chan2 = find_files_open_binaries(ops1, False)
    ops = ops1[0]

    #nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']
    if nchannels>1:
        nfunc = ops['functional_chan']-1
    else:
        nfunc = 0
    batch_size = ops['batch_size']

    # which tiff reader works for user's tiffs
    sktiff = choose_tiff_reader(fs[0], ops1[0])

    # loop over all tiffs
    which_folder = -1
    ntotal=0
    for ik, file in enumerate(fs):
        # open tiff
        tif, Ltif = open_tiff(file, sktiff)
        if ops['first_tiffs'][ik]:
            which_folder += 1
            iplane = 0
        ix = 0
        while 1:
            if ix >= Ltif:
                break
            nfr = min(Ltif - ix, batch_size)
            if sktiff:
                im = imread(file, pages = range(ix, ix + nfr), fastij = False)
            else:
                if Ltif==1:
                    im = tif.data()
                else:
                    im = tif.data(beg=ix, end=ix+nfr)
            if im.size==0:
                break
            #im = io.imread(file)
            if len(im.shape)<3:
                im = np.expand_dims(im, axis=0)

            if im.shape[0] > nfr:
                im = im[:nfr, :, :]
            nframes = im.shape[0]

            for j in range(0, ops['nplanes']):
                jlines = np.array(ops1[j]['lines']).astype(np.int32)
                jplane = ops1[j]['iplane']
                if ik==0 and ix==0:
                    ops1[j]['meanImg'] = np.zeros((len(jlines), im.shape[2]), np.float32)
                    if nchannels>1:
                        ops1[j]['meanImg_chan2'] = np.zeros((len(jlines), im.shape[2]), np.float32)
                    ops1[j]['nframes'] = 0
                i0 = nchannels * ((iplane+jplane)%nplanes)
                if nchannels>1:
                    nfunc = ops['functional_chan']-1
                else:
                    nfunc = 0
                #frange = np.arange(int(i0)+nfunc, nframes, nplanes*nchannels)
                im2write = im[int(i0)+nfunc:nframes:nplanes*nchannels, jlines[0]:(jlines[-1]+1), :]
                #im2write = im[np.ix_(frange, jlines, np.arange(0,im.shape[2],1,int))]
                #ops1[j]['meanImg'] += im2write.astype(np.float32).sum(axis=0)
                reg_file[j].write(bytearray(im2write))
                ops1[j]['nframes'] += im2write.shape[0]
                ops1[j]['frames_per_folder'][which_folder] += im2write.shape[0]
                if nchannels>1:
                    frange = np.arange(int(i0)+1-nfunc, nframes, nplanes*nchannels)
                    im2write = im[np.ix_(frange, jlines, np.arange(0,im.shape[2],1,int))]
                    reg_file_chan2[j].write(bytearray(im2write))
                    #ops1[j]['meanImg_chan2'] += im2write.astype(np.float32).sum(axis=0)
            iplane = (iplane-nframes/nchannels)%nplanes
            ix+=nframes
            ntotal+=nframes
            if ntotal%(batch_size*4)==0:
                print('%d frames per binary, time %0.2f sec.'%(ntotal,toc(t0)))
        gc.collect()
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
    for j in range(0,ops['nplanes']):
        reg_file[j].close()
        if nchannels>1:
            reg_file_chan2[j].close()
    return ops1

def list_tifs(froot, look_one_level_down):
    '''
    list of tiffs in folder froot + one level down maybe
    '''
    first_tiffs = []
    lpath = os.path.join(froot, "*.tif")
    fs  = natsorted(glob.glob(lpath))
    lpath = os.path.join(froot, "*.tiff")
    fs2 = natsorted(glob.glob(lpath))
    fs.extend(fs2)
    if len(fs) > 0:
        first_tiffs.extend(np.zeros((len(fs),), 'bool'))
        first_tiffs[0] = True
    lfs = len(fs)
    if look_one_level_down:
        fdir = glob.glob(os.path.join(froot, "*", ""))
        for folder_down in fdir:
            lpath = os.path.join(folder_down, "*.tif")
            fs3 = natsorted(glob.glob(lpath))
            lpath = os.path.join(folder_down, "*.tiff")
            fs4 = natsorted(glob.glob(lpath))
            fs.extend(fs3)
            fs.extend(fs4)
            if len(fs3)+len(fs4) > 0:
                first_tiffs.extend(np.zeros((len(fs3)+len(fs4),), 'bool'))
                first_tiffs[lfs] = True
                lfs = len(fs)
    return fs, first_tiffs

def get_tif_list(ops):
    '''
    make list of tiffs to process
    if ops['subfolders'], then all tiffs ops['data_path'][0] / ops['subfolders'] / *.tif
    if ops['look_one_level_down'], then all tiffs in all folders + one level down
    if ops['tiff_list'], then ops['data_path'][0] / ops['tiff_list'] ONLY
    '''
    froot = ops['data_path']
    # use a user-specified list of tiffs
    if 'tiff_list' in ops:
        fsall = []
        for tif in ops['tiff_list']:
            fsall.append(os.path.join(froot[0], tif))
        ops['first_tiffs'] = np.zeros((len(fsall),), dtype=np.bool)
        ops['first_tiffs'][0] = True
        print('** Found %d tifs - converting to binary **'%(len(fsall)))
    else:
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
        fsall = []
        nfs = 0
        first_tiffs = []
        for k,fld in enumerate(fold_list):
            fs, ftiffs = list_tifs(fld, ops['look_one_level_down'])
            fsall.extend(fs)
            first_tiffs.extend(ftiffs)
        if len(fs)==0:
            print('Could not find any tiffs')
            raise Exception('no tiffs')
        else:
            ops['first_tiffs'] = np.array(first_tiffs)
            print('** Found %d tifs - converting to binary **'%(len(fsall)))
            #print('Found %d tifs'%(len(fsall)))
    return fsall, ops

def sub2ind(array_shape, rows, cols):
    inds = rows * array_shape[1] + cols
    return inds

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
    if 'meanImg_chan2_corrected' in ops:
        meanImg_chan2_corrected = np.zeros((LY, LX))
    if 'max_proj' in ops:
        max_proj = np.zeros((LY, LX))

    Vcorr = np.zeros((LY, LX))
    Nfr = np.amax(np.array([ops['nframes'] for ops in ops1]))
    for k,ops in enumerate(ops1):
        fpath = ops['save_path']
        stat0 = np.load(os.path.join(fpath,'stat.npy'), allow_pickle=True)
        xrange = np.arange(ops['dx'],ops['dx']+ops['Lx'])
        yrange = np.arange(ops['dy'],ops['dy']+ops['Ly'])
        meanImg[np.ix_(yrange, xrange)] = ops['meanImg']
        meanImgE[np.ix_(yrange, xrange)] = ops['meanImgE']
        if ops['nchannels']>1:
            meanImg_chan2[np.ix_(yrange, xrange)] = ops['meanImg_chan2']
        if 'meanImg_chan2_corrected' in ops:
            meanImg_chan2_corrected[np.ix_(yrange, xrange)] = ops['meanImg_chan2_corrected']

        xrange = np.arange(ops['dx']+ops['xrange'][0],ops['dx']+ops['xrange'][-1])
        yrange = np.arange(ops['dy']+ops['yrange'][0],ops['dy']+ops['yrange'][-1])
        Vcorr[np.ix_(yrange, xrange)] = ops['Vcorr']
        if 'max_proj' in ops:
            max_proj[np.ix_(yrange, xrange)] = ops['max_proj']
        for j in range(len(stat0)):
            stat0[j]['xpix'] += ops['dx']
            stat0[j]['ypix'] += ops['dy']
            stat0[j]['med'][0] += ops['dy']
            stat0[j]['med'][1] += ops['dx']
            stat0[j]['iplane'] = k
        F0    = np.load(os.path.join(fpath,'F.npy'))
        Fneu0 = np.load(os.path.join(fpath,'Fneu.npy'))
        spks0 = np.load(os.path.join(fpath,'spks.npy'))
        iscell0 = np.load(os.path.join(fpath,'iscell.npy'))
        if os.path.isfile(os.path.join(fpath,'redcell.npy')):
            redcell0 = np.load(os.path.join(fpath,'redcell.npy'))
            hasred = True
        else:
            redcell0 = []
            hasred = False
        nn,nt = F0.shape
        if nt<Nfr:
            fcat    = np.zeros((nn,Nfr-nt), 'float32')
            #print(F0.shape)
            #print(fcat.shape)
            F0      = np.concatenate((F0, fcat), axis=1)
            spks0   = np.concatenate((spks0, fcat), axis=1)
            Fneu0   = np.concatenate((Fneu0, fcat), axis=1)
        if k==0:
            F, Fneu, spks,stat,iscell,redcell = F0, Fneu0, spks0,stat0, iscell0, redcell0
        else:
            F    = np.concatenate((F, F0))
            Fneu = np.concatenate((Fneu, Fneu0))
            spks = np.concatenate((spks, spks0))
            stat = np.concatenate((stat,stat0))
            iscell = np.concatenate((iscell,iscell0))
            if hasred:
                redcell = np.concatenate((redcell,redcell0))
    ops['meanImg']  = meanImg
    ops['meanImgE'] = meanImgE
    if ops['nchannels']>1:
        ops['meanImg_chan2'] = meanImg_chan2
    if 'meanImg_chan2_corrected' in ops:
        ops['meanImg_chan2_corrected'] = meanImg_chan2_corrected
    if 'max_proj' in ops:
        ops['max_proj'] = max_proj
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
    if hasred:
        np.save(os.path.join(fpath, 'redcell.npy'), redcell)

    # save as matlab file
    if ('save_mat' in ops) and ops['save_mat']:
        matpath = os.path.join(ops['save_path'],'Fall.mat')
        scipy.io.savemat(matpath, {'stat': stat,
                                    'ops': ops,
                                    'F': F,
                                    'Fneu': Fneu,
                                    'spks': spks,
                                    'iscell': iscell,
                                    'redcell': redcell})
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

def sample_frames(ops, ix, reg_file):
    ''' get frames ix from reg_file
        frames are cropped by ops['yrange'] and ops['xrange']
    '''
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread =  np.int64(Ly*Lx*2)
    Lyc = ops['yrange'][-1] - ops['yrange'][0]
    Lxc = ops['xrange'][-1] - ops['xrange'][0]
    mov = np.zeros((len(ix), Lyc, Lxc), np.int16)
    # load and bin data
    with open(reg_file, 'rb') as reg_file:
        for i in range(len(ix)):
            reg_file.seek(nbytesread*ix[i], 0)
            buff = reg_file.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            data = np.reshape(data, (Ly, Lx))
            mov[i,:,:] = data[ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]]
    return mov
