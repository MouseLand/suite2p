import gc
import numpy as np
from natsort import natsorted
import math, time
import glob, os, json
from scipy import signal
from scipy import stats, signal
from scipy.sparse import linalg
import scipy.io
#from skimage import io
from ScanImageTiffReader import ScanImageTiffReader
from skimage.external.tifffile import imread, TiffFile, TiffWriter
from . import utils

def write(data, ops, k, ichan):
    """ writes frames to tiffs

    Parameters
    ----------
    data : int16
        frames x Ly x Lx

    ops : dictionary
        requires 'functional_chan', 'align_by_chan'

    k : int
        number of tiff

    ichan : bool
        channel is ops['align_by_chan']

    """
    if ichan:
        if ops['functional_chan']==ops['align_by_chan']:
            tifroot = os.path.join(ops['save_path'], 'reg_tif')
            wchan = 0
        else:
            tifroot = os.path.join(ops['save_path'], 'reg_tif_chan2')
            wchan = 1
    else:
        if ops['functional_chan']==ops['align_by_chan']:
            tifroot = os.path.join(ops['save_path'], 'reg_tif_chan2')
            wchan = 1
        else:
            tifroot = os.path.join(ops['save_path'], 'reg_tif')
            wchan = 0
    if not os.path.isdir(tifroot):
        os.makedirs(tifroot)
    fname = 'file%0.3d_chan%d.tif'%(k,wchan)
    with TiffWriter(os.path.join(tifroot, fname)) as tif:
        for i in range(data.shape[0]):
            tif.save(data[i])


def open_tiff(file, sktiff):
    """ opens tiff with either ScanImageTiffReader or skimage
    returns tiff and its length

    """
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
    """  chooses tiff reader (ScanImageTiffReader is default)

    tries to open tiff with ScanImageTiffReader and if it fails sets sktiff to True

    Parameters
    ----------
    fs0 : string
        path to first tiff in list

    ops : dictionary
        'batch_size'

    Returns
    -------
    sktiff : bool
        whether or not to use scikit-image tiff reader

    """

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
    """  finds tiff files and writes them to binaries

    Parameters
    ----------
    ops : dictionary
        'nplanes', 'data_path', 'save_path', 'save_folder', 'fast_disk', 'nchannels', 'keep_movie_raw', 'look_one_level_down'

    Returns
    -------
        ops1 : list of dictionaries
            ops1[j]['reg_file'] or ops1[j]['raw_file'] is created binary
            assigns keys 'Ly', 'Lx', 'tiffreader', 'first_tiffs',
            'frames_per_folder', 'nframes', 'meanImg', 'meanImg_chan2'

    """

    t0=time.time()
    # copy ops to list where each element is ops for each plane
    ops1 = utils.init_ops(ops)
    nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']

    # open all binary files for writing
    # look for tiffs in all requested folders
    ops1, fs, reg_file, reg_file_chan2 = utils.find_files_open_binaries(ops1, False)
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
                    ops1[j]['frames_per_file'] = np.zeros((len(fs),), dtype=int)
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
                ops1[j]['frames_per_file'][ik] += im2write.shape[0]
                ops1[j]['frames_per_folder'][which_folder] += im2write.shape[0]
                #print(ops1[j]['frames_per_folder'][which_folder])
                if nchannels>1:
                    im2write = im[int(i0)+1-nfunc:nframes:nplanes*nchannels]
                    reg_file_chan2[j].write(bytearray(im2write))

            iplane = (iplane-nframes/nchannels)%nplanes
            ix+=nframes
            ntotal+=nframes
            if ntotal%(batch_size*4)==0:
                print('%d frames of binary, time %0.2f sec.'%(ntotal,time.time()-t0))
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

def mesoscan_to_binary(ops):
    """ finds mesoscope tiff files and writes them to binaries

    Parameters
    ----------
    ops : dictionary
        'nplanes', 'data_path', 'save_path', 'save_folder', 'fast_disk',
        'nchannels', 'keep_movie_raw', 'look_one_level_down', 'lines', 'dx', 'dy'

    Returns
    -------
        ops1 : list of dictionaries
            ops1[j]['reg_file'] or ops1[j]['raw_file'] is created binary
            assigns keys 'Ly', 'Lx', 'tiffreader', 'first_tiffs', 'frames_per_folder',
            'nframes', 'meanImg', 'meanImg_chan2'

    """
    t0 = time.time()
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
    ops1 = utils.init_ops(ops)

    # this shouldn't make it here
    if 'lines' not in ops:
        for j in range(len(ops1)):
            ops1[j] = {**ops1[j], **opsj[j]}.copy()

    # open all binary files for writing
    # look for tiffs in all requested folders
    ops1, fs, reg_file, reg_file_chan2 = utils.find_files_open_binaries(ops1, False)
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
                print('%d frames per binary, time %0.2f sec.'%(ntotal,time.time()-t0))
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


def ome_to_binary(ops):
    """
    converts ome.tiff to *.bin file for non-interleaved red channel recordings
    assumes SINGLE-PAGE tiffs where first channel has string 'Ch1'
    and also SINGLE FOLDER

    Parameters
    ----------
    ops : dictionary
        keys nplanes, nchannels, data_path, look_one_level_down, reg_file

    Returns
    -------
    ops1 : list of dictionaries
        creates binaries ops1[j]['reg_file']
        assigns keys: tiffreader, first_tiffs, frames_per_folder, nframes, meanImg, meanImg_chan2
    """
    t0=time.time()
    # copy ops to list where each element is ops for each plane
    ops1 = utils.init_ops(ops)
    nplanes = ops1[0]['nplanes']

    # open all binary files for writing
    # look for tiffs in all requested folders
    ops1, fs, reg_file, reg_file_chan2 = utils.find_files_open_binaries(ops1, False)
    ops = ops1[0]

    fs_Ch1, fs_Ch2 = [], []
    for f in fs:
        if f.find('Ch1')>-1:
            if ops['functional_chan'] == 1:
                fs_Ch1.append(f)
            else:
                fs_Ch2.append(f)
        else:
            if ops['functional_chan'] == 1:
                fs_Ch2.append(f)
            else:
                fs_Ch1.append(f)

    if len(fs_Ch2)==0:
        ops1[0]['nchannels'] = 1
    nchannels = ops1[0]['nchannels']

    # loop over all tiffs
    which_folder = 0
    tif = ScanImageTiffReader(fs_Ch1[0])
    im0 = tif.data()
    Ly,Lx = im0.shape
    j=0

    for j in range(nplanes):
        ops1[j]['nframes'] = 0
        ops1[j]['frames_per_folder'][0] = 0
        ops1[j]['meanImg'] = np.zeros((Ly,Lx))
        if nchannels > 1:
            ops1[j]['meanImg_chan2'] = np.zeros((Ly,Lx))
    ix=0

    j=0
    for ik, file in enumerate(fs_Ch1):
        # open tiff
        tif = ScanImageTiffReader(file)
        im = tif.data()
        tif.close()
        if type(im[0,0]) == np.uint16:
            im = im // 2
            im = im.astype(np.int16)

        ops1[j]['meanImg'] += im
        ops1[j]['nframes'] += 1
        ops1[j]['frames_per_folder'][0] += 1

        reg_file[j].write(bytearray(im))
        ix+=1
        if ix%(1000)==0:
            print('%d frames of binary, time %0.2f sec.'%(ix,time.time()-t0))
        gc.collect()
        j+=1
        j = j%nplanes

    if nchannels>1:

        ix=0
        j=0
        for ik, file in enumerate(fs_Ch2):
            # open tiff
            tif = ScanImageTiffReader(file)
            im = tif.data()

            if type(im[0,0]) == np.uint16:
                im = im // 2
                im = im.astype(np.int16)

            ops1[j]['meanImg_chan2'] += im

            reg_file_chan2[j].write(bytearray(im))
            ix+=1
            if ix%(1000)==0:
                print('%d frames of binary, time %0.2f sec.'%(ix,time.time()-t0))
            gc.collect()
            j+=1
            j = j%nplanes

    # write ops files
    do_registration = ops['do_registration']
    do_nonrigid = ops1[0]['nonrigid']
    for ops in ops1:
        ops['Ly'],ops['Lx'] = Ly,Lx
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
