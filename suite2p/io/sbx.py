import os

import numpy as np

from .utils import init_ops, find_files_open_binaries


def sbx_get_info(sbxfile):
    ''' 
    Read info from a scanbox mat file [pass the sbx extension].
    info = sbx_get_info(sbxfile)
    '''
    matfile = os.path.splitext(sbxfile)[0] + '.mat'
    if not os.path.exists(matfile):
        raise FileNotFoundError('Metadata not found: {0}'.format(matfile))
    from scipy.io import loadmat
    info = loadmat(matfile,squeeze_me=True,struct_as_record=False)
    return info['info']

def sbx_get_shape(sbxfile):
    ''' 
    Get shape from scanbox file.
    Reads it from the file size and the info mat file.
    (chan,ncols,nrows,max_idx),nplanes = sbx_get_shape(sbxfile)
    '''
    info = sbx_get_info(sbxfile)
    fsize = os.path.getsize(sbxfile)
    nrows,ncols = info.sz
    chan = info.channels
    if chan == 1:
        chan = 2; 
    elif chan == 2:
        chan = 1
    elif chan == 3:
        chan = 1
    max_idx = fsize/nrows/ncols/chan/2
    if max_idx != info.config.frames:
        print('SBX filesize doesnt match accompaning MAT [{0},{1}]. Check recording.'.format(
            max_idx,
            info.config.frames))
    nplanes = 1
    if not isinstance(info.otwave,int):
        if len(info.otwave) and info.volscan:
            nplanes = len(info.otwave)
    # make sure that if there are multiple planes it works regardless of the number of recorded  frames
    max_idx = np.floor((max_idx/nplanes)) * nplanes
    return (int(chan),int(ncols),int(nrows),int(max_idx)),nplanes

def sbx_memmap(filename,plane_axis=True):
    '''
    Memory maps a scanbox file.

    npmap = sbx_memmap(filename,reshape_planes=True)
    Returns a N x 1 x NChannels x H x W memory map object; data can be accessed like a numpy array.
    Reshapes data to (N,nplanes,nchan,H,W) if plane_axis=1

    Actual data are 65535 - sbxmmap; data format is uint16
    '''
    if filename[-3:] == 'sbx':
        sbxshape,nplanes = sbx_get_shape(filename)
        if plane_axis:
            return np.memmap(filename,
                             dtype='uint16',
                             shape=sbxshape,order='F').transpose([3,0,2,1]).reshape(
                int(sbxshape[3]/nplanes),
                nplanes,
                sbxshape[0],
                sbxshape[2],
                sbxshape[1])
        else:
            return np.memmap(filename,
                             dtype='uint16',
                             shape=sbxshape,order='F').transpose([3,0,2,1]).reshape(
                int(sbxshape[3]),
                sbxshape[0],
                sbxshape[2],
                sbxshape[1])            
    else:
        raise ValueError('Not sbx:  ' + filename)


def sbx_to_binary(ops,ndeadcols = -1):
    """  finds scanbox files and writes them to binaries

    Parameters
    ----------
    ops : dictionary
        'nplanes', 'data_path', 'save_path', 'save_folder', 'fast_disk',
        'nchannels', 'keep_movie_raw', 'look_one_level_down'

    Returns
    -------
        ops : dictionary of first plane
            'Ly', 'Lx', ops['reg_file'] or ops['raw_file'] is created binary

    """

    ops1 = init_ops(ops)
    # the following should be taken from the metadata and not needed but the files are initialized before...
    nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']
    # open all binary files for writing
    ops1, sbxlist, reg_file, reg_file_chan2 = find_files_open_binaries(ops1)
    iall = 0
    for j in range(ops1[0]['nplanes']):
        ops1[j]['nframes_per_folder'] = np.zeros(len(sbxlist), np.int32)
    ik = 0
    if 'sbx_ndeadcols' in ops1[0].keys():
        ndeadcols = int(ops1[0]['sbx_ndeadcols'])
    if ndeadcols == -1:
        sbxinfo = sbx_get_info(sbxlist[0])
        if sbxinfo.scanmode == 1:
            # do not remove dead columns in unidirectional scanning mode
            ndeadcols = 0
        else:
            # compute dead cols from the first file
            tmpsbx = sbx_memmap(sbxlist[0])
            colprofile = np.mean(tmpsbx[0][0][0],axis = 0)
            ndeadcols = np.argmax(np.diff(colprofile)) + 1
            del tmpsbx
            print('Removing {0} dead columns while loading sbx data.'.format(ndeadcols))
    ops1[0]['sbx_ndeadcols'] = ndeadcols
    
    for ifile,sbxfname in enumerate(sbxlist):
        f = sbx_memmap(sbxfname)
        nplanes = f.shape[1]
        nchannels = f.shape[2]
        nframes = f.shape[0]
        iblocks = np.arange(0,nframes,ops1[0]['batch_size'])
        if iblocks[-1] < nframes:
            iblocks = np.append(iblocks,nframes)

        # data = nframes x nplanes x nchannels x pixels x pixels
        if nchannels>1:
            nfunc = ops1[0]['functional_chan'] - 1
        else:
            nfunc = 0
        # loop over all frames
        for ichunk,onset  in enumerate(iblocks[:-1]):
            offset = iblocks[ichunk+1]
            im = (np.uint16(65535)-f[onset:offset,:,:,:,ndeadcols:])//2
            im = im.astype(np.int16)
            im2mean = im.mean(axis = 0).astype(np.float32)/len(iblocks) 
            for ichan in range(nchannels):
                nframes = im.shape[0]
                im2write = im[:,:,ichan,:,:]
                for j in range(0,nplanes):
                    if iall==0:
                        ops1[j]['meanImg'] = np.zeros((im.shape[3],im.shape[4]),np.float32)
                        if nchannels>1:
                            ops1[j]['meanImg_chan2'] = np.zeros((im.shape[3],im.shape[4]),np.float32)
                        ops1[j]['nframes'] = 0
                    if ichan == nfunc:
                        ops1[j]['meanImg'] += np.squeeze(im2mean[j,ichan,:,:])
                        reg_file[j].write(bytearray(im2write[:,j,:,:].astype('int16')))
                    else:
                        ops1[j]['meanImg_chan2'] += np.squeeze(im2mean[j,ichan,:,:])
                        reg_file_chan2[j].write(bytearray(im2write[:,j,:,:].astype('int16')))
                        
                    ops1[j]['nframes'] += im2write.shape[0]
                    ops1[j]['nframes_per_folder'][ifile] += im2write.shape[0]
            ik += nframes
            iall += nframes

    # write ops files
    do_registration = ops1[0]['do_registration']
    do_nonrigid = ops1[0]['nonrigid']
    for ops in ops1:
        ops['Ly'] = im.shape[3]
        ops['Lx'] = im.shape[4]
        if not do_registration:
            ops['yrange'] = np.array([0,ops['Ly']])
            ops['xrange'] = np.array([0,ops['Lx']])
        #ops['meanImg'] /= ops['nframes']
        #if nchannels>1:
        #    ops['meanImg_chan2'] /= ops['nframes']
        np.save(ops['ops_path'], ops)
    # close all binary files and write ops files
    for j in range(0,nplanes):
        reg_file[j].close()
        if nchannels>1:
            reg_file_chan2[j].close()
    return ops1[0]
