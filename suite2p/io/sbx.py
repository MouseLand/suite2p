import os

import numpy as np

from .utils import init_ops, find_files_open_binaries

try:
    from sbxreader import sbx_memmap
except:
    print('Could not load the sbx reader, installing with pip.')
    from subprocess import call
    call('pip install sbxreader',shell = True)
    from sbxreader import sbx_memmap



def sbx_to_binary(ops, ndeadcols=-1, ndeadrows=0):
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
    if 'sbx_ndeadrows' in ops1[0].keys():
        ndeadrows = int(ops1[0]['sbx_ndeadrows'])
    
    if ndeadcols==-1 or ndeadrows==-1:
        # compute dead rows and cols from the first file
        tmpsbx = sbx_memmap(sbxlist[0])
        # do not remove dead rows in non-multiplane mode
        # This number should be different for each plane since the artifact is larger
        # for larger ETL jumps. 
        if nplanes > 1 and ndeadrows==-1:
            colprofile = np.array(np.mean(tmpsbx[0][0][0], axis=1))
            ndeadrows = np.argmax(np.diff(colprofile)) + 1
        else:
            ndeadrows = 0
        # do not remove dead columns in unidirectional scanning mode
        # do this only if ndeadcols is -1 
        if tmpsbx.metadata['scanning_mode'] == 'bidirectional' and ndeadcols==-1:
            ndeadcols = tmpsbx.ndeadcols
        else:
            ndeadcols = 0
        del tmpsbx
        print('Removing {0} dead columns while loading sbx data.'.format(ndeadcols))
        print('Removing {0} dead rows while loading sbx data.'.format(ndeadrows))

    ops1[0]['sbx_ndeadcols'] = ndeadcols
    ops1[0]['sbx_ndeadrows'] = ndeadrows
    
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
            im = np.array(f[onset:offset,:,:,ndeadrows:,ndeadcols:])//2
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
