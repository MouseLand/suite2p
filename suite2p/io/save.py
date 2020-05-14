import numpy as np
import os
import scipy
import datetime
try:
    from pynwb import NWBFile
    from pynwb.device import Device
    from pynwb.ophys import OpticalChannel
    from pynwb.ophys import TwoPhotonSeries
    from pynwb.ophys import ImageSegmentation
    from pynwb.ophys import RoiResponseSeries
    from pynwb.ophys import Fluorescence
    from pynwb import NWBHDF5IO
    NWB = True
except:
    NWB = False


def combined(ops1):
    """ Combines all the entries in ops1 into a single result file.

    Multi-plane recordings are arranged to best tile a square.
    Multi-roi recordings are arranged by their dx,dy physical localization.
    """
    ops = ops1[0].copy()
    dx = np.array([o['dx'] for o in ops1])
    dy = np.array([o['dy'] for o in ops1])
    unq = np.unique(np.vstack((dy,dx)), axis=1)
    nrois = unq.shape[1]
    if ('dx' not in ops) or ('dy' not in ops):
        Lx = ops['Lx']
        Ly = ops['Ly']
        nX = np.ceil(np.sqrt(ops['Ly'] * ops['Lx'] * len(ops1))/ops['Lx'])
        nX = int(nX)
        nY = int(np.ceil(len(ops1)/nX))
        for j in range(len(ops1)):
            ops1[j]['dx'] = (j%nX) * Lx
            ops1[j]['dy'] = int(j/nX) * Ly
    elif nrois < len(ops1):
        nplanes = len(ops1) // nrois
        Lx = np.array([o['Lx'] for o in ops1])
        Ly = np.array([o['Ly'] for o in ops1])
        ymax = (dy+Ly).max()
        xmax = (dx+Lx).max()
        nX = np.ceil(np.sqrt(ymax * xmax * nplanes)/xmax)
        nX = int(nX)
        nY = int(np.ceil(len(ops1)/nX))
        for j in range(nplanes):
            for k in range(nrois):
                ops1[j*nrois + k]['dx'] += (j%nX) * xmax
                ops1[j*nrois + k]['dy'] += int(j/nX) * ymax

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
            if 'meanImg_chan2' in ops:
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
    if len(ops['save_folder']) > 0:
        fpath = os.path.join(ops['save_path0'], ops['save_folder'], 'combined')
    else:
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


def NWB_output(ops1):
    if NWB and not ops1[0]['mesoscan']:
        if len(ops1)>1:
            multiplane = True
        else:
            multiplane = False

        ops = ops1[0]
        ### INITIALIZE NWB FILE
        nwbfile = NWBFile(
            session_description='suite2p_proc',
            identifier=ops['data_path'][0],
            session_start_time=(ops['date_proc'] if 'date_proc' in ops 
                                else datetime.datetime.now())
        )
        print(nwbfile)


        device = nwbfile.create_device(
            name='Microscope', 
            description='My two-photon microscope',
            manufacturer='The best microscope manufacturer'
        )
        optical_channel = OpticalChannel(
            name='OpticalChannel', 
            description='an optical channel', 
            emission_lambda=500.
        )

        imaging_plane = nwbfile.create_imaging_plane(
            name='ImagingPlane',
            optical_channel=optical_channel,
            imaging_rate=ops['fs'],
            description='standard',
            device=device,
            excitation_lambda=600.,
            indicator='GCaMP',
            location='V1',
            grid_spacing=([2,2,30] if multiplane else [2,2]),
            grid_spacing_unit='microns'
        )

        # link to external data
        image_series = TwoPhotonSeries(
            name='TwoPhotonSeries', 
            dimension=[ops['Ly'], ops['Lx']],
            external_file=(ops['filelist'] if 'filelist' in ops else ''), 
            imaging_plane=imaging_plane,
            starting_frame=[0], 
            format='external', 
            starting_time=0.0, 
            rate=ops['fs'] * ops['nplanes']
        )
        nwbfile.add_acquisition(image_series)

        # processing
        img_seg = ImageSegmentation()
        ps = img_seg.create_plane_segmentation(
            name='PlaneSegmentation',
            description='suite2p output',
            imaging_plane=imaging_plane,
            reference_images=image_series
        )
        ophys_module = nwbfile.create_processing_module(
            name='ophys', 
            description='optical physiology processed data'
        )
        ophys_module.add(img_seg)

        
        file_strs = ['F.npy', 'Fneu.npy', 'spks.npy']
        traces = []
        k=0
        for iplane, ops in enumerate(ops1):
            if k==0:
                for fstr in file_strs:
                    traces.append(np.load(os.path.join(ops['save_path'], fstr)))
            else:
                for i,fstr in enumerate(file_strs):
                    traces[i] = np.append(traces[i], 
                                        np.load(os.path.join(ops['save_path'], fstr)), axis=0) 
            
            stat = np.load(os.path.join(ops['save_path'], 'stat.npy'), allow_pickle=True)
            ncells = len(stat)
            for n in range(ncells):
                if multiplane:
                    pixel_mask = np.array([stat[n]['ypix'], stat[n]['xpix'], 
                                        iplane*np.ones(stat[n]['npix']), 
                                        stat[n]['lam']])
                    ps.add_roi(voxel_mask=pixel_mask.T)
                else:
                    pixel_mask = np.array([stat[n]['ypix'], stat[n]['xpix'], 
                                        stat[n]['lam']])
                    ps.add_roi(pixel_mask=pixel_mask.T)


        rt_region = ps.create_roi_table_region(
            region=list(np.arange(0, ncells)),
            description='all ROIs'
        )

        file_strs = ['F.npy', 'Fneu.npy', 'spks.npy']
        name_strs = ['Fluorescence', 'Neuropil', 'Deconvolved']

        for fstr,nstr in zip(file_strs, name_strs):
            traces = np.load(os.path.join(ops['save_path'], fstr))
            roi_resp_series = RoiResponseSeries(
                name=nstr,
                data=traces,
                rois=rt_region,
                unit='lumens',
                rate=ops['fs']
            )
            fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
            ophys_module.add(fl)

        with NWBHDF5IO(os.path.join(ops['save_path0'], 'suite2p', 'ophys.nwb'), 'w') as fio:
            fio.write(nwbfile)