import datetime
import os
from natsort import natsorted 

import numpy as np
import scipy

from ..detection.stats import roi_stats
from .. import run_s2p

try:
    from pynwb import NWBFile
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage
    from pynwb.device import Device
    from pynwb.ophys import OpticalChannel
    from pynwb.ophys import TwoPhotonSeries
    from pynwb.ophys import ImageSegmentation
    from pynwb.ophys import RoiResponseSeries
    from pynwb.ophys import Fluorescence
    from pynwb import NWBHDF5IO
    NWB = True
except ModuleNotFoundError:
    NWB = False


def read_nwb(fpath):
    with NWBHDF5IO(fpath, 'r') as fio:
        nwbfile = fio.read()
        
        # ROIs
        try:
            rois = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation']['pixel_mask']
            multiplane = False
        except:
            rois = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation']['voxel_mask']
            multiplane = True
        stat = []
        for n in range(len(rois)):
            stat.append({'ypix': rois[n][:,0].astype('int'), 
                         'xpix': rois[n][:,1].astype('int'), 
                         'lam': rois[n][:,-1]})
            if multiplane:
                stat[-1]['iplane'] = int(rois[n][0,-2])
        ops = run_s2p.default_ops()
        if 'aspect' in ops:
            d0 = np.array([int(ops['aspect'] * 10), 10])
        else:
            d0 = ops['diameter']
            if isinstance(d0, int):
                d0 = [d0, d0]
        
        if multiplane:
            nplanes = np.max(np.array([stat[n]['iplane'] for n in range(len(stat))]))+1
        else:
            nplanes = 1
        stat = np.array(stat)

        # ops with backgrounds
        ops1 = []
        for iplane in range(nplanes):
            ops = run_s2p.default_ops()
            bg_strs = ['meanImg', 'Vcorr', 'max_proj', 'meanImg_chan2']
            ops['nchannels'] = 1
            for bstr in bg_strs:
                if bstr in nwbfile.processing['ophys']['Backgrounds_%d'%iplane].images:
                    ops[bstr] = np.array(nwbfile.processing['ophys']['Backgrounds_%d'%iplane][bstr].data)
                    if bstr=='meanImg_chan2':
                        ops['nchannels'] = 2
            ops['Ly'], ops['Lx'] = ops[bg_strs[0]].shape
            ops['yrange'] = [0, ops['Ly']]
            ops['xrange'] = [0, ops['Lx']]    
            ops['tau'] = 1.0
            ops['fs'] = nwbfile.acquisition['TwoPhotonSeries'].rate
            ops1.append(ops.copy())

        stat = roi_stats(stat, *d0, ops['Ly'], ops['Lx'])
    
        # fluorescence
        F = np.array(nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].data)
        Fneu = np.array(nwbfile.processing['ophys']['Neuropil']['Neuropil'].data)
        spks = np.array(nwbfile.processing['ophys']['Deconvolved']['Deconvolved'].data)
        dF = F - ops['neucoeff'] * Fneu
        for n in range(len(stat)):
            stat[n]['skew'] = scipy.stats.skew(dF[n])

        # cell probabilities
        iscell = [nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][n] 
                for n in range(len(stat))]
        iscell = np.array(iscell)
        probcell = iscell[:,1]
        iscell = iscell[:,0].astype(np.bool)
        redcell = np.zeros_like(iscell)
        probredcell = np.zeros_like(probcell)
                
        if multiplane:
            ops = ops1[0].copy()
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
            max_proj = np.zeros((LY, LX))
            if ops['nchannels']>1:
                meanImg_chan2 = np.zeros((LY, LX))

            Vcorr = np.zeros((LY, LX))
            for k,ops in enumerate(ops1):
                xrange = np.arange(ops['dx'],ops['dx']+ops['Lx'])
                yrange = np.arange(ops['dy'],ops['dy']+ops['Ly'])
                meanImg[np.ix_(yrange, xrange)] = ops['meanImg']
                Vcorr[np.ix_(yrange, xrange)] = ops['Vcorr']
                max_proj[np.ix_(yrange, xrange)] = ops['max_proj']
                if ops['nchannels']>1:
                    if 'meanImg_chan2' in ops:
                        meanImg_chan2[np.ix_(yrange, xrange)] = ops['meanImg_chan2']            
                for j in np.nonzero(np.array([stat[n]['iplane']==k for n in range(len(stat))]))[0]:
                    stat[j]['xpix'] += ops['dx']
                    stat[j]['ypix'] += ops['dy']
                    stat[j]['med'][0] += ops['dy']
                    stat[j]['med'][1] += ops['dx']
            ops['Vcorr'] = Vcorr
            ops['max_proj'] = max_proj
            ops['meanImg'] = meanImg
            if 'meanImg_chan2' in ops:
                ops['meanImg_chan2'] = meanImg_chan2
            ops['Ly'], ops['Lx'] = LY, LX
            ops['yrange'] = [0, LY]
            ops['xrange'] = [0, LX]
    return stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell


def save_nwb(save_folder):
    """ convert folder with plane folders to NWB format """

    plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5]=='plane'])
    ops1 = [np.load(os.path.join(f, 'ops.npy'), allow_pickle=True).item() for f in plane_folders]

    if NWB and not ops1[0]['mesoscan']:
        if len(ops1)>1:
            multiplane = True
        else:
            multiplane = False

        ops = ops1[0]

        ### INITIALIZE NWB FILE
        nwbfile = NWBFile(
            session_description='suite2p_proc',
            identifier=str(ops['data_path'][0]),
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
            external_file=(ops['filelist'] if 'filelist' in ops else ['']), 
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
        ncells_all = 0
        for iplane, ops in enumerate(ops1):
            if iplane==0:
                iscell = np.load(os.path.join(ops['save_path'], 'iscell.npy'))
                for fstr in file_strs:
                    traces.append(np.load(os.path.join(ops['save_path'], fstr)))
            else:
                iscell = np.append(iscell, np.load(os.path.join(ops['save_path'], 'iscell.npy')), axis=0)
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
            ncells_all+=ncells

        ps.add_column('iscell', 'two columns - iscell & probcell', iscell)
            
        rt_region = ps.create_roi_table_region(
            region=list(np.arange(0, ncells_all)),
            description='all ROIs'
        )

        # FLUORESCENCE (all are required)
        file_strs = ['F.npy', 'Fneu.npy', 'spks.npy']
        name_strs = ['Fluorescence', 'Neuropil', 'Deconvolved']

        for i, (fstr,nstr) in enumerate(zip(file_strs, name_strs)):
            roi_resp_series = RoiResponseSeries(
                name=nstr,
                data=traces[i],
                rois=rt_region,
                unit='lumens',
                rate=ops['fs']
            )
            fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
            ophys_module.add(fl)

        # BACKGROUNDS
        # (meanImg, Vcorr and max_proj are REQUIRED)
        bg_strs = ['meanImg', 'Vcorr', 'max_proj', 'meanImg_chan2']
        nplanes = ops['nplanes']
        for iplane in range(nplanes):
            images = Images('Backgrounds_%d'%iplane)
            for bstr in bg_strs:
                if bstr in ops:
                    if bstr=='Vcorr' or bstr=='max_proj':
                        img = np.zeros((ops['Ly'], ops['Lx']), np.float32)
                        img[ops['yrange'][0]:ops['yrange'][-1], 
                            ops['xrange'][0]:ops['xrange'][-1]] = ops[bstr]
                    else:
                        img = ops[bstr]
                    images.add_image(GrayscaleImage(name=bstr, data=img))
                
            ophys_module.add(images)

        with NWBHDF5IO(os.path.join(save_folder, 'ophys.nwb'), 'w') as fio:
            fio.write(nwbfile)
    else:
        print('pip install pynwb OR don"t use mesoscope recording')
