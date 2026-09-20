import numpy as np
import sys
import os
from pathlib import Path
from suite2p import default_db, default_settings, parameters
from suite2p.run_s2p import logger_setup, run_s2p
from cellpose import utils
from scipy import stats
from suite2p.extraction import dcnv, extraction_wrapper
from suite2p.io import BinaryFile
import torch
from tqdm import trange
import h5py
from scipy.sparse import csr_matrix
from tqdm import trange

def interneuron_data(root='/media/carsen/disk2/test_suite2p/VGa14/green/'):
    """ data in figshare """
    root = Path(root)
    iplane = 1
    F0 = np.load(root / f'suite2p/plane{iplane}/F.npy')
    Fneu0 = np.load(root / f'suite2p/plane{iplane}/Fneu.npy')
    iscell0 = np.load(root / f'suite2p/plane{iplane}/iscell.npy')
    F0 -= 0.7 * Fneu0
    icell = iscell0[:,1]>0.05
    F0 = F0[icell]
    
    ops = np.load(root / f'suite2p/plane{iplane}/ops.npy', allow_pickle=True).item()

    mimg_chan2 = ops['meanImg_chan2']
    mimg = ops['meanImg']
    max_proj = ops['max_proj']
    Ly, Lx = mimg.shape
    mimg = np.zeros_like(mimg)
    mimg[ops['yrange'][0]:ops['yrange'][1], ops['xrange'][0]:ops['xrange'][1]] = max_proj.copy()
    redcell = np.load(root / f'suite2p/plane{iplane}/redcell.npy')[icell]
    stat = np.load(root / f'suite2p/plane{iplane}/stat.npy', allow_pickle=True)[icell]

    ired = (redcell[:,1]>0.6).nonzero()[0]
    dF = F0[ired]
    snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)
    dF = dF[snr.argsort()[::-1]]
    yrange, xrange = ops['yrange'], ops['xrange']
    masks = np.zeros((Ly, Lx), dtype='uint16')
    for i, s in enumerate(stat):
        masks[s['ypix'], s['xpix']] = i + 1

    outlines = utils.outlines_list(masks)

    np.save('results/interneurons.npy', {'outlines': outlines, 'mimg': mimg, 'mimg_chan2': mimg_chan2, 'ired': ired, 'yrange': yrange, 'xrange': xrange, 'dF': dF})


def voltage_data(root='/media/carsen/disk1/suite2p_paper/other_datasets/OPM_Voltage_Zebrafish_6dpf_SpinalCord_Fig2/'):
    """ data from https://zenodo.org/records/19235146 """
    
    root = Path(root)
    logger_setup()

    db = default_db()
    db['data_path'] = [str(root)]
    db['save_path0'] = str(root)
    settings = default_settings()
    settings['detection']['inverted_activity'] = True

    settings['detection']['algorithm'] = 'cellpose'
    settings['detection']['cellpose_settings']['cellpose_model'] = 'cpsam_v2' # default

    settings['diameter'] = 8
    settings['detection']['cellpose_settings']['cellprob_threshold'] = -6
    settings['detection']['cellpose_settings']['flow_threshold'] = 0
    settings['detection']['cellpose_settings']['img'] = 'meanImg'
    settings['detection']['cellpose_settings']['params'] = {'normalize': {'sharpen_radius': 5}}

    settings['registration']['block_size'] = [64, 64]

    run_s2p(settings=settings, db=db)

    db = np.load(root / 'suite2p/plane0/db.npy', allow_pickle=True).item()
    Ly, Lx = db['Ly'], db['Lx']
    stat = np.load(root / 'suite2p/plane0/stat.npy', allow_pickle=True)
    F = np.load(root / 'suite2p/plane0/F.npy')
    Fneu = np.load(root / 'suite2p/plane0/Fneu.npy')
    reg_outputs = np.load(root / 'suite2p/plane0/reg_outputs.npy', allow_pickle=True).item()
    mean_img = reg_outputs['meanImg']

    dF_s2p = F.copy() - Fneu
    npix = np.array([len(s['ypix']) for s in stat])
    icell = (npix < (6**2*np.pi)) 

    print(icell.sum(), len(stat))
    stat_s2p = stat[icell]
    dF_s2p = dF_s2p[icell]
    skew_s2p = stats.skew(dF_s2p[:,:5000], axis=1)

    dF_s2p = dF_s2p[skew_s2p.argsort()]

    masks_s2p = np.zeros((Ly, Lx), dtype='uint16')
    for i, s in enumerate(stat_s2p):
        masks_s2p[s['ypix'], s['xpix']] = i + 1
    outlines_s2p = utils.outlines_list(masks_s2p)

    np.save('results/voltage.npy', {'dF': dF_s2p, 'outlines': outlines_s2p, 'mean_img': mean_img})


def onep_data(root='/media/carsen/disk1/suite2p_paper/other_datasets/N9_relapse_1P/'):
    """ data from https://zenodo.org/records/18279500 """
    root = Path(root)
    logger_setup()

    settings = default_settings()
    db = default_db()
    db['data_path'] = [str(root)]
    db['fs'] = 30
    db['tau'] = 1.0
    db['input_format'] = 'movie'
    db['keep_movie_raw'] = True

    settings['detection']['algorithm'] = 'sparsery'
    settings['detection']['threshold_scaling'] = 0.75
    settings['detection']['sparsery_settings']['highpass_neuropil'] = 60 
    settings['detection']['highpass_time'] = 15 
    settings['detection']['max_overlap'] = 0.85
    settings['extraction']['neuropil_coefficient'] = 1.0

    settings['registration']['spatial_taper'] = 100
    settings['registration']['smooth_sigma'] = 3
    settings['registration']['block_size'] = (128, 128) 
    settings['registration']['maxregshiftNR'] = 10

    if not (root / 'suite2p/plane0/stat.npy').exists():
        run_s2p(settings=settings, db=db)

    db = np.load(root / 'suite2p/plane0/db.npy', allow_pickle=True).item()
    Ly, Lx = db['Ly'], db['Lx']
    stat = np.load(root / 'suite2p/plane0/stat.npy', allow_pickle=True)
    F = np.load(root / 'suite2p/plane0/F.npy')
    Fneu = np.load(root / 'suite2p/plane0/Fneu.npy')
    iscell = np.load(root / 'suite2p/plane0/iscell.npy')
    reg_outputs = np.load(root / 'suite2p/plane0/reg_outputs.npy', allow_pickle=True).item()
    mean_img = reg_outputs['meanImg']
    yrange, xrange = reg_outputs['yrange'], reg_outputs['xrange']

    dF_s2p = F.copy() - Fneu

    npix_norm = np.array([s['npix_norm_no_crop'] for s in stat])
    npix = np.array([len(s['ypix']) for s in stat])
    icell = (iscell[:, 1] > 0.5)  & (npix_norm > 0.5)

    print(icell.sum())
    stat_s2p = stat[icell]
    dF_s2p = dF_s2p[icell]

    skew_s2p = stats.skew(dF_s2p, axis=1)

    Ly, Lx = mean_img.shape

    masks_s2p = np.zeros((Ly, Lx), 'uint16')
    for i in range(len(stat_s2p)):
        masks_s2p[stat_s2p[i]['ypix'], stat_s2p[i]['xpix']] = i + 1

    outlines_s2p = utils.outlines_list(masks_s2p)

    settings = default_settings()
    dmean_s2p = np.zeros((Ly, Lx), 'float32')
    with BinaryFile(Ly=Ly, Lx=Lx, filename=root / 'suite2p/plane0/data.bin') as f_reg:
        # get frames
        batch_size = 500
        k = 0
        for i in trange(0, db['nframes']-batch_size, batch_size):
            mov = f_reg[i : i+batch_size].reshape(batch_size, -1).T
            dmov = dcnv.preprocess(mov.astype('float32'), baseline='maximin', 
                            win_baseline=int(settings['dcnv_preprocess']['win_baseline']), 
                            sig_baseline=int(settings['dcnv_preprocess']['sig_baseline']),
                            fs=30.0, device=torch.device('cuda'))
            dmean_s2p += dmov.mean(axis=1).reshape(Ly, Lx)
            k+=1
        dmean_s2p /= k

    ### Assuming cnmfE is run first using demo_pipeline_cnmfE.ipynb >>    
    
    mean_img_caiman = np.load(root / 'caiman_mean_img.npy')
    cnm = h5py.File(root / 'cnmfe_results.hdf5', 'r')
    A = cnm['estimates']['A']
    A = csr_matrix((A['data'][:], A['indices'][:], A['indptr'][:]), 
                    shape=A['shape'][:][::-1])

    stat_caiman = []
    ncells = len(A.indptr) - 1
    stat_caiman = []
    for i in trange(ncells):
        if i in cnm['estimates']['idx_components']:
            ipix = A.indices[A.indptr[i]:A.indptr[i+1]]
            lam = A.data[A.indptr[i]:A.indptr[i+1]]
            xpix, ypix = np.unravel_index(ipix, (Lx, Ly)) # transposed

            stat_caiman.append({'ypix': ypix, 'xpix': xpix, 'lam': lam,
                            "radius": (len(ypix) / np.pi) ** 0.5,
                            "med": np.array([int(np.median(ypix)), int(np.median(xpix))]),
                            "overlap": np.zeros(len(ypix), "bool"),
                            "npix": len(ypix),})

    stat_caiman = np.array(stat_caiman)

    device = torch.device('cuda')

    settings = default_settings()
    with BinaryFile(Ly=Ly, Lx=Lx, filename=root / 'suite2p/plane0/data.bin') as f_reg:
        F_caiman, Fneu_caiman, _, _ = extraction_wrapper(
                stat_caiman, f_reg, settings=settings['extraction'],
                device=device)

    dF_caiman = F_caiman.copy() - 1. * Fneu_caiman.copy()   

    fname = '/home/carsen/caiman_data/temp/memmap__d1_608_d2_608_d3_1_order_C_frames_53895.mmap'

    f_reg = np.memmap(fname, dtype='float32', mode='r', shape=(608, 608, 53895))

    dmean_caiman = np.zeros((Ly, Lx), 'float32')
    # get frames
    batch_size = 500
    k = 0
    for i in trange(0, db['nframes']-batch_size, batch_size):
        mov = f_reg[:, :, i : i+batch_size].transpose(1,0,2).reshape(-1, batch_size)
        dmov = dcnv.preprocess(mov.astype('float32'), baseline='maximin', 
                        win_baseline=int(settings['dcnv_preprocess']['win_baseline']), 
                        sig_baseline=int(settings['dcnv_preprocess']['sig_baseline']),
                        fs=30.0, device=torch.device('cuda'))
        dmean_caiman += dmov.mean(axis=1).reshape(Ly, Lx)
        k+=1
    dmean_caiman /= k

    skew_caiman = stats.skew(dF_caiman, axis=1)
    
    masks_caiman = np.zeros((Ly, Lx), 'uint16')
    for i in range(len(stat_caiman)):
        masks_caiman[stat_caiman[i]['ypix'], stat_caiman[i]['xpix']] = i + 1

    outlines_caiman = utils.outlines_list(masks_caiman)

    np.save('results/onep.npy', {'dmean_s2p': dmean_s2p, 'dmean_caiman': dmean_caiman, 
                                    'outlines_s2p': outlines_s2p, 'outlines_caiman': outlines_caiman,
                                    'dF_s2p': dF_s2p, 'dF_caiman': dF_caiman,
                                    'skew_s2p': skew_s2p, 'skew_caiman': skew_caiman})