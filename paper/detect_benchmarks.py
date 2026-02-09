import tifffile
from pathlib import Path
import numpy as np
from suite2p.detection import bin_movie, utils
from suite2p import default_settings
from suite2p import extraction, default_settings 
from suite2p.io import BinaryFile
from suite2p.run_s2p import logger_setup 
import torch 
import torch.nn.functional as F
from tqdm import trange 
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.utils import outlines_list
from scipy.stats import zscore
from detect_f1_score import detect_f1_score


def make_ellipses(Ly, Lx, n=1000):
    ly = 16
    a = np.hstack((np.random.uniform(0.75, 2, size=n - n//3),
                   np.random.uniform(0.75, 2, size=n//3)))
    b = np.hstack((np.random.uniform(0.75, 2, size=n - n//3),
                   np.random.uniform(1, 6, size=n//3)))
    x0 = np.random.uniform(a, ly - a, size=n)
    y0 = np.random.uniform(b, ly - b, size=n)
    theta = np.random.uniform(0, 2 * np.pi, size=n)
    y, x = np.meshgrid(np.arange(ly), np.arange(ly), indexing='ij')
    Y = ((y - y0[:, np.newaxis, np.newaxis]) * np.cos(theta[:, np.newaxis, np.newaxis]) + 
         (x - x0[:, np.newaxis, np.newaxis]) * np.sin(theta[:, np.newaxis, np.newaxis]))
    X = ((y - y0[:, np.newaxis, np.newaxis]) * -np.sin(theta[:, np.newaxis, np.newaxis]) +
         (x - x0[:, np.newaxis, np.newaxis]) * np.cos(theta[:, np.newaxis, np.newaxis]))
    ell = (Y**2 / a[:, np.newaxis, np.newaxis]**2 + X**2 / b[:, np.newaxis, np.newaxis]**2) 
    
    ell += np.random.randn(*ell.shape) * 0.25
    ell = np.maximum(0, ell)
    ell0 = (1-ell).copy()
    ell = ell <= 1.0
    
    y0 = np.random.randint(0, Ly - ly, size=n)
    x0 = np.random.randint(0, Lx - ly, size=n)

    ipix = [np.nonzero(ell[i]) for i in range(n)]
    lam = [np.maximum(0.1, ell0[i][yp, xp]) for i, (yp, xp) in enumerate(ipix)]
    lam = [l / l.sum() for l in lam]
    ypix = [ip[0] + y0[i] for i, ip in enumerate(ipix)]
    xpix = [ip[1] + x0[i] for i, ip in enumerate(ipix)]
        
    return ypix, xpix, lam


def make_dendrites(root, n_planes=4, iplane=1, ds=2, n_ell=2000, snr_threshold=0.3):
    # make random ellipses 
    db = np.load(root / 'suite2p_ds' / f'plane{iplane}' / 'db.npy', allow_pickle=True).item()
    reg_outputs = np.load(root / 'suite2p_ds' / f'plane{iplane}' / 'reg_outputs.npy', allow_pickle=True).item()
    Ly, Lx = db['Ly'], db['Lx']
    yrange, xrange = reg_outputs['yrange'], reg_outputs['xrange']
    Ly0, Lx0 = yrange[1] - yrange[0], xrange[1] - xrange[0]

    ypix_ell, xpix_ell, lam_ell = make_ellipses(Ly0, Lx0, n=n_ell)
    stat_ell = []
    for ypix, xpix, lam in zip(ypix_ell, xpix_ell, lam_ell):
        stat_ell.append({'ypix': ypix + yrange[0],
                         'xpix': xpix + xrange[0],
                         'lam': lam,
                         'radius': (len(ypix) / np.pi) ** 0.5,
                         'med': np.array([int(np.median(ypix)), int(np.median(xpix))]),
                         'overlap': np.zeros(len(ypix), 'bool'),
                         'npix': len(ypix),})

    # get activity from ground-truth ROIs from other planes
    i0 = 0
    for ipl in range(n_planes):
        if ipl == iplane:
            continue
        F_gt0 = np.load(root / 'benchmarks' / f'F_gt_filt_plane{ipl}.npy')
        F_neu0 = np.load(root / 'benchmarks' / f'Fneu_gt_filt_plane{ipl}.npy')
        F_gt0 -= 0.7 * F_neu0
        stat_gt0 = np.load(root / 'benchmarks' / f'stat_gt_plane{ipl}.npy', allow_pickle=True)
        if i0 == 0 :
            stat_gt_2 = stat_gt0.copy()
            F_gt_2 = F_gt0.copy()
        else:
            F_gt_2 = np.vstack((F_gt_2[:, :F_gt0.shape[1]], F_gt0))
            stat_gt_2 = np.hstack((stat_gt_2, stat_gt0))
        i0 += 1

    # filter by SNR
    snr_gt_2 = 1 - 0.5 * np.diff(F_gt_2, axis=1).var(axis=1) / F_gt_2.var(axis=1)
    F_gt_2 = F_gt_2[snr_gt_2 > snr_threshold, :]
    stat_gt_2 = stat_gt_2[snr_gt_2 > snr_threshold] 

    # get activity from nearby ROIs on other planes
    meds = np.array([s['med'] for s in stat_gt_2])
    meds_ell = np.array([s['med'] for s in stat_ell])
    dists = ((meds[:, np.newaxis, :] - meds_ell[np.newaxis, :, :]) ** 2).sum(axis=-1) ** 0.5 
    dists_sort = dists.argsort(axis=0)

    # sample ROIs
    isample = np.random.exponential(5, size=len(stat_ell)).astype('int')
    isample = dists_sort[isample, np.arange(dists_sort.shape[1])]
    F_ell = F_gt_2[isample, :].copy()

    return stat_ell, F_ell


def make_neuropil(root, n_planes=4, iplane=1, device=torch.device('cuda')):
    db = np.load(root / 'suite2p_ds' / f'plane0' / 'db.npy', allow_pickle=True).item()
    Ly, Lx = db['Ly'], db['Lx']

    ntilesY, ntilesX = 7, 7
    ys = np.arange(0, Ly)
    xs = np.arange(0, Lx)

    Kx = np.ones((Lx, ntilesX), 'float32')
    Ky = np.ones((Ly, ntilesY), 'float32')
    # basis functions are fourier modes
    for k in range(int((ntilesX - 1) / 2)):
        Kx[:, 2 * k + 1] = np.sin(2 * np.pi * (xs + 0.5) * (1 + k) / Lx)
        Kx[:, 2 * k + 2] = np.cos(2 * np.pi * (xs + 0.5) * (1 + k) / Lx)
    for k in range(int((ntilesY - 1) / 2)):
        Ky[:, 2 * k + 1] = np.sin(2 * np.pi * (ys + 0.5) * (1 + k) / Ly)
        Ky[:, 2 * k + 2] = np.cos(2 * np.pi * (ys + 0.5) * (1 + k) / Ly)

    S = np.zeros((ntilesY, ntilesX, Ly, Lx), np.float32)
    for kx in range(ntilesX):
        for ky in range(ntilesY):
            S[ky, kx, :, :] = np.outer(Ky[:, ky], Kx[:, kx])

    ky, kx = np.meshgrid(np.arange(ntilesY), np.arange(ntilesX), indexing='ij')
    S = S.reshape(ntilesY * ntilesX, Ly, Lx)
    ky, kx = ky.flatten(), kx.flatten()
    S, ky, kx = S[1:], ky[1:], kx[1:] # remove DC component

    S /= (S**2).sum(axis=(-1,-2), keepdims=True)**0.5
    #S /= S.sum(axis=(-1, -2), keepdims=True)

    
    # compute 'neuropil' as projection of data onto basis functions
    n_frames = db['nframes']
    Sr = S.reshape(S.shape[0], -1)
    Sr = torch.from_numpy(Sr).to(device).float()
    Fneu_sim = np.zeros((Sr.shape[0], n_frames), 'float32')

    # gaussian = torch.arange(-50, 51, device=device).float()
    # gaussian = torch.exp(-gaussian**2 / (2 * 3**2))
    # gaussian /= gaussian.sum()
    # gaussian = gaussian.unsqueeze(0).unsqueeze(0)
    for ipl in range(n_planes):
        if ipl == iplane:
            continue
        with BinaryFile(Ly=Ly, Lx=Lx, 
                        filename=root / 'suite2p_ds' / f'plane{ipl}' / 'data_filt.bin') as f_reg:
            batch_size = 2000 
            n_batches = int(np.ceil(n_frames / batch_size))
            n_frames = f_reg.shape[0]
            for n in trange(n_batches, mininterval=10):
                tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
                data = torch.from_numpy(f_reg[tstart : tend]).to(device).float()
                data = data.reshape(-1, Ly*Lx)
                # data_smooth = F.conv1d(data.T.unsqueeze(1), gaussian, 
                #                 padding=gaussian.shape[-1]//2)
                # data = data_smooth.squeeze().T
                if n==0:
                    dmean = data.mean()
                    #dstd = data.std(axis=0).mean()
                    #print(dstd.shape)
                data -= dmean # trying this out
                Fneu0 = (Sr @ data.T).cpu().numpy() # + dmean.mean()

                Fneu_sim[:, tstart : tend] += Fneu0 

    return S, Fneu_sim


def hybrid_gt(root, n_planes=4, iplane=1, ds=2, n_ell=2000, neu_coeff=5,
                poisson_coeff=50, test=False, device=torch.device('cuda')):

    if n_ell > 0:
        stat_ell, F_ell = make_dendrites(root, n_planes=n_planes, iplane=iplane, ds=ds, n_ell=n_ell)

    if neu_coeff > 0:
        S, Fneu_sim = make_neuropil(root, n_planes=n_planes, iplane=iplane)
        Sr = S.reshape(S.shape[0], -1)
        Sr = torch.from_numpy(Sr).to(device).float()

    db = np.load(root / 'suite2p_ds' / f'plane{iplane}' / 'db.npy', allow_pickle=True).item()
    Ly, Lx = db['Ly'], db['Lx']
    n_frames = db['nframes']
    (root / 'sims').mkdir(parents=True, exist_ok=True)
    with BinaryFile(Ly=Ly, Lx=Lx, n_frames=n_frames, dtype='int16',
                    filename=root / 'sims' / f'data_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}.bin',
                    write=True) as f_out:
        with BinaryFile(Ly=Ly, Lx=Lx, 
                    filename=root / 'suite2p_ds' / f'plane{iplane}' / 'data.bin',
                    ) as f_reg:
            batch_size = 2000 
            n_batches = int(np.ceil(n_frames / batch_size))
            for n in trange(n_batches, mininterval=10):   
                tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
                data = np.array(f_reg[tstart : tend])

                if n_ell > 0:
                    fr_ell = np.zeros(data.shape, 'float32')
                    for i in range(len(stat_ell)):
                        ypix, xpix, lam = stat_ell[i]['ypix'], stat_ell[i]['xpix'], stat_ell[i]['lam']
                        F_ell0 = F_ell[i, tstart : tend, np.newaxis] * (lam / (lam).sum()) * len(ypix)
                        fr_ell[:, ypix, xpix] += F_ell0

                    fr_ell = torch.from_numpy(fr_ell).to(device) 
                    fr_ell = torch.clamp(fr_ell, 0)

                if n == 0:
                    dmean = data.mean() 
                
                if neu_coeff > 0:
                    Fsim = torch.from_numpy(Fneu_sim).to(device).float()
                    dadd = Fsim[:, tstart : tend].T @ Sr 
                    dadd = dadd.reshape(-1, Ly, Lx)
                    # dadd = torch.clamp(dadd, 0) 
                    # if n == 0:
                    #     nmean = dadd.mean(axis=0)
                    # dadd -= 0.75 * nmean
                    dadd += dmean
                    dadd = torch.clamp(dadd, 0)    
                    #dadd = torch.poisson(dadd / 10) * 10

                d_out = torch.from_numpy(data).to(device).float()
                
                if neu_coeff > 0:
                    d_out *= 1 - neu_coeff
                    d_out += neu_coeff * dadd
                
                if n_ell > 0:
                    d_out += fr_ell * (1 - neu_coeff)

                if poisson_coeff > 0:
                    d_out = torch.clamp(d_out, 0)   
                    d_out = torch.poisson(d_out / poisson_coeff) * poisson_coeff
                
                d_out = torch.clamp(d_out, 0, 65534//2).int().cpu().numpy()

                f_out[tstart : tend] = d_out

                if test:
                    break
                
    return d_out


def downsample_movie(root, ipl=1, ds=2, do_filt=True):
    (root / 'suite2p_ds' / f'plane{ipl}').mkdir(parents=True, exist_ok=True)    
    
    settings = np.load(root / 'suite2p' / f'plane{ipl}' / 'settings.npy', allow_pickle=True).item()
    np.save(root / 'suite2p_ds' / f'plane{ipl}' / 'settings.npy', settings)

    db = np.load(root / 'suite2p' / f'plane{ipl}' / 'db.npy', allow_pickle=True).item()
    reg_outputs = np.load(root / 'suite2p' / f'plane{ipl}' / 'reg_outputs.npy', allow_pickle=True).item()
    
    Ly, Lx = db['Ly'], db['Lx']
    n_frames = db['nframes']
    db['save_path'] = str(root / 'suite2p_ds' / f'plane{ipl}')
    db['fast_disk'] = str(root / 'suite2p_ds' / f'plane{ipl}')
    db['reg_file'] = str(root / 'suite2p_ds' / f'plane{ipl}' / 'data.bin')
    yrange, xrange = reg_outputs['yrange'], reg_outputs['xrange']
    yrange = [yrange[0] // ds, yrange[1] // ds]
    xrange = [xrange[0] // ds, xrange[1] // ds]
    db['Ly'], db['Lx'] = Ly // ds, Lx // ds
    reg_outputs['yrange'] = yrange
    reg_outputs['xrange'] = xrange
    
    print(db['Ly'], db['Lx'], n_frames, yrange, xrange)
    np.save(root / 'suite2p_ds' / f'plane{ipl}' / 'db.npy', db)
    np.save(root / 'suite2p_ds' / f'plane{ipl}' / 'reg_outputs.npy', reg_outputs, allow_pickle=True)
        
    with BinaryFile(Ly=Ly//ds, Lx=Lx//ds, n_frames=n_frames, dtype='int16',
                    filename=root / 'suite2p_ds' / f'plane{ipl}' / 'data.bin',
                    write=True) as f_out:
        with BinaryFile(Ly=Ly, Lx=Lx, 
                        filename=root / 'suite2p' / f'plane{ipl}' / 'data.bin') as f_reg:
            batch_size = 500 
            n_batches = int(np.ceil(n_frames / batch_size))
            for n in trange(n_batches, mininterval=10):
                tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
                # data = f_reg[tstart : tend].reshape(-1, Ly//2, 2, Lx//2, 2).mean(axis=(2, 4))
                data = f_reg[tstart : tend, ::ds, ::ds]
                f_out[tstart : tend] = data

    if do_filt:
        with BinaryFile(Ly=Ly//ds, Lx=Lx//ds, n_frames=n_frames, dtype='int16',
                    filename=root / 'suite2p_ds' / f'plane{ipl}' / 'data_filt.bin',
                    write=True) as f_out:
            with BinaryFile(Ly=Ly, Lx=Lx, 
                            filename=root / 'suite2p' / f'plane{ipl}' / 'data_filt.bin') as f_reg:
                batch_size = 500 
                n_batches = int(np.ceil(n_frames / batch_size))
                for n in trange(n_batches, mininterval=10):
                    tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
                    #data = f_reg[tstart : tend].reshape(-1, Ly//2, 2, Lx//2, 2).mean(axis=(2, 4))
                    data = f_reg[tstart : tend, ::ds, ::ds]
                    f_out[tstart : tend] = data
    


def baseline_movie(root, ipl=0, device=torch.device('cuda')):
    db = np.load(root / 'suite2p' / f'plane{ipl}' / 'db.npy', allow_pickle=True).item()
    settings = np.load(root / 'suite2p' / f'plane{ipl}' / 'settings.npy', allow_pickle=True).item()
    reg_outputs = np.load(root / 'suite2p' / f'plane{ipl}' / 'reg_outputs.npy', allow_pickle=True).item()
    yrange = reg_outputs['yrange']
    xrange = reg_outputs['xrange']
    Ly, Lx = db['Ly'], db['Lx']
    n_frames = db['nframes']

    bin_size = int(np.round(settings['tau'] * settings['fs']))
    ks = (int(np.round(settings['fs']))//2) * 2 + 1

    # save baselined data
    with BinaryFile(Ly=Ly, Lx=Lx, n_frames=n_frames, dtype='int16',
                    filename=root / 'suite2p' / f'plane{ipl}' / 'data_filt.bin',
                    write=True) as f_out:
        with BinaryFile(Ly=Ly, Lx=Lx, 
                        filename=root / 'suite2p' / f'plane{ipl}' / 'data.bin') as f_reg:
            batch_size = 500 
            n_batches = int(np.ceil(n_frames / batch_size))
            print(f_out.shape, f_reg.shape)
            for n in trange(n_batches, mininterval=10):
                tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
                data = torch.from_numpy(f_reg[tstart : tend]).to(device).float()
                dfilt = data.permute(1, 2, 0).reshape(1, Ly*Lx, -1).clone()
                dfilt = -1 * F.max_pool1d(-dfilt, kernel_size=ks, stride=1, padding=ks//2)
                dfilt = F.max_pool1d(dfilt, kernel_size=ks, stride=1, padding=ks//2)
                data -= dfilt.squeeze().T.reshape(-1, Ly, Lx)
                f_out[tstart : tend] = data.int().cpu().numpy()


def cellpose_gt(root, n_planes=4):
    for ipl in range(4):
        db = np.load(root / 'suite2p' / f'plane{ipl}' / 'db.npy', allow_pickle=True).item()
        Ly, Lx = db['Ly'], db['Lx']
        reg_outputs = np.load(root / 'suite2p' / f'plane{ipl}' / 'reg_outputs.npy', allow_pickle=True).item()
        yrange, xrange = reg_outputs['yrange'], reg_outputs['xrange']
        detect_outputs = np.load(root / 'suite2p' / f'plane{ipl}' / 'detect_outputs.npy', allow_pickle=True).item()
        max_proj0 = detect_outputs['max_proj']
        max_proj = np.zeros((Ly, Lx), dtype='float32')
        max_proj[yrange[0] : yrange[1], xrange[0] : xrange[1]] = max_proj0
        
        img = max_proj.copy()

        masks, stat_gt, F_gt, Fneu_gt = cellpose_extract(img, Ly, Lx, 
                                                        [root / 'suite2p' / f'plane{ipl}' / 'data.bin', 
                                                        root / 'suite2p' / f'plane{ipl}' / 'data_filt.bin'])
        print(masks.max())

        (root / 'benchmarks').mkdir(exist_ok=True)
        np.save(root / 'benchmarks' / f'stat_gt_plane{ipl}.npy', stat_gt)
        np.save(root / 'benchmarks' / f'F_gt_plane{ipl}.npy', F_gt[0])
        np.save(root / 'benchmarks' / f'F_gt_filt_plane{ipl}.npy', F_gt[1])
        np.save(root / 'benchmarks' / f'Fneu_gt_plane{ipl}.npy', Fneu_gt[0])
        np.save(root / 'benchmarks' / f'Fneu_gt_filt_plane{ipl}.npy', Fneu_gt[1])
        np.save(root / 'benchmarks' / f'masks_gt_plane{ipl}.npy', masks)
        np.save(root / 'benchmarks' / f'img_gt_plane{ipl}.npy', img)


def cellpose_extract(img, Ly, Lx, bin_files, ds=2):
    model = models.CellposeModel(gpu=True) 
    #masks = model.eval(img, diameter=12, flow_threshold=0., cellprob_threshold=-3, 
    #                   normalize={'tile_norm_blocksize':75})[0]
    masks = model.eval(img, diameter=16, flow_threshold=0., 
                    cellprob_threshold=-3, 
                    normalize=True)[0]
    #masks = model.eval(img, diameter=18, cellprob_threshold=-2, flow_threshold=0)[0]

    settings = default_settings() 
    settings['extraction']['allow_overlap'] = True

    device = torch.device('cuda')

    ncells = masks.max()
    stat_cp = []
    pix = [np.nonzero(masks==(i+1)) for i in trange(ncells)]
    for i in trange(ncells):
        ypix, xpix = pix[i]
        lam = img[ypix, xpix]
        if ds > 1:
            ipix = (ypix // ds) * (Lx // ds) + xpix // ds
            ipix, idx = np.unique(ipix, return_index=True)
            lam = lam[idx]
            ypix, xpix = np.unravel_index(ipix, (Ly//2, Lx//2))
        
        stat_cp.append({'ypix': ypix, 'xpix': xpix, 'lam': lam,
                        'radius': (len(ypix) / np.pi) ** 0.5 / 2,
                        'med': np.array([int(np.median(ypix)), int(np.median(xpix))]),
                        'overlap': np.zeros(len(ypix), 'bool'),
                        'npix': len(ypix),})
        
    stat_cp = np.array(stat_cp)

    F_cp, Fneu_cp = [], []
    
    for bin_file in bin_files:
        with BinaryFile(Ly=Ly//ds, Lx=Lx//ds, filename=bin_file) as f_reg:
            F_cp0, Fneu_cp0, _, _ = extraction.extraction_wrapper(
                    stat_cp, f_reg, settings=settings['extraction'],
                    device=device)
        F_cp.append(F_cp0)
        Fneu_cp.append(Fneu_cp0)
        
    return masks, stat_cp, F_cp, Fneu_cp

    
def create_gt(root=Path('/media/carsen/ssd2/suite2p_paper/VG1/3/')):

    db = np.load(root / 'suite2p' / 'plane0' / 'db.npy', allow_pickle=True).item()
    reg_outputs = np.load(root / 'suite2p' / 'plane0' / 'reg_outputs.npy', allow_pickle=True).item()
    detect_outputs = np.load(root / 'suite2p' / 'plane0' / 'detect_outputs.npy', allow_pickle=True).item()
    max_proj = detect_outputs['max_proj']
    yrange = reg_outputs['yrange']
    xrange = reg_outputs['xrange']
    Ly, Lx = db['Ly'], db['Lx']

    img = max_proj.copy()
    # add mean image?
    img += reg_outputs['meanImg'][yrange[0]:yrange[1], xrange[0]:xrange[1]].astype('float32')
    img_gt = np.zeros((Ly, Lx), 'float32')
    img_gt = img[yrange[0]:yrange[1], xrange[0]:xrange[1]]

    masks, stat_gt, F_gt, Fneu_gt = cellpose_extract(img, Ly, Lx, yrange, xrange,
                                                     root / 'suite2p' / 'plane0' / 'data.bin')

    outlines = outlines_list(masks)

    plt.figure(figsize=(10, 10))
    plt.imshow(img, vmin=0, vmax=2000, cmap='gray')
    plt.title(f'Cellpose outlines, {len(outlines)} cells detected')
    for outline in outlines:
        plt.plot(outline[:, 0], outline[:, 1], color='red', linewidth=2)
    plt.show()

    dF_gt = F_gt.copy() - 0.7 * Fneu_gt
    snr_gt = 1 - 0.5 * np.diff(dF_gt, axis=1).var(axis=1) / dF_gt.var(axis=1)
    
    for i in range(len(stat_gt)):
        stat_gt[i]['snr'] = snr_gt[i]

    (root / 'benchmarks').mkdir(parents=True, exist_ok=True)
    np.save(root / 'benchmarks' / 'stat_gt.npy', stat_gt)
    np.save(root / 'benchmarks' / 'F_gt.npy', F_gt)
    np.save(root / 'benchmarks' / 'Fneu_gt.npy', Fneu_gt)
    np.save(root / 'benchmarks' / 'masks_gt.npy', masks)
    np.save(root / 'benchmarks' / 'img_gt.npy', img_gt)

    return masks, stat_gt, F_gt, Fneu_gt, img_gt


from suite2p import detection, extraction
from suite2p.run_s2p import logger_setup 

def hybrid_gt_run(root, n_planes=4, iplane=1, ds=2, 
                  n_ell=2000, neu_coeff=0.4, poisson_coeff=20,
                  threshold_scaling=0.7):

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    device = torch.device('cuda')
    (root / 'sims').mkdir(parents=True, exist_ok=True)
    if neu_coeff > 0 or n_ell > 0 or poisson_coeff > 0:
        reg_file = root / 'sims' / f'data_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}.bin'
    else:
        reg_file = root / 'suite2p_ds' / f'plane{iplane}' / 'data.bin'

    if neu_coeff > 0 or n_ell > 0 or poisson_coeff > 0:
        if not reg_file.exists():
            print(f'creating hybrid ground-truth with n_ell={n_ell}, neu_coeff={neu_coeff:.2f}, poisson_coeff={poisson_coeff}')
            hybrid_gt(root, n_planes=n_planes, iplane=iplane, ds=ds,
                        n_ell=n_ell, neu_coeff=neu_coeff, poisson_coeff=poisson_coeff, 
                        test=False, device=device)
        
    print('running suite2p detection and extraction')
    logger_setup(root)

    db = np.load(root / 'suite2p_ds' / f'plane{iplane}' / 'db.npy', allow_pickle=True).item()
    Ly, Lx = db['Ly'], db['Lx']
    settings = np.load(root / 'suite2p_ds' / f'plane{iplane}' / 'settings.npy', allow_pickle=True).item()
    reg_outputs = np.load(root / 'suite2p_ds' / f'plane{iplane}' / 'reg_outputs.npy', allow_pickle=True).item()

    settings['detection']['threshold_scaling'] = threshold_scaling
    settings['detection']['sparsery_settings']['max_ROIs'] = 10000
    settings['detection']['sparsery_settings']['highpass_neuropil'] = 25
    settings['tau'] = 0.25

    # data_bin = root / 'sims' / f'data_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}.bin'
    

    with BinaryFile(Ly=Ly, Lx=Lx, 
                        filename=reg_file) as f_reg:
        detect_outputs, stat, _ = detection.detection_wrapper(f_reg, diameter=settings['diameter'],
                                    tau=settings['tau'], fs=settings['fs'], 
                                    settings=settings['detection'],
                                    yrange=reg_outputs['yrange'],
                                    xrange=reg_outputs['xrange'],)
    
        F, Fneu, _, _ = extraction.extraction_wrapper(stat, f_reg, settings=settings['extraction'],
                                                      device=device)
    dF = F.copy() - 0.7 * Fneu

    th = threshold_scaling
    np.save(root / 'sims' / f'stat_s2p_{th:.1f}_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}.npy', stat)
    np.save(root / 'sims' / f'F_s2p_{th:.1f}_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}.npy', F)
    np.save(root / 'sims' / f'Fneu_s2p_{th:.1f}_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}.npy', Fneu)
    
    stat_gt = np.load(root / 'benchmarks' / f'stat_gt_plane{iplane}.npy', allow_pickle=True)
    F_gt = np.load(root / 'benchmarks' / f'F_gt_plane{iplane}.npy', allow_pickle=True)
    Fneu_gt = np.load(root / 'benchmarks' / f'Fneu_gt_plane{iplane}.npy', allow_pickle=True)
    dF_gt = F_gt.copy() - 0.7 * Fneu_gt
    
    tp, fp, fn, f1 = detect_f1_score(dF, dF_gt, stat, stat_gt, Ly=Ly, Lx=Lx, snr_threshold=0.25)

    np.save(root / 'sims' / f'results_s2p_{th:.1f}_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}.npy',
            np.array([tp, fp, fn, f1]))


import argparse
import os

if __name__ == '__main__':
    # argparse 
    arg_parser = argparse.ArgumentParser(description='Run hybrid ground-truth generation and Suite2p detection/extraction.')
    arg_parser.add_argument('--root', type=str, default='')
    arg_parser.add_argument('--sweep', action='store_true',
                            help='Run a sweep of hybrid ground-truth generation with different parameters.')
    arg_parser.add_argument('--param_sweep', action='store_true',
                            help='Run a parameter sweep over threshold scaling for the middle sim.')
    arg_parser.add_argument('--threshold_scaling', type=float, default=0.7,
                            help='threshold_scaling param for ROI detection in suite2p.')
    arg_parser.add_argument('--n_ell', type=int, default=2000,
                            help='Number of dendritic ellipses to generate.')
    arg_parser.add_argument('--neu_coeff', type=float, default=0.4,
                            help='Coefficient for neuropil contribution.')
    arg_parser.add_argument('--poisson_coeff', type=int, default=20,
                            help='Coefficient for Poisson noise.')
    arg_parser.add_argument('--iplane', type=int, default=1,
                            help='which plane to run hybrid ground-truth generation on.')
    
                        
    args = arg_parser.parse_args()

    if len(args.root) > 0 and not args.sweep and not args.param_sweep:
        root = Path(args.root)
        (root / 'sims').mkdir(parents=True, exist_ok=True)
        hybrid_gt_run(root, n_planes=4, iplane=args.iplane, ds=2, 
                      n_ell=args.n_ell, neu_coeff=args.neu_coeff, poisson_coeff=args.poisson_coeff,
                      threshold_scaling=args.threshold_scaling)        
    elif args.param_sweep:
        root = Path(args.root)
        (root / 'sims').mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        neu_coeff = 0.4
        poisson_coeff = 20
        n_ell = 2000
        for th in np.arange(0.6, 1.5, 0.1):
            bsub = f'bsub -n 8 -gpu "num=1" -q gpu_a100 ' \
                f'-J {root}/logs/hybrid_gt_th_{th:.1f} ' \
                f'-o {root}/logs/hybrid_gt_th_{th:.1f}.out ' \
                f'"~/miniforge3/envs/s2p/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
                f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} ' \
                f'--iplane {args.iplane} --threshold_scaling {th:.1f} ' \
                f' > {root}/logs/hybrid_gt_th_{th:.1f}.log"'
            print(bsub)
            os.system(bsub)
        
    elif args.sweep:
        root = Path(args.root)
        (root / 'sims').mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        root = args.root
        th = args.threshold_scaling
        iplane = args.iplane

        # n_ell = 0 
        # neu_coeff = 0
        # poisson_coeff = 0
        # bsub = f'bsub -n 8 -gpu "num=1" -q gpu_l4 ' \
        #         f'-J {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff} ' \
        #         f'-o {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.out ' \
        #         f'"~/miniforge3/envs/s2p/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
        #         f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} --iplane {iplane} ' \
        #         f' > {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.log"'
        # print(bsub)
        # os.system(bsub)


        for n_ell in np.arange(0, 4001, 500):
            neu_coeff = 0.4
            poisson_coeff = 20
            bsub = f'bsub -n 8 -gpu "num=1" -q gpu_a100 ' \
                f'-J {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff} ' \
                f'-o {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.out ' \
                f'"~/miniforge3/envs/s2p/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
                f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} ' \
                f'--iplane {args.iplane} --threshold_scaling {th:.1f} ' \
                f' > {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.log"'
            print(bsub)
            os.system(bsub)

        for neu_coeff in np.arange(0, 0.81, 0.1):
            n_ell = 2000
            poisson_coeff = 20
            if neu_coeff == 0.4:
                continue # in loop above
            bsub = f'bsub -n 8 -gpu "num=1" -q gpu_a100 ' \
                f'-J {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff} ' \
                f'-o {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.out ' \
                f'"~/miniforge3/envs/s2p/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
                f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} --iplane {iplane} ' \
                f' > {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.log"'
            print(bsub)
            os.system(bsub)


        for poisson_coeff in [0, 5, 10, 20, 50, 100, 200]:
            n_ell = 2000
            neu_coeff = 0.4
            if poisson_coeff == 20:
                continue # in loop above
            bsub = f'bsub -n 8 -gpu "num=1" -q gpu_a100 ' \
                f'-J {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff} ' \
                f'-o {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.out ' \
                f'"~/miniforge3/envs/s2p/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
                f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} --iplane {iplane} ' \
                f' > {root}/logs/hybrid_gt_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.log"'
            print(bsub)
            os.system(bsub)

            

            
        
