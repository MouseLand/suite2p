import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import cv2
import logging
import numpy as np
import os
import tifffile
from pathlib import Path
from tqdm import trange 

try: 
    cv2.setNumThreads(0)
except():
    pass

import time
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf, params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour

from detect_f1_score import detect_f1_score

# params from caiman paper
# https://github.com/flatironinstitute/CaImAn/blob/v1.3/use_cases/CaImAnpaper/compare_gt_cnmf_CNN.py

global_params_caiman = {'min_SNR': 2,        # minimum SNR when considering adding a new neuron
                    'gnb': 2,             # number of background components
                    'rval_thr' : 0.80,     # spatial correlation threshold
                    'min_cnn_thresh' : 0.95,
                    'p' : 1,
                    'min_rval_thr_rejected': 0, # length of mini batch for OnACID in decay time units (length would be batch_length_dt*decay_time*fr)
                    'max_classifier_probability_rejected' : 0.1,    # flag for motion correction (set to False to compare directly on the same FOV)
                    'max_fitness_delta_accepted' : -20,
                    'Npeaks' : 5,
                    'min_SNR_patch' : -10,
                    'min_r_val_thr_patch': 0.5,
                    'fitness_delta_min_patch': -5,
                    'update_background_components' : True,# whether to update the background components in the spatial phase
                    'low_rank_background'  : True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                                                #(to be used with one background per patch)
                    'only_init_patch'  : True,
                    'is_dendrites'  : False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                    'init_method'  : 'greedy_roi',
                    'filter_after_patch'  :  False
                    }

# yuste params - visual cortex with low pixel resolution
# from https://github.com/flatironinstitute/CaImAn/blob/80bd3752efd69d51323bcf55c1858feb0a3148f5/caiman/tests/comparison_humans.py#L126
yuste_params = { 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 10,  # number of components per patch
                 'gSig': [5,5],  # expected half width of neurons
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix':0
                 }

caiman_params = {**global_params_caiman, **yuste_params}

# general dataset-dependent parameters
data_params = {
    'fr': 15.5,                    # imaging rate in frames per second
    'decay_time': 0.25,            # length of a typical transient in seconds
    'dxy': (2., 2.)                # spatial resolution in x and y in (um per pixel)
}


def hybrid_detect(root, iplane=1, neu_coeff=0.4, n_ell=2000, poisson_coeff=20,
                  num_processors_to_use=45, K=10, 
                  p=1, gnb=2, gSig0=5, rf=15,
                  tiff_file=None, filename=None, delete=True):

    if isinstance(K, int):
        K = [K]
    Ksweep = len(K) > 1

    if delete:
        delete_tiff = False if tiff_file is not None else True
        delete_memmap = False if filename is not None else True
    else:
        delete_tiff = False 
        delete_memmap = False
    
    if neu_coeff > 0 or n_ell > 0 or poisson_coeff > 0:
        reg_file = root / 'sims' / f'data_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}.bin'
    else:
        reg_file = root / 'suite2p_ds' / f'plane{iplane}' / 'data.bin'

    
    run_name = f'caiman_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}'

    # import sys
    # file = open(root / 'logs' / f'log_{run_name}.log', 'w')
    # sys.stdout = file

    db = np.load(root / "suite2p_ds" / f"plane{iplane}" / "db.npy", allow_pickle=True).item()
    n_frames, Ly, Lx = db["nframes"], db["Ly"], db["Lx"]
    shape = (n_frames, Ly, Lx)
    reg_outputs = np.load(root / "suite2p_ds" / f"plane{iplane}" / "reg_outputs.npy", allow_pickle=True).item()
    yrange, xrange = reg_outputs["yrange"], reg_outputs["xrange"]

    if tiff_file is None:
        print(f'create tiff from {reg_file}')
        tiff_file = root / 'sims' /  f"data_{run_name}.tif"
        
        f_reg = np.memmap(reg_file, mode='r', dtype='int16', shape=shape)
        data = f_reg[:, yrange[0]:yrange[1], xrange[0]:xrange[1]]
        #data = f_reg[:1000]
        #data = data[:, yrange[0]:yrange[1], xrange[0]:xrange[1]]
        print(f"data shape: {data.shape}, dtype: {data.dtype}") 
        tifffile.imwrite(tiff_file, data)


    # set up logging
    logfile = root / 'log.log' # Replace with a path if you want to log to a file
    logger = logging.getLogger('caiman')
    # Set to logging.INFO if you want much output, potentially much more output
    logger.setLevel(logging.WARNING)
    logfmt = logging.Formatter('%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s')
    if logfile is not None:
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logfmt)
    logger.addHandler(handler)

    
    fnames = [str(tiff_file)]

    t0 = time.time()

    if 'cluster' in locals():  # 'locals' contains list of current local variables
        print('Closing previous cluster')
        cm.stop_server(dview=cluster)
    print(f"Setting up new cluster w/ {num_processors_to_use} processors")
    c, cluster, n_processes = cm.cluster.setup_cluster(
            backend='multiprocessing', n_processes=num_processors_to_use, single_thread=False)
    print(f"Successfully initilialized multicore processing with a pool of {n_processes} CPU cores", flush=True)    

    if filename is None:
        os.environ['CAIMAN_TEMP'] = str(root / 'sims')
        filename = cm.save_memmap(fnames, order="C", base_name=run_name)
        print('created memmap file:', filename, flush=True)

    if delete_tiff:
        print(f"removing tiff file: {tiff_file}")
        os.remove(str(tiff_file))  # remove the tiff file


    for K0 in K:
        parameter_dict = {**data_params, **caiman_params}
        print(K0)
        parameter_dict['K'] = K0
        parameter_dict['p'] = p 
        parameter_dict['gnb'] = gnb
        parameter_dict['gSig'] = [gSig0, gSig0]
        parameter_dict['rf'] = rf
        run_name0 = f'caiman_K_{K0}_p_{p}_gnb_{gnb}_gSig0_{gSig0}_rf_{rf}_neu_{neu_coeff:.2f}_ell_{n_ell}_poisson_{poisson_coeff}_plane{iplane}'
        
        parameters = params.CNMFParams(params_dict=parameter_dict) # CNMFParams is the parameters class

        #print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
        print(root / 'sims' / f'F_{run_name0}.npy')

        Yr, dims, T = cm.load_memmap(filename)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        print(f"images shape: {images.shape}, dtype: {images.dtype}", flush=True)

        cnmf_model = cnmf.CNMF(num_processors_to_use, 
                        params=parameters, 
                        dview=cluster)
        
        cnmf_model.fit(images)
        print(f'first fit done, time {time.time()-t0:0.2f}s', flush=True)

        cnmf_refit = cnmf_model.refit(images, dview=cluster)
        print(f'refit done, time {time.time()-t0:0.2f}s', flush=True)

        cnmf_refit.estimates.detrend_df_f(quantileMin=8, 
                                        frames_window=250,
                                        flag_auto=False,
                                        use_residuals=False);  
        print(f'detrend_df_f done, time {time.time()-t0:0.2f}s', flush=True)

        cell_pix = cnmf_refit.estimates.A.toarray()
        ncells = cell_pix.shape[-1]
        Ly, Lx = images.shape[-2:]
        cell_pix = cell_pix.reshape(Lx, Ly, ncells).transpose(2, 1, 0)
        stat_caiman = []
        pix = [np.nonzero(cell_pix[i]) for i in trange(ncells)]
        for i in trange(ncells):
            ypix, xpix = pix[i]
            lam = cell_pix[i, ypix, xpix]
            stat_caiman.append({'ypix': ypix, 'xpix': xpix, 'lam': lam,
                            "radius": (len(ypix) / np.pi) ** 0.5,
                            "med": np.array([int(np.median(ypix)), int(np.median(xpix))]),
                            "overlap": np.zeros(len(ypix), "bool"),
                            "npix": len(ypix),})
            
        stat_caiman = np.array(stat_caiman)

        np.save(root / 'sims' / f"F_{run_name0}.npy", cnmf_refit.estimates.F_dff)
        np.save(root / 'sims' / f"stat_{run_name0}.npy", stat_caiman)
        np.save(root / 'sims' / f"C_{run_name0}.npy", cnmf_refit.estimates.C)
        np.save(root / 'sims' / f"YrA_{run_name0}.npy", cnmf_refit.estimates.YrA)
        np.save(root / 'sims' / f"r_{run_name0}.npy", cnmf_refit.estimates.r_values)
        np.save(root / 'sims' / f"S_{run_name0}.npy", cnmf_refit.estimates.S)
        
        dF = cnmf_refit.estimates.F_dff.copy()
            
        # load ground-truth ROIs
        stat_gt = np.load(root / "benchmarks" / f"stat_gt_plane{iplane}.npy", allow_pickle=True)
        F_gt = np.load(root / "benchmarks" / f"F_gt_plane{iplane}.npy", allow_pickle=True)
        Fneu_gt = np.load(root / "benchmarks" / f"Fneu_gt_plane{iplane}.npy", allow_pickle=True)
        dF_gt = F_gt.copy() - 0.7 * Fneu_gt    
        
        stat = stat_caiman.copy()
        #dF = cnmf_refit.estimates.C.copy() + cnmf_refit.estimates.YrA

        lam_threshold = 0.1
        for i in range(len(stat)):
            lam = stat[i]["lam"]
            stat[i]["ypix"] = stat[i]["ypix"][lam > lam_threshold * lam.max()]
            stat[i]["xpix"] = stat[i]["xpix"][lam > lam_threshold * lam.max()]
            stat[i]['ypix'] += yrange[0]
            stat[i]['xpix'] += xrange[0]
            stat[i]["lam"] = stat[i]["lam"][lam > lam_threshold * lam.max()]

        Ly, Lx = db['Ly'], db['Lx']
    
        tp, fp, fn, f1 = detect_f1_score(dF, dF_gt, stat, stat_gt, Ly=Ly, Lx=Lx, snr_threshold=0.25)

        np.save(root / 'sims' / f'results_{run_name0}.npy',
                np.array([tp, fp, fn, f1]))

    if delete_memmap:
        print(f"removing memmap file: {filename}")
        os.remove(filename)  # remove the memmap file

    
import argparse

if __name__ == '__main__':
    # argparse 
    arg_parser = argparse.ArgumentParser(description='Run hybrid ground-truth generation and Suite2p detection/extraction.')
    arg_parser.add_argument('--root', type=str, default='')
    arg_parser.add_argument('--param_sweep', action='store_true',
                            help='sweep number of components.')
    arg_parser.add_argument('--param_sweep_grid', action='store_true',
                            help='sweep several caiman parameters.')
    arg_parser.add_argument('--sweep', action='store_true',
                            help='Run a sweep of hybrid ground-truth generation with different parameters.')
    arg_parser.add_argument('--no_delete', action='store_true',
                            help='Do not delete tiff and memmap.')
    arg_parser.add_argument('--n_ell', type=int, default=1500,
                            help='Number of dendritic ellipses to generate.')
    arg_parser.add_argument('--Ksweep', action='store_true',
                            help='Run a sweep over K param in caiman.')
    arg_parser.add_argument('--K', type=int, default=8,
                            help='Number of components for caiman.')
    arg_parser.add_argument('--p', type=int, default=1,
                            help='Autoregressive order for caiman.')
    arg_parser.add_argument('--gnb', type=int, default=2,
                            help='Number of background components for caiman.')
    arg_parser.add_argument('--gSig0', type=int, default=5,
                            help='Expected half width of neurons in pixels for caiman.')
    arg_parser.add_argument('--rf', type=int, default=15,
                            help='Half-size of the patches in pixels for caiman.')
    arg_parser.add_argument('--neu_coeff', type=float, default=3,
                            help='Coefficient for neuropil contribution.')
    arg_parser.add_argument('--poisson_coeff', type=int, default=50,
                            help='Coefficient for Poisson noise.')
    arg_parser.add_argument('--n_processors', type=int, default=16,
                            help='Number of processors to use for parallel processing.')
    arg_parser.add_argument('--tiff_file', type=str, default=None,
                            help='Path to the tiff file to use instead of generating one.')
    arg_parser.add_argument('--filename', type=str, default=None,
                            help='Path to the memmap file to use instead of generating one.')
    arg_parser.add_argument('--iplane', type=int, default=1,
                            help='which plane to run hybrid ground-truth generation on.')
    
                        
    args = arg_parser.parse_args()

    if len(args.root) > 0 and not args.sweep and not args.param_sweep and not args.param_sweep_grid:
        root = Path(args.root)
        (root / 'sims').mkdir(parents=True, exist_ok=True)
        if args.Ksweep:
            K = [7,8,9,10,11,12,13]
        else:
            K = args.K
        hybrid_detect(root, iplane=args.iplane, K=K, p=args.p, gnb=args.gnb, gSig0=args.gSig0, rf=args.rf,
                      n_ell=args.n_ell, neu_coeff=args.neu_coeff, poisson_coeff=args.poisson_coeff, 
                      num_processors_to_use=args.n_processors, 
                      tiff_file=args.tiff_file, filename=args.filename, delete=(not args.no_delete))
    elif args.param_sweep:
        root = Path(args.root)
        (root / 'sims').mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        root = args.root
        n_ell = 2000
        neu_coeff = 0.4
        poisson_coeff = 20
        # p=1, gnb=2, K=8, gSig0=5, rf=15
        p, gnb, K, gSig0, rf = 1, 2, 8, 5, 15
        rstr = f'gt_{p}_{gnb}_{K}_{gSig0}_{rf}_plane{args.iplane}'
        bsub = f'bsub -n 24 ' \
            f'-J {root}/logs/caiman_{rstr} ' \
            f'-o {root}/logs/caiman_{rstr}.out ' \
            f'"source ~/add_mini.sh; source activate cm; ~/miniforge3/envs/cm/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
            f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} --n_processors 24 ' \
            f'--no_delete --iplane {args.iplane} --Ksweep > {root}/logs/caiman_{rstr}.log"'
        print(bsub)
        os.system(bsub)
        
    elif args.param_sweep_grid:
        iplane = args.iplane
        root = Path(args.root)
        (root / 'sims').mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(parents=True, exist_ok=True)

        n_ell = 2000
        neu_coeff = 0.4
        poisson_coeff = 20
        mmap_file = list((root / 'sims').glob(f'caiman_neu_0.40_ell_2000_poisson_20_plane{iplane}*.mmap'))[0]
        print(mmap_file)
        root = args.root
        # p=1, gnb=2, merge_thr=0.85, K=9, gSig0=5, rf=15
        for p in [1, 2]:
            for gnb in [1, 2, 3]:
                for K in [6, 9, 12]:
                    for gSig0 in [3, 5, 7]:
                        for rf in [12, 15, 18]:
                            rstr = f'gt_{p}_{gnb}_{K}_{gSig0}_{rf}_plane{iplane}'
                            bsub = f'bsub -n 16 ' \
                                f'-J {root}/logs/caiman_{rstr} ' \
                                f'-o {root}/logs/caiman_{rstr}.out ' \
                                f'"source ~/add_mini.sh; source activate cm; ~/miniforge3/envs/cm/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
                                f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} --n_processors 16 ' \
                                f'--p {p} --gnb {gnb} --K {K} --gSig0 {gSig0} --rf {rf} --iplane {iplane} ' \
                                f'--tiff_file {root}/sims/caiman_neu_0.40_ell_2000_poisson_20_plane{iplane}.tif ' \
                                f'--filename {mmap_file} ' \
                                f'> {root}/logs/caiman_{rstr}.log"'
                            print(bsub)
                            os.system(bsub)
                        
                                

    elif args.sweep:
        root = Path(args.root)
        (root / 'sims').mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        root = args.root
        
        iplane = args.iplane

        # # run original data
        # n_ell = 0 
        # neu_coeff = 0 
        # poisson_coeff = 0
        # bsub = f'bsub -n 16 ' \
        #     f'-J {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff} ' \
        #     f'-o {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.out ' \
        #     f'"source ~/add_mini.sh; source activate cm; ~/miniforge3/envs/cm/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
        #     f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} ' \
        #     f'--iplane {iplane} ' \
        #     f' > {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}.log"'
        # print(bsub)
        # os.system(bsub)            

        for n_ell in np.arange(0, 4001, 500):
            neu_coeff = 0.4
            poisson_coeff = 20
            bsub = f'bsub -n 16 ' \
                f'-J {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane} ' \
                f'-o {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane}.out ' \
                f'"source ~/add_mini.sh; source activate cm; ~/miniforge3/envs/cm/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
                f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} --K 9 ' \
                f'--iplane {iplane} ' \
                f' > {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane}.log"'
            print(bsub)
            os.system(bsub)
            #time.sleep(15)
            
        
        for neu_coeff in np.arange(0, 0.81, 0.1):
            n_ell = 2000
            poisson_coeff = 20
            if neu_coeff == 0.4:
                continue # in loop above
            bsub = f'bsub -n 16 ' \
                f'-J {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane} ' \
                f'-o {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane}.out ' \
                f'"source ~/add_mini.sh; source activate cm; ~/miniforge3/envs/cm/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
                f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} --K 9 ' \
                f'--iplane {iplane} ' \
                f' > {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane}.log"'
            print(bsub)
            os.system(bsub)


        for poisson_coeff in [0, 5, 10, 20, 50, 100, 200]:
            n_ell = 2000
            neu_coeff = 0.4
            if poisson_coeff == 20:
                continue # in loop above
            bsub = f'bsub -n 16 ' \
                f'-J {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane} ' \
                f'-o {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane}.out ' \
                f'"source ~/add_mini.sh; source activate cm; ~/miniforge3/envs/cm/bin/python {__file__} --root {root} --n_ell {n_ell} ' \
                f'--neu_coeff {neu_coeff:.2f} --poisson_coeff {poisson_coeff} --K 9 ' \
                f' --iplane {iplane} ' \
                f' > {root}/logs/caiman_{n_ell}_{neu_coeff:.2f}_{poisson_coeff}_plane{iplane}.log"'
            print(bsub)
            os.system(bsub)

# python detect_caiman.py --root /groups/stringer/stringerlab/suite2p_paper/hybrid_gt/ --sweep --iplane 2
