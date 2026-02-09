import numpy as np
from scipy.stats import zscore
from tqdm import trange
from pathlib import Path

def detect_f1_score(dF, dF_gt, stat, stat_gt, Ly=512, Lx=512, 
              snr_threshold=0.25):
    snr_gt = 1 - 0.5 * np.diff(dF_gt, axis=1).var(axis=1) / dF_gt.var(axis=1)
    npix_gt = np.array([len(s['ypix']) for s in stat_gt])

    snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)
    npix = np.array([len(s['ypix']) for s in stat])

    # filter ground-truth and predicted ROIs
    min_size = np.percentile(npix_gt, 5)
    max_size = np.percentile(npix_gt, 95)
    # snr_threshold = 0.4
    igood_gt = (snr_gt > snr_threshold) * (npix_gt > min_size) * (npix_gt < max_size)
    # snr_threshold = 0.25
    igood = (snr > snr_threshold) * (npix > min_size) * (npix < max_size)
    print(igood_gt.sum(), igood.sum())

    if igood.sum() > 0:
        stat_gt = stat_gt[igood_gt]
        dF_gt = dF_gt[igood_gt]
        snr_gt = snr_gt[igood_gt]

        stat = stat[igood]
        dF = dF[igood]
        snr = snr[igood]

        # correlate activity traces
        cc = (zscore(dF_gt, axis=1) @ zscore(dF, axis=1).T) / dF.shape[1]

        # find overlapping ROIs
        matched = np.zeros((len(stat_gt), len(stat)), 'float32')
        iou = np.zeros((len(stat_gt), len(stat)), 'float32')
        ly = 20
        for i in trange(len(stat)):
            sf = stat[i]
            if sf['ypix'].size < 10:
                continue
            ypix, xpix, lam = sf['ypix'].copy(), sf['xpix'].copy(), sf['lam'].copy()
            lam /= (lam**2).sum()**0.5
            # box around ROI
            if 'med' not in sf:
                ymed, xmed = int(np.median(ypix)), int(np.median(xpix))
            else:
                ymed, xmed = int(sf['med'][0]), int(sf['med'][1])
            inds = (slice(max(0, ymed - ly), min(ymed + ly, Ly)), slice(max(0, xmed - ly), min(xmed + ly, Lx)))
            mf = np.zeros((Ly, Lx), np.float32)
            #if 'soma_crop' in sf:
            #    ypix, xpix, lam = ypix[sf['soma_crop']], xpix[sf['soma_crop']], lam[sf['soma_crop']]
            mf[ypix, xpix] = lam
            mfc = mf > 0. *  lam.max()
            mfc = mfc[inds].flatten()
            
            # matched anatomical masks (will not compute IOU for all masks)
            for j, sa in enumerate(stat_gt):
                ypix_a, xpix_a = sa['ypix'], sa['xpix']
                if (np.logical_and(ypix_a > inds[0].start, ypix_a < inds[0].stop).sum() > 0
                        and np.logical_and(xpix_a > inds[1].start, xpix_a
                                            < inds[1].stop).sum() > 0):
                    lam_a = sa['lam'].copy()
                    ma = np.zeros((Ly, Lx), 'bool')
                    ma[ypix_a, xpix_a] = lam_a > 0#.1 * lam_a.max()
                    mac = ma[inds].flatten()
                    intersection = (mac[mfc] > 0).sum()
                    matched[j, i] = (intersection / mac.sum() > 0.5) * (intersection / mfc.sum() > 0.5)
                    iou[j, i] = intersection / (mac.sum() + mfc.sum() - intersection)

        cc_filt = cc.copy() 
        #cc_filt *= (matched > 0.5)
        cc_filt *= iou > 0.5
        
        # print((cc_filt.max(axis=1) > 0.5).sum(), len(np.unique(cc_filt[cc_filt.max(axis=1)>0.5].argmax(axis=1))))

        imatch_gt = cc_filt.max(axis=1) > 0.5
        imatch_uq = len(np.unique(cc_filt[imatch_gt].argmax(axis=1)))
        print(imatch_uq, imatch_gt.sum() / igood_gt.sum())

        tp = imatch_uq
        fp = len(stat) - imatch_uq
        fn = len(stat_gt) - imatch_uq
        f1 = tp / (tp + 0.5 * (fp + fn))

    else:
        print('no good ROIs found')
        tp = 0
        fp = 0
        fn = igood_gt.sum()
        f1 = 0

    print('TP: %d, FP: %d, FN: %d, F1: %.3f' % (tp, fp, fn, f1), flush=True)

    return tp, fp, fn, f1
