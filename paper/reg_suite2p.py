from suite2p.registration.register import (compute_reference, compute_filters_and_norm, 
                                           compute_shifts, shift_frames)
from suite2p.io import BinaryFile
import numpy as np
import argparse 
import os
import torch 
from torch.nn import functional as F
import time
import logging 
from tqdm import trange
import sys
from suite2p.logger import TqdmToLogger
from suite2p.registration import register, rigid, nonrigid


logger = logging.getLogger("s2p")


def example_refinit(root):
    db = np.load(os.path.join(root, "suite2p/plane0/db.npy"), allow_pickle=True).item()
    settings = np.load(os.path.join(root, "suite2p/plane0/settings.npy"), allow_pickle=True).item()
    Ly, Lx = db["Ly"], db["Lx"]
    raw_file = db["raw_file"]
    reg_file = db["reg_file"]   
    settings = settings['registration']
    n_frames = db['nframes']
    ix_frames = np.linspace(0, n_frames, 1 + min(settings["nimg_init"], n_frames), 
                                    dtype=int)[:-1]
    with BinaryFile(Ly, Lx, filename=raw_file, write=False) as f:
        frames = f[ix_frames].copy()
    frames = frames[:, -400:, -400:]

    frames = torch.from_numpy(frames)

    nimg, Ly, Lx = frames.shape
    fr_z = frames.clone().reshape(nimg, -1).double()
    fr_z -= fr_z.mean(dim=1, keepdim=True)
    cc = fr_z @ fr_z.T
    ndiag = torch.diag(cc)**0.5
    cc = cc / torch.outer(ndiag, ndiag)
    CCsort = -torch.sort(-cc, dim=1)[0]
    # find frame most correlated to other frames
    bestCC = CCsort[:, 1:20].mean(dim=1) # 1-20 to exclude own frame
    imax = torch.argmax(bestCC)
    # average top 20 frames most correlated to imax
    indsort = torch.argsort(-cc[imax, :])
    refImg = fr_z[indsort[:20]].mean(axis=0).cpu().numpy().astype("int16")
    refImg = refImg.reshape(Ly, Lx)
    refImg += fr_z.mean().numpy().astype("int16")
    refImg0 = refImg.copy()

    refImg = compute_reference(frames.numpy().copy(), settings)

    frames_mean = frames.numpy().mean(axis=0)
    return frames_mean, cc, imax, refImg0, refImg


def example_reg(root):
    reg_outputs = np.load(os.path.join(root, "suite2p/plane0/reg_outputs.npy"), allow_pickle=True).item()
    db = np.load(os.path.join(root, "suite2p/plane0/db.npy"), allow_pickle=True).item()
    settings = np.load(os.path.join(root, "suite2p/plane0/settings.npy"), allow_pickle=True).item()
    Ly, Lx = db["Ly"], db["Lx"]
    raw_file = db["raw_file"]
    reg_file = db["reg_file"]   
    with BinaryFile(Ly, Lx, filename=raw_file, write=False) as f:
        frand = f[::1000].copy()

    frand = frand[:,-400:,-400:]
    print(frand.shape)
    Ly, Lx = frand.shape[1:]

    yoff, xoff = reg_outputs["yoff"], reg_outputs["xoff"]
    tPC = reg_outputs["tPC"]
    regPC = reg_outputs["regPC"]
    regDX = reg_outputs["regDX"]


    fr_reg = torch.from_numpy(frand).to(torch.device("cuda"))
    refImg = reg_outputs["refImg"]
    refImg = refImg[-400:,-400:]
    refAndMasks = register.compute_filters_and_norm(refImg)
    (maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, 
        blocks, rmin, rmax) = refAndMasks
    fr_reg = torch.clip(fr_reg, rmin, rmax)
    device = fr_reg.device

    # rigid registration
    ymax, xmax, cmax, cc = rigid.phasecorr(fr_reg, cfRefImg, maskMul, maskOffset,
                                        maxregshift=settings["registration"]["maxregshift"],
                                        smooth_sigma_time=settings["registration"]["smooth_sigma_time"], 
                                        return_cc=True)
        
    # non-rigid registration
    if maskMulNR is not None:
        # shift torch frames to reference
        fr_reg = torch.stack([torch.roll(frame, shifts=(-dy, -dx), dims=(0, 1))
                                for frame, dy, dx in zip(fr_reg, ymax, xmax)], axis=0)
        ymax1, xmax1, cmax1, ccsm, ccb = nonrigid.phasecorr(fr_reg, blocks, 
                                                    maskMulNR, maskOffsetNR, cfRefImgNR, 
                                                snr_thresh=settings["registration"]["snr_thresh"], 
                                                maxregshiftNR=settings["registration"]["maxregshiftNR"])

    ifr = 27
    cc_ex = cc[ifr]
    cc_nr_ex = ccsm[:,ifr]
    cc_up_ex = ccb[:,ifr].cpu().numpy().reshape(-1, 61, 61)

    yblock, xblock, nblocks = blocks[:3]
    ymax1 = ymax1.reshape(-1, *nblocks)
    xmax1 = xmax1.reshape(-1, *nblocks)
    mshy, mshx = torch.meshgrid(torch.arange(Ly
    , dtype=torch.float, device=device),
                            torch.arange(Lx, dtype=torch.float, device=device), indexing="ij")
    yb = np.array(yblock[::nblocks[1]]).mean(axis=1).astype("int")
    xb = np.array(xblock[:nblocks[1]]).mean(axis=1).astype("int")
    Lyc, Lxc = int(yb.max() - yb.min()), int(xb.max() - xb.min())
    yxup = F.interpolate(torch.stack((ymax1, xmax1), dim=1), 
                            size=(Lyc, Lxc), mode="bilinear", align_corners=True)
    yxup = F.pad(yxup, (int(xb.min()), Lx - int(xb.max()), 
                        int(yb.min()), Ly - int(yb.max())), mode="replicate")
    print(yxup.shape)
    yxup = yxup[ifr].cpu().numpy()

    u = ymax1[ifr].cpu().numpy()
    v = xmax1[ifr].cpu().numpy()

    freg = fr_reg.cpu().numpy()
    return frand, freg, refImg, cc_ex, yoff, xoff, yblock, xblock, nblocks, cc_nr_ex, cc_up_ex, yxup, u, v, tPC, regPC, regDX

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--tfr", type=int, default=500)
    parser.add_argument("--rigid", action="store_true")
    args = parser.parse_args()
    device=torch.device("cuda")
    tfr = args.tfr
    rigid = args.rigid
    root = args.root
    batch_size = 250
    print(tfr, rigid)

    db = np.load(os.path.join(root, "suite2p/plane0/db.npy"), allow_pickle=True).item()
    settings = np.load(os.path.join(root, "suite2p/plane0/settings.npy"), allow_pickle=True).item()["registration"]
    Ly, Lx = db["Ly"], db["Lx"]
    n_frames = db["nframes"]
    print(Ly, Lx)
    bin_name = os.path.join(root, "suite2p/plane0/data_raw.bin")
    
    tic = time.time()
    with BinaryFile(Ly, Lx, filename=bin_name) as f_align_in, \
        BinaryFile(Ly, Lx, filename=bin_name[:-4]+f"_test{tfr}_{rigid}.bin", n_frames=tfr, write=True) as f_align_out:

        n_frames = tfr
        ix_frames = np.linspace(0, n_frames, 1 + min(settings["nimg_init"], n_frames), 
                                dtype=int)[:-1]
        frames = f_align_in[ix_frames].copy()
        bidiphase = 0

        t0 = time.time()
        refImg = compute_reference(frames, settings=settings, device=device)
        logger.info("Reference frame, %0.2f sec." % (time.time() - t0))
        
        ### ----- register frames to reference image -------------- ###
        # (from register_frames - copied here to allow a certain number of frames to be registered)

        norm_frames = settings["norm_frames"]
        smooth_sigma = settings["smooth_sigma"]
        smooth_sigma_time = settings["smooth_sigma_time"]
        spatial_taper = settings["spatial_taper"]
        block_size = settings["block_size"]
        nonrigid = not rigid #settings["nonrigid"]
        maxregshift = settings["maxregshift"]
        maxregshiftNR = settings["maxregshiftNR"]
        snr_thresh = settings["snr_thresh"]
        refAndMasks = compute_filters_and_norm(refImg, norm_frames=norm_frames, 
                                            spatial_smooth=smooth_sigma,
                                            spatial_taper=spatial_taper, 
                                            block_size=block_size if nonrigid else None, 
                                            device=device)
        (maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, 
            cfRefImgNR, blocks, rmin, rmax) = refAndMasks
        ### ------------- register frames to reference image ------------ ###

        mean_img = np.zeros((Ly, Lx), "float32")
        rigid_offsets, nonrigid_offsets, zpos, cmax_all = [], [], [], []

        n_batches = int(np.ceil(n_frames / batch_size))
        logger.info(f"Registering {n_frames} frames in {n_batches} batches")
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        for n in trange(n_batches, mininterval=10, file=tqdm_out):
            tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
            frames = f_align_in[tstart : tend]
            if device.type == "cuda":
                fr_torch = torch.from_numpy(frames).pin_memory().to(device)
            else:
                fr_torch = torch.from_numpy(frames)

            fr_reg = fr_torch.clone()
            offsets = compute_shifts(refAndMasks, fr_reg, maxregshift=maxregshift, 
                                    smooth_sigma_time=smooth_sigma_time, 
                                    snr_thresh=snr_thresh, maxregshiftNR=maxregshiftNR)
            ymax, xmax, cmax, ymax1, xmax1, cmax1, zest, cmax_all = offsets
            frames = shift_frames(fr_torch, ymax, xmax, ymax1, xmax1, blocks)
            
            # convert to numpy and concatenate offsets
            ymax, xmax, cmax = ymax.cpu().numpy(), xmax.cpu().numpy(), cmax.cpu().numpy()
            if ymax1 is not None:
                ymax1, xmax1 = ymax1.cpu().numpy(), xmax1.cpu().numpy()
                cmax1 = cmax1.cpu().numpy()
            offsets = [ymax, xmax, cmax, ymax1, xmax1, cmax1, zest, cmax_all]
            offsets_all = ([np.concatenate((offset_all, offset), axis=0) 
                        if offset is not None else None
                        for offset_all, offset in zip(offsets_all, offsets)] 
                            if n > 0 else offsets)
            
            # make mean image from all registered frames
            mean_img += frames.sum(axis=0) / n_frames

            # save aligned frames to bin file
            if f_align_out is not None:
                f_align_out[tstart : tend] = frames
            else:
                f_align_in[tstart : tend] = frames

    print(time.time() - tic)
    os.remove(bin_name[:-4]+f"_test{tfr}_{rigid}.bin")
    os.makedirs(os.path.join(root, "timings/"), exist_ok=True)
    np.save(os.path.join(root, f"timings/suite2p_{['', 'rigid_'][rigid]}{tfr}.npy"), time.time()-tic)

if __name__ == "__main__":
    main()
