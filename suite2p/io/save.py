import os
from natsort import natsorted
import numpy as np
from datetime import datetime
import scipy
import pathlib

def save_mat(ops, stat, F, Fneu, spks, iscell, redcell, F_chan2=None, Fneu_chan2=None):
    ops_matlab = ops.copy()
    if ops_matlab.get('date_proc'):
        try:
            ops_matlab['date_proc'] = str(datetime.strftime(ops_matlab['date_proc'], "%Y-%m-%d %H:%M:%S.%f"))
        except:
            pass
    for k in ops_matlab.keys():
        if isinstance(ops_matlab[k], (pathlib.WindowsPath, pathlib.PosixPath)):
            ops_matlab[k] = os.fspath(ops_matlab[k].absolute())
        elif isinstance(ops_matlab[k], list) and len(ops_matlab[k]) > 0:
            if isinstance(ops_matlab[k][0], (pathlib.WindowsPath, pathlib.PosixPath)):
                ops_matlab[k] = [os.fspath(p.absolute()) for p in ops_matlab[k]]
                print(k, ops_matlab[k])

    if F_chan2 is not None:
        scipy.io.savemat(
            file_name=os.path.join(ops['save_path'], 'Fall.mat'),
            mdict={
                'stat': stat,
                'ops': ops_matlab,
                'F': F,
                'Fneu': Fneu,
                'F_chan2': F_chan2,
                'Fneu_chan2': Fneu_chan2,
                'spks': spks,
                'iscell': iscell,
                'redcell': redcell
            }
        )
    else:
        scipy.io.savemat(
            file_name=os.path.join(ops['save_path'], 'Fall.mat'),
            mdict={
                'stat': stat,
                'ops': ops_matlab,
                'F': F,
                'Fneu': Fneu,
                'spks': spks,
                'iscell': iscell,
                'redcell': redcell
            }
        )

def compute_dydx(ops1):
    ops = ops1[0].copy()
    dx = np.zeros(len(ops1), np.int64)
    dy = np.zeros(len(ops1), np.int64)
    if ('dx' not in ops) or ('dy' not in ops):
        Lx = ops['Lx']
        Ly = ops['Ly']
        nX = np.ceil(np.sqrt(ops['Ly'] * ops['Lx'] * len(ops1))/ops['Lx'])
        nX = int(nX)
        for j in range(len(ops1)):
            dx[j] = (j%nX) * Lx
            dy[j] = int(j/nX) * Ly
    else:
        dx = np.array([o['dx'] for o in ops1])
        dy = np.array([o['dy'] for o in ops1])
        unq = np.unique(np.vstack((dy,dx)), axis=1)
        nrois = unq.shape[1]
        if nrois < len(ops1):
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
                    dx[j*nrois + k] += (j%nX) * xmax
                    dy[j*nrois + k] += int(j/nX) * ymax 
    return dy, dx

def combined(save_folder, save=True):
    """ Combines all the folders in save_folder into a single result file.

    can turn off saving (for gui loading)

    Multi-plane recordings are arranged to best tile a square.
    Multi-roi recordings are arranged by their dx,dy physical localization.
    Multi-plane / multi-roi recordings are tiled after using dx,dy.
    """
    plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5]=='plane'])
    ops1 = [np.load(os.path.join(f, 'ops.npy'), allow_pickle=True).item() for f in plane_folders]
    dy, dx = compute_dydx(ops1)
    Ly = np.array([ops['Ly'] for ops in ops1])
    Lx = np.array([ops['Lx'] for ops in ops1])
    LY = int(np.amax(dy + Ly))
    LX = int(np.amax(dx + Lx))
    meanImg = np.zeros((LY, LX))
    meanImgE = np.zeros((LY, LX))
    if ops1[0]['nchannels']>1:
        meanImg_chan2 = np.zeros((LY, LX))
    if any(['meanImg_chan2_corrected' in ops for ops in ops1]):
        meanImg_chan2_corrected = np.zeros((LY, LX))
    if any(['max_proj' in ops for ops in ops1]):
        max_proj = np.zeros((LY, LX))

    Vcorr = np.zeros((LY, LX))
    Nfr = np.amax(np.array([ops['nframes'] for ops in ops1]))
    ii=0
    for k,ops in enumerate(ops1):
        fpath = plane_folders[k]
        if not os.path.exists(os.path.join(fpath,'stat.npy')):
            continue
        stat0 = np.load(os.path.join(fpath,'stat.npy'), allow_pickle=True)
        xrange = np.arange(dx[k], dx[k] + Lx[k])
        yrange = np.arange(dy[k], dy[k] + Ly[k])
        meanImg[np.ix_(yrange, xrange)] = ops['meanImg']
        meanImgE[np.ix_(yrange, xrange)] = ops['meanImgE']
        if ops['nchannels']>1:
            if 'meanImg_chan2' in ops:
                meanImg_chan2[np.ix_(yrange, xrange)] = ops['meanImg_chan2']
        if 'meanImg_chan2_corrected' in ops:
            meanImg_chan2_corrected[np.ix_(yrange, xrange)] = ops['meanImg_chan2_corrected']

        xrange = np.arange(dx[k]+ops['xrange'][0],dx[k]+ops['xrange'][-1])
        yrange = np.arange(dy[k]+ops['yrange'][0],dy[k]+ops['yrange'][-1])
        Vcorr[np.ix_(yrange, xrange)] = ops['Vcorr']
        if 'max_proj' in ops:
            max_proj[np.ix_(yrange, xrange)] = ops['max_proj']
        for j in range(len(stat0)):
            stat0[j]['xpix'] += dx[k]
            stat0[j]['ypix'] += dy[k]
            stat0[j]['med'][0] += dy[k]
            stat0[j]['med'][1] += dx[k]
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
        if ii==0:
            F, Fneu, spks, stat, iscell, redcell = F0, Fneu0, spks0, stat0, iscell0, redcell0
        else:
            F    = np.concatenate((F, F0))
            Fneu = np.concatenate((Fneu, Fneu0))
            spks = np.concatenate((spks, spks0))
            stat = np.concatenate((stat,stat0))
            iscell = np.concatenate((iscell,iscell0))
            if hasred:
                redcell = np.concatenate((redcell,redcell0))
        ii+=1
        print('appended plane %d to combined view'%k)
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

    if save:
        if len(ops['save_folder']) > 0:
            fpath = os.path.join(ops['save_path0'], ops['save_folder'], 'combined')
        else:
            fpath = os.path.join(ops['save_path0'], 'suite2p', 'combined')
    else:
        fpath = os.path.join(save_folder, 'combined')
    
    if not os.path.isdir(fpath):
        os.makedirs(fpath)

    ops['save_path'] = fpath

    # need to save iscell regardless (required for GUI function)
    np.save(os.path.join(fpath, 'iscell.npy'), iscell)
    if hasred:
        np.save(os.path.join(fpath, 'redcell.npy'), redcell)
    else:
        redcell = np.zeros_like(iscell)

    if save:
        np.save(os.path.join(fpath, 'F.npy'), F)
        np.save(os.path.join(fpath, 'Fneu.npy'), Fneu)
        np.save(os.path.join(fpath, 'spks.npy'), spks)
        np.save(os.path.join(fpath, 'ops.npy'), ops)
        np.save(os.path.join(fpath, 'stat.npy'), stat)
        
        # save as matlab file
        if ops.get('save_mat'):
            matpath = os.path.join(ops['save_path'],'Fall.mat')
            save_mat(ops, stat, F, Fneu, spks, iscell, redcell)
            
    return stat, ops, F, Fneu, spks, iscell[:,0], iscell[:,1], redcell[:,0], redcell[:,1], hasred

