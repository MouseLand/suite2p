from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.stats import mode
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import uniform_filter
import numpy as np
import time
from numba import vectorize, float32

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

@vectorize('float32(float32, float32)', target = 'parallel', nopython=True)
def my_max(a, b):
    return max(a,b)
@vectorize('float32(float32, float32)', target = 'parallel', nopython=True)
def my_sum(a, b):
    return a+b

def get_mov(ops):
    t0 = tic()
    badframes = False
    if 'badframes' in ops:
        badframes = True
        nframes = ops['nframes'] - ops['badframes'].sum()
    else:
        nframes = ops['nframes']
    bin_min = np.floor(nframes / ops['nbinned']).astype('int32');
    bin_min = max(bin_min, 1)
    bin_tau = np.round(ops['tau'] * ops['fs']).astype('int32');
    nt0 = max(bin_min, bin_tau)
    ops['nbinned'] = np.floor(nframes/nt0).astype('int32')
    print('Binning movie in chunks of length %2.2d'%(nt0))
    Ly = ops['Ly']
    Lx = ops['Lx']
    Lyc = ops['yrange'][-1] - ops['yrange'][0]
    Lxc = ops['xrange'][-1] - ops['xrange'][0]

    nimgbatch = 500
    nimgbatch = min(nframes, nimgbatch)
    nimgbatch = nt0 * np.floor(nimgbatch/nt0)
    nbytesread = np.int64(Ly*Lx*nimgbatch*2)
    mov = np.zeros((ops['nbinned'], Lyc, Lxc), np.float32)
    max_proj = np.zeros((Lyc, Lxc), np.float32)
    ix = 0
    idata = 0
    # load and bin data
    with open(ops['reg_file'], 'rb') as reg_file:
        while True:
            buff = reg_file.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            buff = []
            nimgd = int(np.floor(data.size / (Ly*Lx)))
            if nimgd < nt0:
                break
            data = np.reshape(data, (-1, Ly, Lx))
            dinds = idata + np.arange(0,data.shape[0],1,int)
            idata+=data.shape[0]
            if badframes and np.sum(ops['badframes'][dinds])>.5:
                data = data[~ops['badframes'][dinds],:,:]
            nimgd = data.shape[0]
            if nimgd < nimgbatch:
                nmax = int(np.floor(nimgd / nt0) * nt0)
                data = data[:nmax,:,:]
            dbin = np.reshape(data, (-1,nt0,Ly,Lx))
            DD = dbin.mean(axis=1)
            # crop into valid area
            mov[ix:ix+dbin.shape[0],:,:] = DD[:, ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]]
            ix += dbin.shape[0]
    mov = mov[:ix,:,:]
    max_proj = np.max(mov, axis=0)
    print('Binned movie [%d,%d,%d], %0.2f sec.'%(mov.shape[0], mov.shape[1], mov.shape[2], toc(t0)))

    #nimgbatch = min(mov.shape[0] , max(int(500/nt0), int(240./nt0 * ops['fs'])))
    if ops['high_pass']<10:
        for j in range(mov.shape[1]):
            mov[:,j,:] -= gaussian_filter(mov[:,j,:], [ops['high_pass'], 0])
    else:
        ki0 = 0
        while 1:
            irange = ki0 + np.arange(0,int(ops['high_pass']))
            irange = irange[irange<mov.shape[0]]
            if len(irange)>0:
                mov[ki0:ki0+int(ops['high_pass']),:,:] -= mov[ki0:ki0+int(ops['high_pass']),:,:].mean(axis=0)
                ki0 += len(irange)
            else:
                break
    return mov, max_proj

def get_sdmov(mov, ops):
    ix = 0

    if len(mov.shape)>2:
        nbins,Ly, Lx = mov.shape
        npix = (Ly , Lx)
    else:
        nbins, npix = mov.shape
    batch_size = min(500,nbins)
    sdmov = np.zeros(npix, 'float32')
    while 1:
        if ix>=nbins:
            break
        sdmov += np.sum(np.diff(mov[ix:ix+batch_size,:, :], axis = 0)**2, axis = 0)
        ix = ix + batch_size
    sdmov = np.maximum(1e-10, (sdmov/nbins)**0.5)
    return sdmov


def get_overlaps(stat, ops):
    '''computes overlapping pixels from ROIs in stat
    inputs:
        stat, Ly, Lx
    outputs:
        stat
        assigned to stat: overlap: (npix,1) boolean whether or not pixels also in another cell
    '''
    Ly, Lx = ops['Ly'], ops['Lx']
    stat2 = []
    ncells = len(stat)
    mask = np.zeros((Ly,Lx))
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        mask[ypix,xpix] += 1
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        stat[n]['overlap'] = mask[ypix,xpix] > 1.5
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        stat2.append(stat[n])
    return stat2

def remove_overlaps(stat, ops, Ly, Lx):
    '''removes overlaps iteratively
    '''
    ncells = len(stat)
    mask = np.zeros((Ly,Lx))
    ix = [k for k in range(ncells)]
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        mask[ypix,xpix] += 1
    while 1:
        O = np.zeros((len(stat),1))
        for n in range(len(stat)):
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
            O[n] = np.mean(mask[ypix,xpix] > 1.5)
        #i = np.argmax(O)
        inds = (O > ops['max_overlap']).nonzero()[0]
        if len(inds) > 0:
            i = np.max(inds)
            ypix = stat[i]['ypix']
            xpix = stat[i]['xpix']
            mask[ypix,xpix] -= 1
            del stat[i], ix[i]
        else:
            break
    return stat, ix

def two_comps(mpix0, lam, Th2):
    mpix = mpix0.copy()
    xproj = mpix @ lam
    gf0 = xproj>Th2

    mpix[gf0, :] -= np.outer(xproj[gf0] , lam)
    vexp0 = np.sum(mpix0**2) - np.sum(mpix**2)

    k = np.argmax(np.sum(mpix * np.float32(mpix>0), axis=1))
    mu = [mpix[k].copy()]
    mu.append(mpix[k].copy())
    mu[0] = lam * np.float32(mu[0]<0)
    mu[1] = lam * np.float32(mu[1]>0)
    mpix = mpix0.copy()
    goodframe = []
    xproj = []
    for k in range(2):
        mu[k] /=(1e-6 + np.sum(mu[k]**2)**.5)
        xp = mpix @ mu[k]
        goodframe.append(gf0)
        xproj.append(xp[goodframe[k]])
        mpix[goodframe[k],:] -= np.outer(xproj[k], mu[k])

    flag = [False, False]
    V = np.zeros(2)
    for t in range(3):
        for k in range(2):
            if flag[k]:
                continue
            mpix[goodframe[k],:] += np.outer(xproj[k], mu[k])
            xp = mpix @ mu[k]
            goodframe[k]  = xp > Th2
            V[k] = np.sum(xp**2)
            if np.sum(goodframe[k])==0:
                flag[k] = True
                V[k] = -1
                continue
            xproj[k] = xp[goodframe[k]]
            mu[k] = np.mean(mpix[goodframe[k], :] * xproj[k][:,np.newaxis], axis=0)
            mu[k][mu[k]<0]  = 0
            mu[k] /=(1e-6 + np.sum(mu[k]**2)**.5)
            mpix[goodframe[k],:] -= np.outer(xproj[k], mu[k])
    k = np.argmax(V)
    vexp = np.sum(mpix0**2) - np.sum(mpix**2)
    return vexp/vexp0, (mu[k], xproj[k], goodframe[k])

def downsample(mov, flag=True):
    if flag:
        nu = 2
    else:
        nu = 1
    if len(mov.shape)<3:
        mov = mov[np.newaxis, :, :]
    nframes, Ly, Lx = mov.shape

    movd = np.zeros((nframes,int(np.ceil(Ly/2)),Lx), 'float32')
    Ly0 = 2*int(Ly/2)
    movd[:,:int(Ly0/2),:] = (mov[:,0:Ly0:2,:] + mov[:,1:Ly0:2,:])/2
    if Ly%2==1:
        movd[:,-1,:] = mov[:,-1,:]/nu

    mov2 = np.zeros((nframes,int(np.ceil(Ly/2)),int(np.ceil(Lx/2))), 'float32')
    Lx0 = 2*int(Lx/2)
    mov2[:,:,:int(Lx0/2)] = (movd[:,:,0:Lx0:2] + movd[:,:,1:Lx0:2])/2
    if Lx%2==1:
        mov2[:,:,-1] = movd[:,:,-1]/nu

    return mov2

def multiscale_mask(ypix0,xpix0,lam0, Lyp, Lxp):
    # given a set of masks on the raw image, this functions returns the downsampled masks for all spatial scales
    xs = [xpix0]
    ys = [ypix0]
    lms = [lam0]
    for j in range(1,len(Lyp)):
        ipix, ind = np.unique(np.int32(xs[j-1]/2)+np.int32(ys[j-1]/2)*Lxp[j], return_inverse=True)
        LAM = np.zeros(len(ipix))
        for i in range(len(xs[j-1])):
            LAM[ind[i]] += lms[j-1][i]/2
        lms.append(LAM)
        ys.append(np.int32(ipix/Lxp[j]))
        xs.append(np.int32(ipix%Lxp[j]))
    for j in range(len(Lyp)):
        ys[j], xs[j], lms[j] = extend_mask(ys[j], xs[j], lms[j], Lyp[j], Lxp[j])
    return ys, xs, lms

def add_square(yi,xi,lx,Ly,Lx):
    lhf = int((lx-1)/2)
    ipix = np.arange(-lhf,-lhf+lx)+ np.zeros(lx, 'int32')[:, np.newaxis]
    x0 = xi + ipix
    y0 = yi + ipix.T
    mask  = np.ones((lx,lx), 'float32')
    ix = np.all((y0>=0, y0<Ly, x0>=0 , x0<Lx), axis=0)
    x0 = x0[ix]
    y0 = y0[ix]
    mask = mask[ix]
    mask = mask/np.sum(mask**2)**.5
    return y0.flatten(), x0.flatten(), mask.flatten()

def extend_mask(ypix, xpix, lam, Ly, Lx):
    nel = len(xpix)
    yx = ((ypix, ypix, ypix, ypix-1, ypix-1,ypix-1, ypix+1,ypix+1,ypix+1),
          (xpix, xpix+1,xpix-1,xpix, xpix+1,xpix-1,xpix, xpix+1,xpix-1))
    yx = np.array(yx)
    yx = yx.reshape((2,-1))
    yu, ind = np.unique(yx, axis=1, return_inverse=True)
    LAM = np.zeros(yu.shape[1])
    for j in range(len(ind)):
        LAM[ind[j]] += lam[j%nel]/3
    ix = np.all((yu[0]>=0, yu[0]<Ly, yu[1]>=0 , yu[1]<Lx), axis = 0)
    ypix1,xpix1 = yu[:, ix]
    lam1 = LAM[ix]
    return ypix1,xpix1,lam1

def extendROI(ypix, xpix, Ly, Lx,niter=1):
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix-1, ypix+1), (xpix, xpix+1,xpix-1,xpix,xpix))
        yx = np.array(yx)
        yx = yx.reshape((2,-1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0]>=0, yu[0]<Ly, yu[1]>=0 , yu[1]<Lx), axis = 0)
        ypix,xpix = yu[:, ix]
    return ypix,xpix

def iter_extend(ypix, xpix, Ucell, Lyc,Lxc, iframes):
    #nsvd, Lyc, Lxc = Ucell.shape
    npix = 0
    iter = 0
    while npix<10000:
        npix = ypix.size
        ypix, xpix = extendROI(ypix,xpix,Lyc,Lxc, 1)
        usub = Ucell[np.ix_(iframes, ypix*Lxc+ xpix)]
        lam = np.mean(usub,axis=0)
        ix = lam>max(0, lam.max()/5.0)
        if ix.sum()==0:
            print('break')
            break;
        ypix, xpix,lam = ypix[ix],xpix[ix], lam[ix]
        if iter == 0:
            sgn = 1.
        if np.sign(sgn * (ix.sum()-npix))<=0:
            break
        else:
            npix = ypix.size
        iter += 1
    lam = lam/np.sum(lam**2)**.5
    return ypix, xpix, lam

def square_conv2(mov,lx):
    if len(mov.shape)<3:
        mov = mov[np.newaxis, :, :]
    nframes, Ly, Lx = mov.shape

    nx = len(lx)
    movt = np.zeros((nx,  nframes, Ly, Lx), 'float32')
    for j in range(nx):
        movt[j] = lx[j] * uniform_filter(mov, size=[0,lx[j], lx[j]], mode = 'constant')

    if nx==1:
        movt = movt[0]
    return movt

def sparsery(ops):
    rez, max_proj = get_mov(ops)
    ops['max_proj'] = max_proj
    nframes, Ly, Lx = rez.shape
    ops['Lyc'] = Ly
    ops['Lxc'] = Lx
    sdmov = get_sdmov(rez, ops)
    rez /= sdmov
    #rez *= -1

    lx = [ops['spatial_hp']]
    c1 = square_conv2(np.ones((Ly,Lx)), lx)
    movu = square_conv2(rez, lx)

    rez -= movu/c1

    LL = np.meshgrid(np.arange(Lx), np.arange(Ly))
    Lyp = np.zeros(5, 'int32')
    Lxp = np.zeros(5,'int32')
    gxy = [np.array(LL).astype('float32')]
    dmov = rez
    movu = []

    for j in range(5):
        movu.append(square_conv2(dmov, [3]))
        dmov = 2 * downsample(dmov)
        gxy0 = downsample(gxy[j], False)
        gxy.append(gxy0)
        nfr, Lyp[j], Lxp[j] = movu[j].shape
        movu[j] = np.reshape(movu[j], (nfr,-1))

    nfr, Lyc,Lxc = rez.shape
    V0 = []
    ops['Vmap']  = []
    for j in range(len(movu)):
        V0.append(np.amax(movu[j], axis=0))
        #V0.append(np.sum(movu[j]**2 * np.float32(movu[j]>Th2), axis=0)**.5)
        V0[j] = np.reshape(V0[j], (Lyp[j], Lxp[j]))
        ops['Vmap'].append(V0[j].copy())
    I = np.zeros((len(gxy), gxy[0].shape[1], gxy[0].shape[2]))
    for t in range(1,len(gxy)-1):
        gmodel = RectBivariateSpline(gxy[t][1,:,0], gxy[t][0, 0,:], ops['Vmap'][t])
        I[t] = gmodel.__call__(gxy[0][1,:,0], gxy[0][0, 0,:])
    I0 = np.amax(I, axis=0)
    ops['Vcorr'] = I0
    imap = np.argmax(I, axis=0).flatten()
    ipk = np.abs(I0 - maximum_filter(I0, size=(11,11))).flatten() < 1e-4
    isort = np.argsort(I0.flatten()[ipk])[::-1]
    im, nm = mode(imap[ipk][isort[:50]])
    if ops['spatial_scale'] > 0:
        im = max(1, min(4, ops['spatial_scale']))
        fstr = 'FORCED'
    else:
        fstr = 'estimated'

    if im==0:
        print('ERROR: best scale was 0, everything should break now!')
    Th2 = ops['threshold_scaling']*5*max(1,im)
    vmultiplier = max(1, np.float32(rez.shape[0])/1200)
    print('NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f '%(fstr, 3*2**im, vmultiplier, vmultiplier*Th2))
    ops['spatscale_pix'] = 3*2**im

    V0 = []
    ops['Vmap']  = []
    for j in range(len(movu)):
        #V0.append(np.amax(movu[j], axis=0))
        V0.append(np.sum(movu[j]**2 * np.float32(movu[j]>Th2), axis=0)**.5)
        V0[j] = np.reshape(V0[j], (Lyp[j], Lxp[j]))
        ops['Vmap'].append(V0[j].copy())
    I = np.zeros((len(gxy), gxy[0].shape[1], gxy[0].shape[2]))
    for t in range(1,len(gxy)-1):
        gmodel = RectBivariateSpline(gxy[t][1,:,0], gxy[t][0, 0,:], ops['Vmap'][t])
        I[t] = gmodel.__call__(gxy[0][1,:,0], gxy[0][0, 0,:])
    I0 = np.amax(I, axis=0)
    ops['Vcorr'] = I0


    xpix,ypix,lam = [],[],[]
    rez = np.reshape(rez, (-1,Ly*Lx))
    lxs = 3 * 2**np.arange(5)
    nscales = len(lxs)

    niter = 250 * ops['max_iterations']
    Vmax = np.zeros((niter))
    ihop = np.zeros((niter))
    vrat = np.zeros((niter))
    Npix = np.zeros((niter))

    t0 = tic()

    for tj in range(niter):
        v0max = np.array([np.amax(V0[j]) for j in range(5)])
        imap = np.argmax(v0max)
        imax = np.argmax(V0[imap])
        yi, xi = np.unravel_index(imax, (Lyp[imap], Lxp[imap]))
        yi, xi = gxy[imap][1,yi,xi], gxy[imap][0,yi,xi]

        Vmax[tj] = np.amax(v0max)
        if Vmax[tj] < vmultiplier*Th2:
            break
        ls = lxs[imap]

        ihop[tj] = imap

        ypix0, xpix0, lam0 = add_square(int(yi),int(xi),ls,Ly,Lx)
        xproj = rez[:, ypix0*Lx+ xpix0] @ lam0
        goodframe = np.nonzero(xproj>Th2)[0]
        for j in range(3):
            ypix0, xpix0, lam0 = iter_extend(ypix0, xpix0, rez, Ly,Lx, goodframe)
            xproj = rez[:, ypix0*Lx+ xpix0] @ lam0
            goodframe = np.nonzero(xproj>Th2)[0]
            if len(goodframe)<1:
                break
        if len(goodframe)<1:
            break
        vrat[tj], ipack = two_comps(rez[:, ypix0*Lx+ xpix0], lam0, Th2)
        if vrat[tj]>1.25:
            lam0, xp, goodframe = ipack
            xproj[goodframe] = xp
            ix = lam0>lam0.max()/5
            xpix0 = xpix0[ix]
            ypix0 = ypix0[ix]
            lam0 = lam0[ix]
        # update residual on raw movie
        rez[np.ix_(goodframe, ypix0*Lx+ xpix0)] -= xproj[goodframe][:,np.newaxis] * lam0
        # update filtered movie
        ys, xs, lms = multiscale_mask(ypix0,xpix0,lam0, Lyp, Lxp)
        for j in range(nscales):
            movu[j][np.ix_(goodframe,xs[j]+Lxp[j]*ys[j])] -= np.outer(xproj[goodframe], lms[j])
            #V0[j][xs[j] + Lxp[j]*ys[j]] = np.amax(movu[j][:,xs[j]+Lxp[j]*ys[j]], axis=0)
            Mx = movu[j][:,xs[j]+Lxp[j]*ys[j]]
            #V0[j][xs[j] + Lxp[j]*ys[j]] = np.sum(Mx**2 * np.float32(Mx>Th2), axis=0)**.5
            V0[j][ys[j], xs[j]] = np.sum(Mx**2 * np.float32(Mx>Th2), axis=0)**.5
            #V0[j][xs[j] + Lxp[j]*ys[j]] = np.sum(movu[j][:,xs[j]+Lxp[j]*ys[j]]**2 * np.float32(movu[j][:,xs[j]+Lxp[j]*ys[j]]>Th2), axis=0)**.5

        xpix.append(xpix0)
        ypix.append(ypix0)
        lam.append(lam0)
        if tj%1000==0:
            print('%d ROIs, score=%2.2f'%(tj, Vmax[tj]))
    #print(tj, time.time()-t0, Vmax[tj])
    ops['Vmax'] = Vmax
    ops['ihop'] = ihop
    ops['Vsplit'] = vrat
    stat  = [{'ypix':ypix[n], 'lam':lam[n]*sdmov[ypix[n], xpix[n]], 'xpix':xpix[n]} for n in range(len(xpix))]

    stat = get_stat(ops, stat)
    return ops,stat

def circleMask(d0):
    ''' creates array with indices which are the radius of that x,y point
        inputs:
            d0 (patch of (-d0,d0+1) over which radius computed
        outputs:
            rs: array (2*d0+1,2*d0+1) of radii
            dx,dy: indices in rs where the radius is less than d0
    '''
    dx  = np.tile(np.arange(-d0[1],d0[1]+1), (2*d0[0]+1,1))
    dy  = np.tile(np.arange(-d0[0],d0[0]+1), (2*d0[1]+1,1))
    dy  = dy.transpose()

    rs  = (dy**2 + dx**2) ** 0.5
    return rs, dx, dy

def get_stat(ops, stat):
    '''computes statistics of cells found using sourcery
    inputs:
        Ly, Lx, d0, mPix (pixels,ncells), mLam (weights,ncells), codes (ncells,nsvd), Ucell (nsvd,Ly,Lx)
    outputs:
        stat
        assigned to stat: ipix, ypix, xpix, med, npix, lam, footprint, compact, aspect_ratio, ellipse
    '''
    Ly = ops['Lyc']
    Lx = ops['Lxc']
    Ly = ops['Lyc']
    Lx = ops['Lxc']
    rs,dy,dx = circleMask(np.array([30, 30]))
    rsort = np.sort(rs.flatten())

    ncells = len(stat)
    mrs = np.zeros((ncells,))
    for k in range(0,ncells):
        stat0 = stat[k]
        ypix = stat0['ypix']
        xpix = stat0['xpix']
        lam = stat0['lam']
        # compute footprint of ROI
        y0 = np.median(ypix)
        x0 = np.median(xpix)

        # compute compactness of ROI
        r2 = ((ypix-y0))**2 + ((xpix-x0))**2
        r2 = r2**.5
        stat0['mrs']  = np.mean(r2)
        mrs[k] = stat0['mrs']
        stat0['mrs0'] = np.mean(rsort[:r2.size])
        stat0['compact'] = stat0['mrs'] / (1e-10+stat0['mrs0'])
        stat0['ypix'] += ops['yrange'][0]
        stat0['xpix'] += ops['xrange'][0]
        stat0['med']  = [np.median(stat0['ypix']), np.median(stat0['xpix'])]
        stat0['npix'] = xpix.size
        stat0['footprint'] = ops['ihop'][k]

    npix = np.array([stat[n]['npix'] for n in range(len(stat))]).astype('float32')
    npix /= np.mean(npix[:100])

    mmrs = np.nanmedian(mrs[:100])
    for n in range(len(stat)):
        stat[n]['mrs'] = stat[n]['mrs'] / (1e-10+mmrs)
        stat[n]['npix_norm'] = npix[n]
    stat = np.array(stat)
    return stat
