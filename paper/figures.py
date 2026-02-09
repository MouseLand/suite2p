from fig_utils import *
from scipy.stats import wilcoxon, ttest_rel
import os 
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.lines import Line2D
from scipy.stats import zscore
import cv2
import fastremap 
from cellpose import transforms

def pipeline_fig(gui_img, planes_img, raw_ex, reg_ex, max_proj, stat, iscell0, iperm, colors, masks, 
                 F, Fneu, masks_all, max_proj_all, Xemb, 
                 corr_starts, corr_ends, istims, running, isort):
    il = 0
    fig = plt.figure(figsize=(14,7), dpi=150)
    yratio = 14./7
    grid = plt.GridSpec(5, 6, wspace=0.3, hspace=0.4, figure=fig, 
                            bottom=0.0, top=0.99, left=0.03, right=0.99)
    
    by = 30
    n, ly, lx = raw_ex.shape
    img = 3 * np.ones((by*(n-1) + ly, by*(n-1) + lx), "float32")
    for i in range(n):
        img[by*i : by*i + ly, 
            by*(n-i-1) : by*(n-i-1) + lx] = raw_ex[i]
        
    ax = plt.subplot(grid[:2,0])
    ax.imshow(img, vmin=0, vmax=1, cmap="gray")
    ax.axis("off")
    ax.set_title('tiff, h5, nwb, etc inputs', fontstyle='italic')
    ax.text(0.85, -0.1, 'x10,000+ frames', transform=ax.transAxes, ha='right')
    transl = mtransforms.ScaledTranslation(-24/72, 3/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    Ly, Lx = reg_ex.shape[-2:]

    ylim = [50, 220]
    xlim = [250, 420]
    for j, fr in enumerate(reg_ex):
        ax = plt.subplot(grid[:2,j+1])
        rgb = np.zeros((Ly, Lx, 3))
        rgb[:,:,0] = fr[0]
        rgb[:,:,2] = fr[0]
        rgb[:,:,1] = fr[1]

        ax.imshow(rgb)
        ax.axis('off')
        ax.set_ylim(ylim)
        #ax.set_xlim([200, 500])
        ax.set_xlim(xlim)
        ax.text(1, -0.1, 'frame t', transform=ax.transAxes, color=[1,0,1], ha="right")
        ax.text(1, -0.2, 'frame t+1', transform=ax.transAxes, color=[0,1,0], ha="right")
        if j==0:
            ax.set_title('motion correction (> 100 frames per sec.)', fontstyle='italic',
                         loc='left')
            transl = mtransforms.ScaledTranslation(-18/72, 3/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)        
            

    alpha = 0.4
    masks0 = masks.copy().astype('float32')
    masks0[masks0==0] = np.nan
    ax = plt.subplot(grid[:2, 3])
    ax.imshow(max_proj, cmap='gray', vmin=2000, vmax=8000)
    ax.imshow(masks0, cmap='hsv', alpha=alpha, vmin=1, vmax=len(stat)+1)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.axis('off')
    ax.set_title('cell detection + extraction (< 10 min.)', fontstyle='italic',
                         loc='left')
    il = plot_label(ltr, il, ax, transl)        

    masks_filt = fastremap.mask(masks.copy(), np.nonzero(~(iscell0>0.5))[0]+1).astype('float32')
    masks_filt[masks_filt==0] = np.nan

    inds = np.unique(masks_filt[ylim[0] : ylim[1], ylim[0] : ylim[1]])[:-1].astype('int') - 1
    iinds = np.array([np.nonzero(iperm==i)[0][0] for i in inds])
    iinds = np.sort(iinds)[8:16]
    ax = plt.subplot(grid[:2, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]+0.5*(pos[3]-pos[2]*yratio), pos[2], pos[2]*yratio])
    for i, n in enumerate(iinds):
        Fi = F[n, 3000:7000].copy()
        Fneui = Fneu[n, 3000:7000].copy()
        Fneui = (Fneui - Fi.min()) / (Fi.max() - Fi.min())
        Fi = (Fi - Fi.min()) / (Fi.max() - Fi.min())
        ax.plot(Fneui + i*1.1, color=0.7*np.ones(3), lw=0.5)
        ax.plot(Fi + i*1.1, color=colors[n], lw=0.25)
    ax.axis('off')
    ax.set_xlim([0, len(Fi)])
    ax.set_ylim([0, len(iinds)*1.05])
    ax.text(1, -0.1, 'fluorescence traces', transform=ax.transAxes, ha='right')
    ax.text(1, -0.2, 'neuropil traces', color=0.7*np.ones(3), transform=ax.transAxes, ha='right')

    ax = plt.subplot(grid[:2, 5])
    ax.imshow(max_proj, cmap='gray', vmin=2000, vmax=8000)
    ax.imshow(masks_filt, cmap='hsv', alpha=alpha, vmin=1, vmax=len(stat)+1)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.axis('off')
    ax.set_title('soma classifier', fontstyle='italic',
                         loc='left')
    il = plot_label(ltr, il, ax, transl)        
    

    ax = plt.subplot(grid[-3:, :2])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.01, pos[1]+0.05, pos[2], pos[3]])
    ax.imshow(gui_img)
    ax.axis('off')
    ax.set_title('GUI for results visualization', fontstyle='italic',
                         loc='left')
    transl = mtransforms.ScaledTranslation(-17/72, 3/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid[-3:, 2:4],
                                                                wspace=0.2, hspace=0.2)
    ax = plt.subplot(grid1[0, 0])
    pos = ax.get_position().bounds
    px = 0.03
    ax.set_position([pos[0]-px, pos[1]+0.2*pos[3], pos[2], pos[3]*0.8])
    ax.imshow(planes_img)
    ax.set_title('> 100,000 neurons in 7 planes (@ 1.87 Hz)', fontstyle='italic',
                        loc='left')
    ax.axis('off')
    transl = mtransforms.ScaledTranslation(-17/72, 3/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    i = 0
    alpha = 0.7
    masks0 = masks_all[i].copy().astype('float32')
    masks0[masks0==0] = np.nan
    ncells = int(np.nanmax(masks0))
    ax0 = plt.subplot(grid1[1, 0])
    pos = ax0.get_position().bounds
    ax0.set_position([pos[0]-px, pos[1]+0.025, pos[2], pos[3]])
    ax0.imshow(masks0, cmap='hsv', alpha=1, vmin=1, vmax=ncells+1)
    Ly, Lx = masks0.shape
    x0, y0 = 580, 300
    dx, dy = 300, 550
    ax0.add_patch(Rectangle((x0, y0), dx, dy, edgecolor='k', facecolor='none'))
    ax0.axis('off')
    ax0.set_title(f'plane 1', fontsize='medium')
    pos = ax0.get_position().bounds
    
    ax = plt.subplot(grid1[:, 1])
    pos1 = ax.get_position().bounds
    ax.set_position([pos1[0] - px, *pos1[1:]])
    ax.imshow(masks0[y0:y0+dy, x0:x0+dx], cmap='hsv', alpha=1, vmin=1, vmax=ncells+1)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    pos1 = ax.get_position().bounds
    ax0.set_position([pos[0], pos1[1], pos[2], pos[3]])
    pos0 = ax0.get_position().bounds
    
    xf = (x0+dx) / masks0.shape[1]
    yf0 = (Ly-(y0+dy)) / masks0.shape[0] * (pos0[3]/pos1[3])
    yf1 = (Ly-y0) / masks0.shape[0]  * (pos0[3]/pos1[3])
    ax = fig.add_axes([pos0[0]+pos0[2]*xf, pos1[1], pos1[0]-pos0[0]-pos0[2]*xf, pos1[3]])
    ax.plot([0, 1], [yf0, 0], color='k')
    ax.plot([0, 1], [yf1, 1], color='k')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    

    #ax.text(1.1, 0.5, '...', fontsize='xx-large', transform=ax.transAxes)

    ax = plt.subplot(grid[-3:, -2:])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.0, pos[1]+0.08*pos[3], pos[2], pos[3]*0.84])
    nn = Xemb.shape[0]
    #nt = 1350 - 1180
    xmin = 6300
    xmax = xmin + 400
    nt = xmax - xmin
    im = ax.imshow(zscore(Xemb[:, xmin:xmax], axis=1), vmin=0, vmax=1., cmap="gray_r", aspect="auto")
    cols = plt.get_cmap('hsv')(np.linspace(0.1, 0.8, 6))
    icorr = (corr_starts >= xmin) | ((corr_ends >= xmin) & (corr_ends < xmax))
    cs = corr_starts[icorr]
    ce = corr_ends[icorr]
    ist = istims[:len(corr_starts)][icorr]
    cs = np.maximum(xmin, cs)
    ce = np.minimum(xmax, ce)
    for isti, csi, cei in zip(ist, cs, ce):
        ax.axvspan(csi - xmin, cei - xmin, color=cols[isti], alpha=0.3)
    ax.axis('off')
    # ax.plot([0, 20], 1.015*nn*np.ones(2), color='k')
    # ax.plot(-0.015*nt*np.ones(2), [nn, nn-100], color='k')
    ax.set_ylim([0, nn])
    ax.set_xlim([0, nt])
    axin = ax.inset_axes([-0.03, 0.8, 0.015, 0.2])
    plt.colorbar(im, cax=axin, orientation='vertical')
    axin.set_ylabel('z-scored activity')
    axin.yaxis.tick_left()
    axin.yaxis.set_label_position('left')

    #ax.invert_yaxis()
    axin = ax.inset_axes([-0.025, 0, 0.015, 1])
    axin.plot([0, 0], [0, 100], color='k', lw=2)
    axin.set_ylim([0, nn])
    axin.axis('off')
    axin.text(-0.25, 0, '10,000 neurons', transform=axin.transAxes, 
              rotation=90, ha='right', va='bottom')
    axin = ax.inset_axes([0, -0.025, 1, 0.015])
    axin.plot([0, 20], [0, 0], color='k', lw=2)
    axin.set_xlim([0, nt])
    axin.axis('off')
    axin.text(0.0, -0.05, '10 sec.', transform=axin.transAxes, ha='left', va='top')

    ax = fig.add_axes([pos[0]-0.0, pos[1]+0.93*pos[3], pos[2], pos[3]*0.08])
    ax.set_title('  Rastermap of activity - 6 corridor virtual reality', loc='left', fontsize='medium')
    ax.fill_between(np.arange(nt), np.maximum(0, running[xmin:xmax]), 
                    color=0.7*np.ones(3))
    ax.set_xlim([0, nt])
    ax.text(1, 1.15, 'running speed', color=0.7*np.ones(3), 
            transform=ax.transAxes, ha='right', va='top')
    ax.axis('off')
    transl = mtransforms.ScaledTranslation(-14/72, 3/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        


    return fig 
    

def suppfig_planes(max_proj_all, masks_all, Xemb, corr_starts, corr_ends, istims, running, isort):
    fig = plt.figure(figsize=(14,8), dpi=150, facecolor=None)
    grid = plt.GridSpec(2, 4, wspace=0.1, hspace=0.1, figure=fig, 
                            bottom=0.04, top=0.93, left=0.02, right=0.98)

    nplanes = 7
    for i in range(nplanes):
        alpha = 0.7
        masks0 = masks_all[i].copy().astype('float32')
        masks0[masks0==0] = np.nan
        ncells = int(np.nanmax(masks0))
        ax = plt.subplot(grid[i//4, i%4])
        ax.imshow(max_proj_all[i], cmap='gray', vmin=200, vmax=2500)
        ax.imshow(masks0, cmap='hsv', alpha=alpha, vmin=1, vmax=ncells+1)
        #ax.set_ylim(ylim)
        #ax.set_xlim(xlim)
        ax.axis('off')
        ax.set_title(f'plane {i}')

    return fig


def zstack_fig(imgs, shear_x=150, by=75, rsz=0.2):
    n = len(imgs)
    h, w = imgs[0].shape
    output_w = w + abs(shear_x)
    output_h = h
    zstack = np.nan * np.zeros((by*(n-1) + int(output_h * rsz), w + shear_x), 'float32')
    imgs_o = []
    for i, img in enumerate(imgs):
        src = (img.copy() * 100).astype('uint16')

        h, w = src.shape
        src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype('float32')
        dst_points = np.array([
            [shear_x, 0],
            [(w + shear_x), 0],
            [w, h*rsz],
            [0, h*rsz]
        ]).astype('float32')


        M = cv2.getPerspectiveTransform(src_points, dst_points)
        dst = cv2.warpPerspective(src, M, (int(output_w), int(output_h)))
        dst = dst[:int(h*rsz)]
        imgs_o.append(dst)

    for i in range(len(imgs_o)):
        ipos = imgs_o[i] > 0
        zstack[by*(n-i-1) : by*(n-i-1) + int(h*rsz)][ipos] = imgs_o[i][ipos]

    return zstack


def detection_fig(ylim, xlim, mov, mov_filt, v_map, ypix_all, xpix_all, lam_all, 
                  f_init, threshold, masks_all, mask_pic, mask_id, iou, traces, colors):
    
    fig = plt.figure(figsize=(14, 9), dpi=150)
    il = 0
    grid = plt.GridSpec(3, 7, wspace=0.25, hspace=0.3, figure=fig, 
                            bottom=0.015, top=0.945, left=0.02, right=0.98)
        
    for j, frand in enumerate([mov[:100, ylim[0]:ylim[1], xlim[0]:xlim[1]], 
                               mov_filt[:100, ylim[0]:ylim[1], xlim[0]:xlim[1]]]):
        Ly, Lx = frand.shape[1:]
        ly = Ly - 20
        lx = Lx - 20
        by = 75
        #iex = np.arange(0, len(frand), 10)
        iex = [10, 20, 30, 40]
        n = len(iex)
        img = 3 * np.ones((by*(n-1) + ly, by*(n-1) + lx), "float32")
        fmin, fmax = np.percentile(frand, 0.1), np.percentile(frand, 98)
        frand = np.clip((frand.astype("float32") - fmin) / (fmax - fmin), 0, 1)
        for i, ix in enumerate(iex):
            img[by*i : by*i + ly, 
                by*(n-i-1) : by*(n-i-1) + lx] = frand[ix, 10:10+ly, 10:10+lx]
            
        ax = plt.subplot(grid[0,j])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + j*0.03, pos[1], pos[2], pos[3]*1.1])
        
        ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        ax.axis("off")
        ax.set_title(["bin frames in time", 'high-pass filter in\nspace and time'][j], fontsize="medium")
        if j==0:
            ax.text(0, 1.17, "preprocess for detection", fontsize="large", 
                fontstyle="italic", transform=ax.transAxes)
            transl = mtransforms.ScaledTranslation(-20/72, 21/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)        

        ax.text(1, -0., "x 5,000", transform=ax.transAxes, ha="right")
        ax.annotate("", xy=(1.45, 0.5), xytext=(1.05, 0.5), 
                    xycoords=ax.transAxes, annotation_clip=False, 
                    textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
        ax.text(1.32, 0.6, "", transform=ax.transAxes, ha="center")
    
    for j in range(5):
        ax = plt.subplot(grid[0,j+2])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]+(4-j)*0.015, pos[1], pos[2], pos[3]])
        vmax = 500 / 4570
        im = ax.imshow(v_map[j][ylim[0]:ylim[1], xlim[0]:xlim[1]] / 4570,
                        cmap='seismic', vmin=-vmax, vmax=vmax)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'template {3 * 2**(j)}x{3 * 2**(j)}', fontsize='medium')
        if j==0:
            ax.text(0, 1.15, 'variance explained of initialization templates', 
                fontsize='large', fontstyle='italic', transform=ax.transAxes)
            transl = mtransforms.ScaledTranslation(-20/72, 20/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl) 
        elif j==4:
            cax = ax.inset_axes([0.45, -0.1, 0.5, 0.05])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')    
            cb.ax.set_xlim([0, vmax])

    ax = plt.subplot(grid[1, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.1*pos[2], pos[1]+0.1*pos[3], pos[2]*0.82, pos[3]*0.82])
    iex = 0
    ly = 40
    med = [np.median(ypix_all[iex][0]).astype(int), np.median(xpix_all[iex][0]).astype(int)]
    mask = np.zeros((ly, ly), 'float32')
    mask[ypix_all[iex][0].copy() - med[0] + ly//2, 
                xpix_all[iex][0].copy() - med[1] + ly//2] = lam_all[iex][0].copy()
    ax.imshow(mask, cmap='seismic', vmin=-mask.max(), vmax=mask.max())
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    transl = mtransforms.ScaledTranslation(-20/72, 21/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        
    ax.text(0, 1.1, 'initialization\ntemplate', 
                fontsize='large', fontstyle='italic', transform=ax.transAxes)
            
    ax = plt.subplot(grid[1, 1:3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.05*pos[2], pos[1]+0.05*pos[3], pos[2]*1.1, pos[3]*0.9])
    ax.plot(f_init[iex], lw=1, color='k')
    ax.plot([-80, len(f_init[iex])+80], [threshold, threshold], lw=1, color='r')
    ax.axis('off')
    transl = mtransforms.ScaledTranslation(-20/72, 1/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        
    ax.text(0, 1.03, 'initial time trace', 
                fontsize='large', fontstyle='italic', transform=ax.transAxes)
    ax.text(0.5, 0.96, 'threshold', transform=ax.transAxes, color='r')
    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 10, subplot_spec=grid[1, 3:],
                                                                wspace=0.2, hspace=0.1)
    for j, iex in enumerate([4, 11, 2330, 372]): #372, 514]):#, 22, 30, 37, 21]):
        med = [np.median(ypix_all[iex][0]).astype(int), np.median(xpix_all[iex][0]).astype(int)]
        ti = np.linspace(0, len(ypix_all[iex])-1, 10).astype(int)
        for i, t in enumerate(ti):
            mask = np.zeros((ly, ly), 'float32')
            yp = ypix_all[iex][t].copy() - med[0] + ly//2
            xp = xpix_all[iex][t].copy() - med[1] + ly//2
            igood = (yp >= 0) & (yp < ly) & (xp >= 0) & (xp < ly)
            yp = yp[igood]
            xp = xp[igood]
            lam = lam_all[iex][t][igood].copy()
            mask[yp, xp] = lam

            ax = plt.subplot(grid1[j, i])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1]+0.02, *pos[2:]])
            ax.imshow(mask, cmap='seismic', vmin=-mask.max(), vmax=mask.max())
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i==0 and j==0:
                transl = mtransforms.ScaledTranslation(-20/72, 5/72, fig.dpi_scale_trans)
                il = plot_label(ltr, il, ax, transl)        
                ax.text(0, 1.18, 'ROI refinement (4 examples)', 
                            fontsize='large', fontstyle='italic', transform=ax.transAxes)       
    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid[-1, :2],
                                                                wspace=0.2, hspace=0.1)
    Lyc, Lxc = mask_id.shape[1:]
    olims = [[np.nonzero(mask_id.sum(axis=(0,2)))[0][0], Lyc - np.nonzero(mask_id.sum(axis=(0,2))[::-1])[0][0]-1],
         [np.nonzero(mask_id.sum(axis=(0,1)))[0][0], Lxc - np.nonzero(mask_id.sum(axis=(0,1))[::-1])[0][0]-1]]
    olims[0][1] -= 20
    olims[1][0] += 0
    olims[1][1] -= 20
    print(olims)

    ax = plt.subplot(grid1[0, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.01, pos[1]+0.05, *pos[2:]])
    ax.imshow(mask_pic[olims[0][0]:olims[0][1], olims[1][0]:olims[1][1]])
    ax.axis('off')
    ax.set_title('example ROIs')
    transl = mtransforms.ScaledTranslation(-15/72, 5/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl) 
            
    
    ax = plt.subplot(grid1[0, 1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.03, pos[1]+0.05, *pos[2:]])
    ax.set_ylabel('ROIs')
    ax.set_xlabel('ROIs')
    im = ax.imshow(iou*100, cmap='plasma_r', vmin=0, vmax=100)
    ax.set_title('active frame overlap (%)')
    cax = ax.inset_axes([0.85, 0.45, 0.05, 0.5])
    cb = plt.colorbar(im, cax=cax)
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.colorbar(im)

    ax = plt.subplot(grid1[1, :])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.03, pos[1]+0.02, pos[2]*1.08, pos[3]])
    for i in range(len(traces)):
        ax.scatter(np.nonzero(traces[i, 2000:3000] > 0)[0], 
                i * np.ones((traces[i, 2000:3000] > 0).sum()), color=colors[i], 
                marker='|', lw=0.5, s=10)
    ax.set_title('active frames     ')
    ax.set_xlim([0, 1000])
    ax.set_ylim([-1, len(traces)])
    ax.invert_yaxis()
    ax.set_ylabel('ROIs')
    ax.set_xlabel('binned frames')
    ax.set_xticks([])
    ax.set_yticks([])

    
    for j in range(5):
        ax = plt.subplot(grid[-1,j+2])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]+(4-j)*0.015, pos[1], pos[2], pos[3]])
        masks0 = masks_all[j][ylim[0]:ylim[1], xlim[0]:xlim[1]].copy().astype('float32')
        masks0[masks0==0] = np.nan
        ax.imshow(masks0, cmap='hsv')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'template {3 * 2**(j)}x{3 * 2**(j)}', fontsize='medium')
        if j==0:
            ax.text(0, 1.15, 'detected ROIs, by initial template size', 
                fontsize='large', fontstyle='italic', transform=ax.transAxes)
            transl = mtransforms.ScaledTranslation(-18/72, 21/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl) 
        elif j>2:
            ax.text(0.5, 0.5, '(no ROIs\ndetected)', ha='center', va='center', transform=ax.transAxes)
                             
    return fig

def detectmetrics_fig(max_proj, cp_outlines, dF_gt, neu_ex, ell_ex, f_ex, 
                      d_out, masks_gt, outlines_gt, outlines_all, 
                      tps, fps, fns, idef):
    fig = plt.figure(figsize=(14,6), dpi=150)
    yratio = 6./14
    il = 0
    grid = plt.GridSpec(2, 5, wspace=0.25, hspace=0.3, figure=fig, 
                            bottom=0.055, top=0.925, left=0.02, right=0.98)

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=grid[0, :],
                                                                    wspace=0.3, hspace=0.1)
        
    f_ex = [np.clip(transforms.normalize99(f_ex[i], 0.1, 98), 0, 1) for i in range(len(f_ex))]
    print(f_ex[0].shape)
    zstack = zstack_fig(f_ex, rsz=0.5, shear_x=300, by=150)    
    ax = plt.subplot(grid1[0])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0] + 0.02, pos[1]-0.02, *pos[2:]])
    ax.imshow(zstack, cmap="gray")
    ax.axis('off')
    ax.set_title('recordings:\nRiboL1-jGCaMP8s\n (4 planes, 30μm spacing)', loc='left', x=-0.1)
    ax.text(-0.09, 1.47, 'Hybrid ground-truth generation', transform=ax.transAxes,
            fontsize='large', fontstyle='italic')
    transl = mtransforms.ScaledTranslation(-28/72, 48/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl) 
    
    ax.text(1.12, -0.1, "x 60,000\n  frames", transform=ax.transAxes, ha="right")
    ax.annotate("", xy=(1.5, 0.5), xytext=(1.02, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
    ax.text(1.26, 0.6, "maximum\nover time", transform=ax.transAxes, ha="center")

    ax = plt.subplot(grid1[1])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0] + 0.05, *pos[1:]])
    ax.imshow(transforms.normalize99(max_proj, 0.1, 98), cmap='gray', vmin=0, vmax=0.6)
    ax.axis('off')
    for outline in cp_outlines:
        ax.plot(outline[:, 0], outline[:, 1], color='y', lw=0.5, alpha=0.5, ls='-')
    ax.set_title('cellpose masks')

    # ax.annotate("", xy=(1.58, 0.5), xytext=(1.05, 0.5), 
    #             xycoords=ax.transAxes, annotation_clip=False, 
    #             textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
    # ax.text(1.32, 0.6, "extract activity", transform=ax.transAxes, ha="center")

    ax = plt.subplot(grid1[2])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0] + 0.015, pos[1]+0.07*pos[3], pos[2]*0.9, pos[3]*0.86])
    iexs = np.arange(0, len(dF_gt), 100)
    for i, iex in enumerate(iexs):
        f0 = dF_gt[iex, 3000:6000]
        f0 = (f0 - f0.min()) / (f0.max() - f0.min())
        ax.plot(f0 + i*0.9, lw=0.5, alpha=1, color='y')
    ax.axis('off')
    ax.set_ylim([-.1, len(iexs)*0.9 + 0.1])
    ax.set_title('activity traces')

    iex = [10, 30, 50, 70, 90][::-1]
    ly, lx = neu_ex.shape[1:]
    by = 25
    n = 5
    img = np.nan * np.ones((by*(n-1) + ly, by*(n-1) + lx), "float32")

    ax = plt.subplot(grid1[3])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0]-0.005, *pos[1:]])
    for i, ix in enumerate(iex):
        img[by*i : by*i + ly, 
            by*(n-i-1) : by*(n-i-1) + lx] = ell_ex[ix]
    ax.imshow(img, vmin=0, vmax=800, cmap='gray')
    ax.axis('off')
    ax.set_title('simulated\ndendrites/axons')
    #ax.text(0.25, 1.2, '(using activity from other planes)', fontsize='large',
    #        transform=ax.transAxes)
    ax.text(-0.14, 0.5, "+", transform=ax.transAxes, ha="center", fontsize='xx-large')
    ax.text(1.12, 0.5, "+", transform=ax.transAxes, ha="center", fontsize='xx-large')
    ax.text(0.84, -0.02, u'time \u2192', transform=ax.transAxes, rotation=45)

    ax = plt.subplot(grid1[4])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0]-0.015, *pos[1:]])
    for i, ix in enumerate(iex):
        img[by*i : by*i + ly, 
            by*(n-i-1) : by*(n-i-1) + lx] = neu_ex[ix]
    ax.imshow(img, vmin=0, vmax=1200, cmap='gray')
    ax.set_title('simulated\nneuropil')
    ax.axis('off')

    ax.annotate("", xy=(1.44, 0.5), xytext=(1.02, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
    ax.text(1.2, 0.54, "+ shot\nnoise", transform=ax.transAxes, 
            ha="center", fontstyle='italic')
    
    ax = plt.subplot(grid1[5])
    for i, ix in enumerate(iex):
        img[by*i : by*i + ly, 
            by*(n-i-1) : by*(n-i-1) + lx] = d_out[ix]
    ax.imshow(img, vmin=0, vmax=1500, cmap='gray')
    ax.set_title('simulated recording')
    ax.axis('off')

    
    ymax = 0.9

    ax = plt.subplot(grid[-1, 0])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0] - 0.02, pos[1]-0.04, pos[2]*1.2, pos[3]*1.2])
    Ly, Lx = masks_gt.shape
    rgb = np.ones((Ly, Lx, 3), 'float32')
    rgb[masks_gt > 0] = 0.9 * np.ones(3)
    rgb[outlines_gt > 0] = 0.8 * np.ones(3)
    ax.imshow(rgb)
    lss = ['-', '--', ':']
    for i in range(len(outlines_all)):
        for o in outlines_all[i]:
            oy = np.hstack((o[:,0], o[:1,0]))
            ox = np.hstack((o[:,1], o[:1,1]))
            plt.plot(oy, ox, color=alg_cols[i], lw=3, ls=lss[i])
    ax.set_ylim([80, 180])
    ax.set_xlim([250, 320])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('segmentation results')
    transl = mtransforms.ScaledTranslation(-15/72, 2/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl) 
    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[1, 1:3],
                                                                wspace=0.8, hspace=0.2)

    f1 = tps[0][:,:,:,idef] / (tps[0][:,:,:,idef] + 0.5 * (fns[0][:,:,:,idef] + fps[0][:,:,:,idef]))
    fnr = fns[0][:,:,:,idef] / (fns[0][:,:,:,idef] + tps[0][:,:,:,idef])
    fpr = fps[0][:,:,:,idef] / (fps[0][:,:,:,idef] + tps[0][:,:,:,idef])
    metrics = [f1, fns[0][:,:,:,idef], fps[0][:,:,:,idef]]
    ylabels = ['F1 score', 'false negatives', 'false positives']
    for k in range(3):
        ax = plt.subplot(grid1[k])
        pos = ax.get_position().bounds 
        ax.set_position([pos[0] - 0.012*k, pos[1] + 0.03, pos[2]*0.8, pos[3]])
        axin = ax.inset_axes([0, 1.0, 1, 0.1])
        ax.set_ylabel(ylabels[k], fontsize='medium')
        nalg = metrics[k].shape[-1]
        ax.set_xticks(np.arange(nalg))
        ax.set_xticklabels(['Suite2p', 'Caiman'], rotation=0)
        ax.plot(metrics[k].reshape(-1, 2).T, color=0.7*np.ones(3), lw=1)
        for i in range(nalg):
            ax.scatter(i*np.ones(12), metrics[k][:,:,i].flatten(), 
                       color=alg_cols[i], s=15, alpha=0.5, zorder=30)
            ax.get_xticklabels()[i].set_color(alg_cols[i])
            if i > 0:
                p = wilcoxon(metrics[k][:,:,0].flatten(), metrics[k][:,:,i].flatten()).pvalue 
                print(p)
                pstr = "n.s." if p > 0.05 else ("*" if p >= 0.01 else "**" if p >= 0.001 else "***")
                axin.plot([0, i], np.ones(2)*(0.95 + i * 0.015), lw=1, color="k")
                axin.text(i/2, 0.95 + i*0.015, pstr, ha="center", va="center")
        # p = wilcoxon(metrics[k][:,:,1].flatten(), metrics[k][:,:,2].flatten()).pvalue 
        # print(p)
        # pstr = "n.s." if p > 0.05 else ("*" if p >= 0.01 else "**" if p >= 0.001 else "***")
        # axin.plot([1, 2], np.ones(2)*(0.95 + 3 * 0.015), lw=1, color="k")
        # axin.text(1.5, 0.95 + 3*0.015, pstr, ha="center", va="center")
        ax.set_ylim([0, ymax] if k==0 else [0, 720])
        axin.axis('off')
        if k==0:
            transl = mtransforms.ScaledTranslation(-35/72, 2/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl) 
    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[1, -2:],
                                                                wspace=0.4, hspace=0.2)

    vars = ['# of dendrites', 'neuropil fraction', 'Poisson noise scale']
    xvars = [np.arange(0, 4001, 500), np.arange(0, 0.81, 0.1), 
            [0, 5, 10, 20, 50, 100, 200]]

    lss = ['--','-.',':']
    for i in range(3):
        ax = plt.subplot(grid1[i])
        pos = ax.get_position().bounds 
        ax.set_position([pos[0] -0.01, pos[1] + 0.03, pos[2]*1.05, pos[3]])
        for j in range(tps[i].shape[-2]):
            for m in range(3):
                f1s = tps[i][m,:,j] / (tps[i][m,:,j] + 0.5 * (fps[i][m,:,j] + fns[i][m,:,j]))
                ax.plot(xvars[i], f1s.T, 
                        color=alg_cols[j], lw=0.75, ls=lss[m])
        ax.set_ylim([0, ymax])
        ax.set_xlabel(vars[i])
        if i==2:
            ax.set_xticks([0, 100, 200])
        elif i==0:
            ax.set_ylabel('F1 score')
            handles = [Line2D([0], [0], color='k', linestyle=ls, markerfacecolor='none') for ls in lss]
            ax.legend(handles, ['mouse 1', 'mouse 2', 'mouse 3'], frameon=False,
                    handletextpad=0.2, handlelength=1.0,
                    bbox_to_anchor=(0.05, 0.02), loc='lower left', borderaxespad=0.)
            # transl = mtransforms.ScaledTranslation(-15/72, 5/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl) 
     
    return fig

def suppfig_detect(fns, fps):
    fig = plt.figure(figsize=(10,3), dpi=150)
    yratio = 3./10
    il = 0
    grid = plt.GridSpec(1, 6, wspace=0.5, hspace=0.3, figure=fig, 
                            bottom=0.17, top=0.86, left=0.05, right=0.98)

    vars = ['# of dendrites', 'neuropil fraction', 'Poisson noise scale']
    xvars = [np.arange(0, 4001, 500), np.arange(0, 0.81, 0.1), 
            [0, 5, 10, 20, 50, 100, 200]]

    lss = ['--','-.',':']
    ylabels = ['false negatives', 'false positives']
    for l in range(2):
        metric = fns.copy() if l==0 else fps.copy()
        for i in range(3):
            ax = plt.subplot(grid[i + 3*l])
            pos = ax.get_position().bounds 
            ax.set_position([pos[0] + 0.01*(2-i), pos[1], *pos[2:]])
            for j in range(metric[i].shape[-2]):
                for m in range(3):
                    ax.plot(xvars[i], metric[i][m,:,j].T, 
                            color=alg_cols[j], lw=0.75, ls=lss[m])
            ax.set_ylim([0, 800])
            ax.set_xlabel(vars[i])
            if i==2:
                ax.set_xticks([0, 100, 200])            
            elif i==0:
                ax.set_ylabel(ylabels[l])
                transl = mtransforms.ScaledTranslation(-50/72, 5/72, fig.dpi_scale_trans)
                il = plot_label(ltr, il, ax, transl) 
                if l==0:
                    handles = [Line2D([0], [0], color='k', linestyle=ls, markerfacecolor='none') for ls in lss]
                    ax.legend(handles, ['mouse 1', 'mouse 2', 'mouse 3'], frameon=False,
                            handletextpad=0.2, handlelength=1.0,
                            bbox_to_anchor=(0.02, 0.85), loc='lower left', borderaxespad=0.)   
            elif i==1:
                if l==0:
                    for j in range(2):
                        ax.text(0.05, 1.1-j*0.1, ['Suite2p', 'Caiman'][j], color=alg_cols[j], 
                                transform=ax.transAxes)
    return fig

def registration_fig(frand, freg, refImg, cc_ex, yoff, xoff, yblock, xblock, nblocks,
                     cc_nr_ex, cc_up_ex, yxup, u, v, tPC, regPC, regDX):
    
    fig = plt.figure(figsize=(14,7), dpi=150)

    grid = plt.GridSpec(3, 7, wspace=0.25, hspace=0.45, figure=fig, 
                            bottom=0.04, top=0.93, left=0.02, right=0.98)

    il = 0 
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=grid[0, :],
                                                                wspace=0.4, hspace=0.)
    Ly, Lx = frand.shape[1:]
    ly = Ly - 20
    lx = Lx - 20
    by = 75
    #iex = np.arange(0, len(frand), 10)
    iex = [0, 15, 28, 30]
    n = len(iex)
    img = 3 * np.ones((by*(n-1) + ly, by*(n-1) + lx), "float32")
    fmin, fmax = np.percentile(frand, 0.1), np.percentile(frand, 98)
    frand = np.clip((frand.astype("float32") - fmin) / (fmax - fmin), 0, 1)
    for i, ix in enumerate(iex):
        img[by*i : by*i + ly, 
            by*(n-i-1) : by*(n-i-1) + lx] = frand[ix, 10:10+ly, 10:10+lx]
        
    ax = plt.subplot(grid1[0,0])
    ax.imshow(img, vmin=0, vmax=1, cmap="gray")
    ax.axis("off")
    ax.set_title("random frames", fontsize="medium")
    ax.text(0, 1.18, "compute reference", fontsize="large", 
            fontstyle="italic", transform=ax.transAxes)
    ax.text(0.95, -0., "x 400", transform=ax.transAxes, ha="right")
    ax.annotate("", xy=(1.58, 0.5), xytext=(1.05, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
    ax.text(1.32, 0.6, "align to\neach other", transform=ax.transAxes, ha="center")
    transl = mtransforms.ScaledTranslation(-20/72, 16/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    ax = plt.subplot(grid1[0,1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.02, pos[1], pos[2], pos[3]])
    ref_img = refImg.copy()
    fmin, fmax = np.percentile(ref_img, 0.1), np.percentile(ref_img, 98)
    ref_img = (ref_img.astype("float32") - fmin) / (fmax - fmin)
    ref_img = np.clip(ref_img, 0, 1)
    ax.imshow(ref_img[10:-10, 10:-10], cmap="gray")
    ax.set_title("reference image", fontsize="medium")
    ax.axis("off")
    
    ax = plt.subplot(grid1[0,2])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.02, pos[1], pos[2], pos[3]])
    from suite2p.registration.utils import ref_smooth_fft
    import torch
    ref_img_w = torch.real(torch.fft.ifft2(ref_smooth_fft(torch.from_numpy(ref_img), smooth_sigma=0.85))).numpy()
    ref_img_w = ref_img_w[::-1][:,::-1]
    # ref_img_w = np.fft.fft2(ref_img)
    # ref_img_w /= np.abs(ref_img_w)
    # ref_img_w = np.real(np.fft.ifft2(ref_img_w))
    vmax = 1e-3
    ax.imshow(ref_img_w, cmap="gray", vmin=-vmax, vmax=vmax)
    ax.set_title("reference", fontsize="medium")
    ax.axis("off")
    ax.annotate("$\\circ$", xy=(1.62, 0.5), fontsize=30, xytext=(1.13, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, ha="center", va="center")
    ax.text(0, 1.18, "rigid registration", transform=ax.transAxes, fontsize="large",
            fontstyle="italic")
    ax.text(0.5, -0.1, "(whitened + smoothed)", transform=ax.transAxes, ha ="center")
    transl = mtransforms.ScaledTranslation(-20/72, 16/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    ax = plt.subplot(grid1[0,3])
    frand_w = np.fft.fft2(frand[0])
    frand_w /= np.abs(frand_w)
    frand_w = np.real(np.fft.ifft2(frand_w))
    
    vmax = 1e-4
    ax.imshow(frand_w, cmap="gray", vmin=-vmax, vmax=vmax)
    ax.set_title("example frame", fontsize="medium")
    ax.text(0.5, -0.1, "(whitened)", transform=ax.transAxes, ha ="center")
    ax.axis("off")
    ax.annotate("=", xy=(1.57, 0.5), fontsize=40, xytext=(1.22, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, ha="center", va="center")

    ax = plt.subplot(grid1[0,4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] - 0.03*pos[2], pos[1] + 0.1*pos[3], pos[2]*0.8, pos[3]*0.8])
    ry = cc_ex.shape[0]
    im = ax.imshow(cc_ex, cmap="magma")
    ax.scatter(ry//2, ry//2, marker="+", color=0.75*np.ones(3))
    ax.text(0.52, 0.55, "(0,0)", transform=ax.transAxes, color=0.75*np.ones(3))
    ax.axis("off")
    ax.set_title("phase-correlation", fontsize="medium")
    cax = ax.inset_axes([1.05, 0.75, 0.05, 0.25])
    cb = plt.colorbar(im, cax=cax)
    ax.plot([0, 20], -5*np.ones(2), "k")
    ax.text(10, -8, "20 px", ha="center", va="top")
    ax.invert_yaxis()

    tmin = 12000
    tmax = 15000
    ax = plt.subplot(grid1[0,5])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] - 0.1*pos[2], pos[1], pos[2]*1.1, pos[3]])
    ocols = ["b", "c"]
    for k in range(2):
        tr = yoff.copy() if k==0 else xoff.copy()
        tr = tr[tmin : tmax].astype("float32")
        tr -= tr.mean()
        ax.plot(tr, lw=1, alpha=0.75, color=ocols[k])
        ax.text(0+0.5*k, 1.05, f"{['y','x'][k]}-offsets", color=ocols[k],
                transform=ax.transAxes)
    y0 = -6.5
    tbar = 60 * 6.76
    ax.plot([-50, tbar], y0*np.ones(2), color="k")
    ax.text(-55, y0+1, "2 px", rotation=90, va="center", ha="right")
    ax.plot(-50*np.ones(2), [y0, y0+2], color="k")
    ax.text(260, y0-0.2, "1 min.", ha="center", va="top")
    ax.set_ylim([y0-0.25, 3])
    ax.set_xlim([-60, tmax-tmin])
    ax.axis("off")
    transl = mtransforms.ScaledTranslation(-20/72, 16/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=grid[1, :],
                                                                wspace=0.3, hspace=0.)
    ylim = [-20, Ly + 20]
    xlim = [-20, Lx + 20]
    ax = plt.subplot(grid1[0,0])
    ax.imshow(np.ones((Ly, Lx)), cmap="gray", vmin=0, vmax=1)
    bcols = plt.get_cmap("rainbow")(np.linspace(0.1, 1, len(yblock)))
    for j, (yb, xb) in enumerate(zip(yblock, xblock)):
        ax.add_patch(Rectangle((xb[0], yb[0]), xb[1] - xb[0], yb[1] - yb[0], 
                    facecolor=0.7*np.ones(3), edgecolor="none", alpha=0.25))
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("divide image into blocks", fontsize="medium")
    ax.text(-0.1, 1.2, "non-rigid registration", transform=ax.transAxes, fontsize="large",
            fontstyle="italic")
    ax.annotate("", xy=(1.55, 0.5), xytext=(1., 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
    transl = mtransforms.ScaledTranslation(-30/72, 16/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(*nblocks, subplot_spec=grid1[0, 1],
                                                                wspace=0.0, hspace=0.)
    cy, cx = cc_nr_ex.shape[1:]
    for j in range(nblocks[0]):
        for k in range(nblocks[1]):
            ax = plt.subplot(grid2[j, k])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]-k*0.005, pos[1], pos[3]*1.2, pos[3]*1.2])
            ax.imshow(cc_nr_ex[nblocks[1]*j + k], cmap="magma")
            ym, xm = np.unravel_index(cc_nr_ex[nblocks[1]*j + k].argmax(), (cy, cx))
            #ax.add_patch(Rectangle((-2, -2), cy+2, cx+2,
            #        facecolor="none", edgecolor=bcols[nblocks[1]*j+k], lw=1.5))
            ax.set_ylim([-3, cy+3])
            ax.set_xlim([-3, cx+3])
            ax.invert_yaxis()
            ax.axis("off")
            if j==0 and k==nblocks[1]//2:
                ax.set_title("blockwise phase-correlation", fontsize="medium")
            elif j==1 and k==4:
                x0 = 0.85
                y0 = 0.18
                x1 = 3
                ax.annotate("", xy=(x1, 0.7), xytext=(x0, 1-y0), 
                            xycoords=ax.transAxes, annotation_clip=False, 
                            textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="-"))
                ax.annotate("", xy=(x1, -1.55), xytext=(x0, y0), 
                            xycoords=ax.transAxes, annotation_clip=False, 
                            textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="-"))
            elif j==2 and k==4:
                x0 = 1.1
                dy = 2.2
                acol = 0.75 * np.ones(3)
                ax.annotate("", xy=(x0, dy+0.5), xytext=(x0, -dy+0.5), 
                            xycoords=ax.transAxes, annotation_clip=False, 
                            textcoords=ax.transAxes, arrowprops=dict(color=acol, lw=2,
                                                                      arrowstyle="<->"))
            elif j==4 and k==2:
                y0 = -0.1
                dx = 2.3
                ax.annotate("", xy=(dx+0.5, y0), xytext=(-dx+0.5, y0), 
                            xycoords=ax.transAxes, annotation_clip=False, 
                            textcoords=ax.transAxes, arrowprops=dict(color=acol, lw=2,
                                                                      arrowstyle="<->"))
                ax.text(0.5, y0-0.05, "adaptive smoothing", fontstyle="italic", color=acol, 
                        va="top", ha="center", transform=ax.transAxes)         

    j, k = 1, 4
    grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid1[0, 2],
                                                                    wspace=0.2, hspace=0.1)
    ax = plt.subplot(grid2[0])
    ax.imshow(cc_nr_ex[nblocks[1]*j + k], cmap="magma")
    ym, xm = np.unravel_index(cc_nr_ex[nblocks[1]*j + k].argmax(), (cy, cx))
    ucol = [0.3,0.8,0.3]#"r" #0.75*np.ones(3)
    print(ym, xm)
    zy = (cy//2 - ym)*10
    zx = (cy//2 - xm)*10
    ax.add_patch(Rectangle((xm-3.5, ym-3.5), 7, 7, facecolor="none", edgecolor=ucol, lw=2, ls="--"))
    ax.scatter(cy//2, cy//2, marker="+", color=0.75*np.ones(3), s=30)
    ax.axis("off")
    ax.set_title("kriging upsampling\nfor subpixel shifts", y=1.1, x=1, loc="center", 
                    fontsize="medium")

    ax = plt.subplot(grid2[1])
    ax.imshow(cc_up_ex[nblocks[1]*j + k], cmap="magma")
    cy, cx = cc_up_ex.shape[1:]
    ax.add_patch(Rectangle((-2, -2), cy+2, cx+2,
            facecolor="none", edgecolor=ucol, lw=2, ls="--"))
    ym, xm = np.unravel_index(cc_up_ex[nblocks[1]*j + k].argmax(), (cy, cx))
    ax.scatter(xm, ym, color=ucol, marker="x", s=30)
    ax.scatter(zx + cy//2, zy + cy//2, marker="+", color=0.75*np.ones(3), s=30)
    ax.text(0.3, 0.7, f"({u[j,k]:.1f}, {v[j,k]:.1f})", transform=ax.transAxes, 
            color=ucol, fontsize="small")
    ax.set_ylim([-3, cy+3])
    ax.set_xlim([-3, cx+3])
    ax.invert_yaxis()
    ax.axis("off")
    ax.annotate("", xy=(1.8, 0.5), xytext=(1.1, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))

    js, ks = np.meshgrid(np.arange(nblocks[0]), np.arange(nblocks[1]), indexing="ij")
    js = js.flatten() 
    ks = ks.flatten()
    ys = [(yb[1]-yb[0])/2 + yb[0] for yb in yblock]
    xs = [(xb[1]-xb[0])/2 + xb[0] for xb in xblock]
    ax = plt.subplot(grid1[0, 3])
    ax.imshow(np.ones((Ly, Lx)), cmap="gray", vmin=0, vmax=1)
    # bcols[nblocks[1]*js+ks]
    ax.quiver(xs, ys, v, -u, color="k", width=0.02, headaxislength=2,
            scale=8, headlength=2)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("block shifts", fontsize="medium")
    ax.annotate("", xy=(1.55, 0.5), xytext=(1., 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
    ax.text(1.28, 0.6, "bilinear\ninterpolation", transform=ax.transAxes, ha="center")

    iqy, iqx = 25, 40
    lymax = Ly #int(Ly*6/14)
    lxmax = Lx #int(Lx*5/8)
    print(lymax, lxmax)
    js, ks = np.meshgrid(np.arange(0, lymax, iqy), np.arange(0, lxmax, iqx), indexing="ij")
    vs = yxup[1][:lymax:iqy, :lxmax:iqx].flatten()
    us = yxup[0][:lymax:iqy, :lxmax:iqx].flatten()
    js = js.flatten()
    ks = ks.flatten()
    ax = plt.subplot(grid1[0, 4])
    # bcols = plt.get_cmap("rainbow")(np.linspace(0.1, 1, len(js)))
    ax.imshow(np.ones((Ly, Lx)), cmap="gray", vmin=0, vmax=1)
    ax.quiver(ks, js, vs, -us, color="k", width=0.01, headaxislength=1, headlength=1)
            #scale=10)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("per-pixel shifts", fontsize="medium")

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=grid[2,:],
                                                                    wspace=0.2, hspace=0.1)
    ax = plt.subplot(grid1[0, 0])
    img = 3 * np.ones((by*(n-1) + ly, by*(n-1) + lx), "float32")
    fmin, fmax = np.percentile(freg, 0.1), np.percentile(freg[:,10:10+ly, 10:10+lx], 98)
    freg = np.clip((freg - fmin) / (fmax - fmin), 0, 1)
    for i, ix in enumerate(iex):
        img[by*i : by*i + ly, 
            by*(n-i-1) : by*(n-i-1) + lx] = freg[ix, 10:10+ly, 10:10+lx]
    ax.imshow(img, vmin=0, vmax=1, cmap="gray")
    ax.axis("off")
    ax.text(0.7, -0., "x 2,000-5,000", transform=ax.transAxes, ha="left")
    ax.set_title("motion-corrected frames", fontsize="medium")
    ax.text(-0.1, 1.18, "registration metrics", transform=ax.transAxes, fontsize="large",
            fontstyle="italic")
    ax.annotate("", xy=(1.43, 0.5), xytext=(1.05, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, 
                textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
    ax.text(1.24, 0.55, "PCA", transform=ax.transAxes, ha="center")
    transl = mtransforms.ScaledTranslation(-30/72, 16/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    ax = plt.subplot(grid1[0, 1])
    for ic in range(3):
        pc = tPC[:300, ic]
        pc -= pc.min()
        pc /= pc.max()
        col = np.zeros(3) #(0.+0.4*ic)*np.ones(3)
        ax.plot(pc - 1.75*ic, color=col)
        ax.text(0, 1.1 - 1.75*ic, f"PC{ic+1}", color=col)
    ax.axis("off")
    ax.add_patch(Rectangle([-5., 0], len(pc)+10, 0.15, facecolor="b", 
                        edgecolor="none", alpha=0.35))
    ax.add_patch(Rectangle([-5., 0.83], len(pc)+10, 0.17, facecolor="r", 
                        edgecolor="none", alpha=0.35))
    # fs = 6.75; 2000 out of 32000 frames used = 16 frames per tPC
    ax.plot([0, 25], (-ic*1.75-0.2)*np.ones(2), color="k")
    ax.text(25/2, -ic*1.75-0.3, "1 min.", ha="center", va="top")


    grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid1[0, 2:4],
                                                                    wspace=0.1, hspace=0.1)
    PC = regPC[:,0].copy()
    fmin, fmax = np.percentile(PC, 0.1), np.percentile(PC, 98)
    PC -= fmin 
    PC /= (fmax - fmin)
    PC = PC[:, PC.shape[1]//2:, :-100]
    for j in range(2):
        ax = plt.subplot(grid2[0, j])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.01, *pos[1:]])
        ax.imshow(PC[j], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(["mean of top frames", "mean of bottom frames"][j], 
                    fontsize="medium",color=["r", "b"][j], alpha=0.35)
        if j==0:
            transl = mtransforms.ScaledTranslation(-20/72, 5/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)        

    ax = plt.subplot(grid2[0, 2])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.01, *pos[1:]])
    vmax = 0.4
    ax.imshow(PC[0] - PC[1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axis("off")
    ax.set_title("difference", fontsize="medium")

    ax = plt.subplot(grid1[0, -1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.25*pos[2], pos[1]+0.1*pos[3], 0.75*pos[2], pos[3]*0.9])
    cols = ["c", "g", "y"]
    #rstr = ["rigid", "nonrigid mean", "nonrigid max"]
    for j in range(1,2):
        ax.plot(regDX[:,j], color=cols[j])
        #ax.text(1, 0.7+0.12*j, rstr[j], color=cols[j], ha="right",
        #        transform=ax.transAxes)
    ax.set_ylim([-0.01, 0.1])
    ax.set_xlabel("PC index")
    ax.set_ylabel("registration offset,\ntop vs bottom (px)")
    ax.set_title("mean offset\nacross blocks", color="g", fontsize="medium")
    transl = mtransforms.ScaledTranslation(-70/72, 5/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)        

    return fig    

def suppfig_initial_ref(frames_mean, cc, imax, refImg0, refImg):
    fig = plt.figure(figsize=(14,3.5), dpi=150)
    grid = plt.GridSpec(1, 6, wspace=0.0, hspace=0.1, figure=fig, 
                                bottom=0.0, top=0.99, left=0.03, right=0.96)
    il = 0 
    transl = mtransforms.ScaledTranslation(-20/72, 16/72, fig.dpi_scale_trans)

    ax = plt.subplot(grid[0, 0])
    ax.imshow(frames_mean, cmap='gray', vmin=0, vmax=1300)
    ax.axis('off')
    ax.set_title('average of 400\n random frames')
    il = plot_label(ltr, il, ax, transl)        

    transl = mtransforms.ScaledTranslation(-40/72, 16/72, fig.dpi_scale_trans)

    ax = plt.subplot(grid[0, 1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.35*pos[2], *pos[1:]])
    ccp = cc.copy() - np.diag(np.diag(cc))
    cmax = 0.6
    #ccp = ccp[bestCC.argsort()][:, bestCC.argsort()]
    im = ax.imshow(ccp, vmin=-cmax, vmax=cmax, cmap='RdBu_r')
    ax.scatter(0, imax, marker='>', s=80, color='k')
    ax.annotate("", xy=(1.55, 0.5), xytext=(1.08, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, textcoords=ax.transAxes, 
                arrowprops=dict(color="k", arrowstyle="->"))
    ax.text(1.33, 0.55, "average top 20\nmost-correlated\nframes", transform=ax.transAxes, ha="center")
    ax.set_title('pairwise correlations of\n random frames')
    ax.set_xlabel('frames')
    ax.set_ylabel('frames')
    il = plot_label(ltr, il, ax, transl)        
    axin = ax.inset_axes([1.02, 0, 0.05, 0.25])
    plt.colorbar(im, cax=axin, orientation='vertical')
    axin.set_ylim([0, cmax])

    ax = plt.subplot(grid[0, 3])
    ax.imshow(refImg0, cmap='gray', vmin=refImg0.min(), vmax=900)
    ax.set_title('initial reference image')
    ax.axis('off')
    ax.annotate("", xy=(1.6, 0.5), xytext=(1.1, 0.5), 
                xycoords=ax.transAxes, annotation_clip=False, textcoords=ax.transAxes, 
                arrowprops=dict(color="k", arrowstyle="->"))
    ax.text(1.35, 0.55, "iteratively align\nrandom frames", transform=ax.transAxes, ha="center")

    ax = plt.subplot(grid[0, 5])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] - 0.26*pos[2], *pos[1:]])
    ax.imshow(refImg, cmap='gray', vmin=refImg.min(), vmax=1300)
    ax.set_title('final reference image')
    ax.axis('off')

    return fig


def regmetrics_fig(fr0, fr_20, regPCs, regDXs, tPCs, timings, alg_names):
    ylim = [450, 950]
    xlim = [180, 470]

    fr = fr0[:, ylim[0]:ylim[1], xlim[0]:xlim[1]].copy()
    fr_2 = fr_20[:, ylim[0]:ylim[1], xlim[0]:xlim[1]].copy()
    Ly, Lx = fr.shape[1:]
    ly = Ly
    lx = Lx
    by = 75
    print(ly, lx)
    iex = np.arange(0, len(fr), 10)
    #iex = [0, 15, 28, 30]

    fig = plt.figure(figsize=(14,6), dpi=150)
    
    il = 0
    
    grid = plt.GridSpec(2, 8, wspace=0.25, hspace=0.35, figure=fig, 
                        bottom=0.08, top=0.9, left=0.04, right=0.97)
        
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[0,:2],
                                                                        wspace=0.2, hspace=0.1)
    n = 4
    img = np.ones((2, by*(n-1) + ly, by*(n-1) + lx), "float32")
    fmin, fmax = np.percentile(fr, 0.1), np.percentile(fr, 98)
    fr = np.clip((fr - fmin) / (fmax - fmin), 0, 1)
    fmin, fmax = np.percentile(fr_2, 0.1), np.percentile(fr_2, 98)
    fr_2 = np.clip((fr_2 - fmin) / (fmax - fmin), 0, 1)
    for i, ix in enumerate(iex[:n]):
        img[0, by*i : by*i + ly, 
            by*(n-i-1) : by*(n-i-1) + lx] = fr[ix]
        img[1, by*i : by*i + ly, 
            by*(n-i-1) : by*(n-i-1) + lx] = fr_2[ix]
        

    for k in range(2):
        ax = plt.subplot(grid1[0, k])
        ax.imshow(img[k], vmin=0, vmax=1, cmap="gray")
        ax.axis("off")
        ax.text(0.75, 0.1, "x 2,000", transform=ax.transAxes, ha="left")
        ax.set_title(["motion-corrected in\nfunctional channel",
                    "shifts applied to\n anatomical channel"][k], fontsize="medium")
        if k==0:
            transl = mtransforms.ScaledTranslation(-40/72, 30/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)
            ax.text(-0.2, 1.25, "Benchmarking registration", transform=ax.transAxes, fontsize="large",
                fontstyle="italic")
        # ax.annotate("", xy=(1.43, 0.5), xytext=(1.05, 0.5), 
        #             xycoords=ax.transAxes, annotation_clip=False, 
        #             textcoords=ax.transAxes, arrowprops=dict(color="k", arrowstyle="->"))
        # ax.text(1.24, 0.55, "PCA", transform=ax.transAxes, ha="center")

    acols = ["g", [1, 0.5, 1], [0.5, 0, 1.0]]

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[0, 2:5],
                                                                        wspace=0.1, hspace=0.05)
    for k in range(3):
        ax = plt.subplot(grid1[0,k])
        pos = ax.get_position().bounds 
        ax.set_position([pos[0] + (1-k)*0.015, *pos[1:]])
        vmax = 1.5e3 if k!=1 else 4e3
        ax.imshow((regPCs[k,0,2,0,0] - regPCs[k,0,2,1,0]), cmap="RdBu_r",
                vmin=-vmax, vmax=vmax)
        ax.set_ylim(ylim); ax.set_xlim(xlim)
        ax.set_title(alg_names[k], color=acols[k], fontsize="medium")
        ax.axis("off")
        if k==0:
            transl = mtransforms.ScaledTranslation(-30/72, 20/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)
            ax.text(-0.1, 1.15, "PC1 difference (functional)", transform=ax.transAxes)

    ax = plt.subplot(grid[0,5])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0]-0.02, pos[1]+0.04, pos[2]*1.1, pos[3]*0.95])
    for k in range(3):
        pc = tPCs[k,0,2,:-50,0].copy()
        pc -= pc.min()
        pc /= pc.max()
        ax.plot(pc - 1.5*k, color=acols[k])
    ax.plot([0, 250], (-k*1.5-0.2)*np.ones(2), color="k")
    ax.text(250/2, -k*1.5-0.3, "10 min.", ha="center", va="top")
    ax.axis("off")
    ax.set_title("PC1 across time", fontsize="medium")
    transl = mtransforms.ScaledTranslation(-15/72, 5/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[0, 6:8],
                                                                        wspace=0.1, hspace=0.05)
    for k in range(2):
        ax = plt.subplot(grid1[0,k])
        pos = ax.get_position().bounds 
        ax.set_position([pos[0] + (1-k)*0.015, *pos[1:]])
        vmax = 1.5e3 if k!=1 else 4e3
        ax.imshow((regPCs[k,1,2,0,0] - regPCs[k,1,2,1,0]), cmap="RdBu_r",
                vmin=-vmax, vmax=vmax)
        ax.set_ylim(ylim); ax.set_xlim(xlim)
        ax.set_title(alg_names[k], color=acols[k], fontsize="medium")
        ax.axis("off")
        if k==0:
            transl = mtransforms.ScaledTranslation(-30/72, 20/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)
            ax.text(-0.1, 1.15, "PC1 difference (anatomical)", transform=ax.transAxes)

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[1, :5],
                                                                        wspace=0.6, hspace=0.6)

    for c in range(2):
        ax = plt.subplot(grid1[0,c])
        pos = ax.get_position().bounds 
        ax.set_position([pos[0] - 0.04*c + 0.02, pos[1]+0.02, *pos[2:]])
        for k in range(3):
            ax.semilogy(regDXs[k, c, :, :, -1].T, color=acols[k], lw=1)
        if c==0:
            transl = mtransforms.ScaledTranslation(-50/72, 5/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)
            ax.set_ylabel("registration offset ($\mu$m)")
        ax.set_xlabel("PC index")
        ax.set_yticks(10.**np.arange(-2, 2))
        ax.set_yticklabels(["0.01", "0.1", "1", "10"])
        ax.set_title(["functional channel", "anatomical channel"][c])
        ax.set_ylim([0.005, 15])
            

    for c in range(2):
        if c==0:
            ax = plt.subplot(grid1[0,2])
            pos = ax.get_position().bounds 
            ax.set_position([pos[0] - 0.02, pos[1]+0.02, 0.6*pos[2], pos[3]])
            pos = ax.get_position().bounds 
        else:
            ax = fig.add_axes([pos[0] + pos[2]*1.5, pos[1], pos[2]*2/3, pos[3]])
        # axin = ax.inset_axes([0, 1.0, 1, 0.1])
        for k in range(3):
            ax.scatter(k*np.ones(3) + np.array([-0.05, 0.05, -0.05]), 
                       regDXs[k,c,:,:,-1].max(axis=-1),
                    color=acols[k], marker='x', lw=2, facecolor=acols[k])
            # if k > 0:
            #     p = ttest_rel(regDXs[0,c,:,:,-1].max(axis=-1), 
            #                  regDXs[k,c,:,:,-1].max(axis=-1)).pvalue 
            #     print(p)
            #     pstr = "n.s." if p > 0.05 else ("*" if p >= 0.01 else "**" if p >= 0.001 else "***")
            #     axin.plot([0, k], np.ones(2)*(0.95 + k * 0.015), lw=1, color="k")
            #     axin.text(k/2, 0.95 + k*0.015, pstr, ha="center", va="center")
        ax.set_yscale("log")
        ax.set_yticks(10.**np.arange(-2, 2))
        ax.set_yticklabels(["0.01", "0.1", "1", "10"])
        ax.set_xlim([-0.5, (2+(c==0))-0.5])
        ax.set_xticks(np.arange(2+(c==0)))
        ax.set_xticklabels(alg_names[:(2+(c==0))], rotation=45, ha="right", va="top")
        for i, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(acols[i % 3])
        if c==0:
            ax.set_ylabel("registration offset max ($\mu$m)")
            transl = mtransforms.ScaledTranslation(-50/72, 5/72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl) 
        
        ax.set_title(["functional", "anatomical"][c])


    #grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[1, 6:],
    #                                                                    wspace=0.5, hspace=0.6)

    tframes = 500 * 2**np.arange(0,7)
    ax = plt.subplot(grid[1, -4:])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0] + 0.45*pos[2], pos[1]+0.02, pos[2] * 0.45, pos[3]])
    #xpos = [[0, 400, 800], [100, 700]]
    yh = [[0, 2, 4], [1, 3]]
    #ypos = [[33000, 16000, 8000], [33000, 12000]]
    for nr in range(2):
        rstr = "rigid" if nr==0 else "nonrigid"
        ustr = "\u2013" if nr==1 else "--"
        for k in range(3 - (nr==1)):
            ax.plot(timings[k, nr], tframes, color=acols[k], ls=["--", "-"][nr])
            tm = timings[k, nr].copy()
            a = ((tm - tm.mean()) / (tframes - tframes.mean())).mean()
            b = (tm - a * tframes).mean()
            print(a, b)
            hz = 1 / a
            xs = 280 if nr==0 and k==0 else 0 
            xs = 20 if nr==1 and k==0 else xs
            ax.text(tm.max() - xs, 33000, f'{int(np.round(hz))}Hz', color=acols[k], ha='left')
            if nr == 0:
                ax.text(1000, 15000-k*3000, alg_names[k], color=acols[k])

            # ax.text(1000, 18000 - yh[nr][k]*4000,
            #         f"{ustr} {alg_names[k]}, {rstr} = {int(np.round(hz))} Hz", 
            #         color=acols[k], va="center", fontsize="large")
        ax.text(1000, 15000-(3+nr)*3000, f'{ustr} {rstr}')
        
    ax.set_xlim([0, 1600])
    ax.set_ylim([0, 32000])
    ax.set_xlabel("runtime (sec.)")
    ax.set_ylabel("number of frames")
    transl = mtransforms.ScaledTranslation(-60/72, 5/72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)
        
    return fig




