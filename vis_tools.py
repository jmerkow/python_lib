try:
    from ipywidgets import interact, interactive, widgets
    print("ipywidgets")
except:
    from IPython.html.widgets import interact, interactive
    from IPython.html import widgets
    print("ipy.html")


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm

from helper_tools import *

def normal_img(E):
    try:
        if (E.max()-E.min())>0:
            E = (E-E.min())/(E.max()-E.min()+1e-10)
            return E
        else:
            return E
    except:
        print E.shape
        return None
def whiten_img(E):
    try:
        return (E-E.mean())/E.std()
    except:
        print E.shape
        return None
def normal_img2(E,vmin=None,vmax=None):
    if vmin is None:
        vmin=E.min()
    if vmax is None:
        vmax=E.max()
    try:
        if (vmax-vmin)>0:
            E[E<vmin]=0
            E[E>vmax]=0
            E = (E-vmin)/(vmax-vmin+1e-10)
            return E
        else:
            return E
    except:
        print E.shape
        return None
        
def get_plots(numx,numy=1,margin=.05,ysize=100,xsize=100,dpi=20,figsize=None):
    
    if figsize is None:
        figsize = (1 + margin) * ysize*numy / dpi, (1 + margin) * xsize*numx / dpi

    return plt.subplots(numx,numy,figsize=figsize, dpi=dpi)

def tile_im(*images,**kwargs):
    num_im = len(images)
    nrows = kwargs.pop('nrows', None)
    ncols = kwargs.pop('ncols', None)
    if ncols is None and nrows is None:
        ncols = 1
    if nrows is None:
        nrows = int(np.ceil(num_im*1.0/ncols))
    if ncols is None:
        ncols = int(np.ceil(num_im*1.0/nrows))
    if nrows*ncols != len(images):
        raise ValueError("bad number of rows")    
    overlay = kwargs.pop('overlay', None)
    Z = np.zeros_like(images[0])
    nda = list()
    for c in range(ncols):
        nda.append(np.vstack(images[nrows*c:nrows*c+nrows]))
    return np.hstack(nda)
def imshow_list(*im_list, **kwargs):
    """
   This function displays a images lists with an IPython widget so the user can scroll through image sets.

   Params:
   Positional args are all assumed to be different image lists.  
   pred_label_list can be used to display text the sets.
   label_list can be used to display text for each image list.

   Example:
   imgs1=[np.random.rand(256,256) for n in range(0,10)]
   imgs2=[np.random.rand(256,256)-1 for n in range(0,10)]
   labels=["img "+str(n) for n in range(0,10)]
   imshow_list(imgs1,imgs2,pred_label_list=labels)

   """
    nrows = kwargs.pop('nrows', 1)
    label_list=kwargs.pop('label_list', None)
    pred_label_list=kwargs.pop('pred_label_list', None)
    true_label_list=kwargs.pop('true_label_list', None)
    overlay_list = kwargs.pop('overlay_list', None)
    title = kwargs.pop('title', None)
    margin = kwargs.pop('margin', 0.05)
    dpi = kwargs.pop('dpi', 80)
    overlay_color = kwargs.pop('overlay_color', 'red')
    overlay_alpha = kwargs.pop('overlay_alpha', .7)
    num_ims = len(im_list)
    num_ims_list = len(im_list[0])
    tile_shape = tile_im(*[np.squeeze(v[0]) for v in im_list],nrows=nrows).shape
    ncols=len(im_list)/nrows
    print(tile_shape)
    ysize = tile_shape[0]
    xsize = tile_shape[1]
    channel_size=tile_shape[2] if len(tile_shape)>2 else 0

    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
    ims = list()
    overlays = list()
    for i in range(num_ims_list):
        im_shapes=[np.squeeze(v[i]).shape for v in im_list]
        imi = [np.squeeze(v[i]) for v in im_list]
        ims.append(tile_im(*imi,nrows=nrows))
        if overlay_list is not None:
            ovi = [np.squeeze(overlay_list[i])]*num_ims
            ovi = tile_im(*ovi,nrows=nrows)
            overlays.append(np.ma.masked_where(ovi==0,ovi))
        else:
            overlays.append(None)
    def callback(i=None):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
        im = ims[i]
        im_shape=im_list[0][i].shape
        overlay = overlays[i]
        ax.imshow(im,interpolation=None)
        if overlay is not None:
#               ax.imshow(overlay,
#                             cmap=colors.ListedColormap([overlay_color]),
#                                       interpolation='hanning',alpha=overlay_alpha)
              ax.imshow(overlay,
                            cmap=overlay_color,
                                      interpolation='hanning',alpha=overlay_alpha)
        if title:
              plt.title(title)
        ax.set_xticklabels([]);ax.set_yticklabels([])
        ax.yaxis.set_ticks_position('none');ax.xaxis.set_ticks_position('none')
        if label_list is not None:
            for idx,t in enumerate(label_list):
                jj,ii=np.unravel_index(idx, (nrows,ncols),order='F')
                ax.text(float(im_shape[1])*ii+10, im_shape[0]*jj+20, t, style='italic',
                bbox={'facecolor':'gray', 'alpha':0.7,"boxstyle":"round"})
                
        if pred_label_list is not None:
            ax.text(10, 20, pred_label_list[i], style='italic',
            bbox={'facecolor':'red', 'alpha':0.7,"boxstyle":"round"})
    interact(callback,
          i=widgets.IntSlider(min=0,max=num_ims_list-1,step=1,value=0))
    plt.show()