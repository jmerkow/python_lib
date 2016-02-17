from __future__ import print_function
from helper_tools import *
import numpy as np
import scipy
from skimage.transform import resize
import PIL
import h5py
import SimpleITK as sitk


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def mirrorlr(img):
    for i in range(img.shape[-2]):
        img[...,i,:]=img[...,i,::-1]
    return img

def piecewise_window(x,wcenter=0,wwidth=200,wslope=.1):    
    return np.piecewise(x,
                    [x < wcenter-wwidth/2,
                     (wcenter-wwidth/2 <= x) & (x <= wcenter+wwidth/2),
                     x > wcenter+wwidth/2-wcenter],
                    [lambda x: wslope*x-wwidth/2+wcenter,
                     lambda x: x,
                     lambda x: wslope*x+wwidth/2])
def window_image(im,wcenter=0,wwidth=200,wslope=.1):
    return piecewise_window(im.flatten(),wcenter=wcenter,
                            wwidth=wwidth,wslope=wslope).reshape(im.shape)

def jtm_scale_croppad_window(image,newdims=None,scale=(.5,.5), padmode='edge',
                             wcenter=100,wwidth=300,wslope=.05):
    if newdims is None:
        raise ValueError('Need newdims')
        
    im=window_image(image,wcenter=wcenter,wwidth=wwidth,wslope=wslope)
    im=scipy.ndimage.interpolation.zoom(im, scale, order=1)
    
    im = np.array(im, dtype=np.float32)
    roi=(np.array(im.shape,dtype=np.float32)-newdims)
    pad=np.maximum(-roi,0)
    roi = np.array([r if np.mod(r,2)==0 else r+1 for r in roi])/2
    roi=(np.maximum(roi,0)).astype(int)
    if(sum(pad)):
#         print(pad,scale)
        pad2=[[p/2,p/2] if np.mod(p,2)==0 else [(p-1)/2,(p+1)/2] for p in pad]
        im=np.pad(im,pad2,mode=padmode)
    im=im[roi[0]:roi[0]+newdims[0],roi[1]:roi[1]+newdims[1]]
    
    return im

def jtm_resize(im, new_dims, interp_order=1):
    roi=np.array(im.shape,dtype=np.float32)-np.array(new_dims)
    im_min, im_max = im.min(), im.max()
    im_std = (im - im_min) / (im_max - im_min)
    resized_std = resize(im_std, new_dims, order=interp_order)
    resized_im = resized_std * (im_max - im_min) + im_min
    return resized_im.astype(np.float32)

def jtm_squash(image,newdims): 
    scale = tuple(np.array(newdims, dtype=float) / np.array(image.shape))
    im=scipy.ndimage.interpolation.zoom(image, scale, order=1) # squash
    
    return im

def jtm_croppad(im,newdims,padmode='edge'): 
    # print('jtm_croppad')
    im = np.array(im, dtype=np.float32)
    roi=(np.array(im.shape,dtype=np.float32)-newdims)
    pad=np.maximum(-roi,0)
    roi = np.array([r if np.mod(r,2)==0 else r+1 for r in roi])/2
    roi=(np.maximum(roi,0)).astype(int)
    if(sum(pad)):
        pad2=[[p/2,p/2] if np.mod(p,2)==0 else [(p-1)/2,(p+1)/2] for p in pad]
        im=np.pad(im,pad2,mode=padmode)
    
    im=im[roi[0]:roi[0]+newdims[0],roi[1]:roi[1]+newdims[1]]
    
    return im

def jtm_scale_croppad(image,newdims=None,scale=(.5,.5), padmode='edge'):
    if newdims is None:
        raise ValueError('Need newdims')
    
    im=scipy.ndimage.interpolation.zoom(image, scale, order=1)
    
    im = np.array(im, dtype=np.float32)
    roi=(np.array(im.shape,dtype=np.float32)-newdims)
    pad=np.maximum(-roi,0)
    roi = np.array([r if np.mod(r,2)==0 else r+1 for r in roi])/2
    roi=(np.maximum(roi,0)).astype(int)
    if(sum(pad)):
#         print(pad,scale)
        pad2=[[p/2,p/2] if np.mod(p,2)==0 else [(p-1)/2,(p+1)/2] for p in pad]
        im=np.pad(im,pad2,mode=padmode)
    im=im[roi[0]:roi[0]+newdims[0],roi[1]:roi[1]+newdims[1]]
    
    return im

def jtm_scale_croppad2(image,newdims=None,scale=(.5,.5), **kwargs):
    if newdims is None:
        raise ValueError('Need newdims')
    
    im=scipy.ndimage.interpolation.zoom(image, scale, order=1)
    
    im = np.array(im, dtype=np.float32)
    roi=(np.array(im.shape,dtype=np.float32)-newdims)
    pad=np.maximum(-roi,0)
    roi = np.array([r if np.mod(r,2)==0 else r+1 for r in roi])/2
    roi=(np.maximum(roi,0)).astype(int)
    if(sum(pad)):
#         print(pad,scale)
        pad2=[[p/2,p/2] if np.mod(p,2)==0 else [(p-1)/2,(p+1)/2] for p in pad]
        im=np.pad(im,pad2,**kwargs)
    im=im[roi[0]:roi[0]+newdims[0],roi[1]:roi[1]+newdims[1]]
    
    return im

def load_dcm2d(path,newdims=None,preproc=None,**preprocargs):
    img=sitk.ReadImage(path)
    isRGB=img.GetNumberOfComponentsPerPixel()>1
    image=sitk.GetArrayFromImage(img)
    image=np.squeeze(image)
    if isRGB:
        image=rgb2gray(image)
        raise
    if len(image.shape)>2:
        print(image.shape)
        raise
    if newdims is not None and preproc is None:
        scale = tuple(np.array(newdims, dtype=float) / np.array(image.shape))
#         print(scale)
        image=scipy.ndimage.interpolation.zoom(image, scale, order=1) # squash
    elif newdims is not None and preproc is not None:
        image=preproc(image,newdims,**preprocargs)
    return image
    
    


def load_image(path, newdims=None, mode='RGB', preproc=None,**preprocargs):
    """
    Load an image from disk
    Returns an np.ndarray (channels x width x height)
    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension
    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    if newdims is not None and preproc is None:
        image = scipy.misc.imresize(image, newdims, 'bilinear') # squash
    elif newdims is not None and preproc is not None:
        image=preproc(image,newdims,**preprocargs)
    return image

def count_lines(fn):
    return len(open(fn,'r').readlines())

def imlist(d,suf='.jpg',recurse=False):
    imlists=[]
    if suf is not None:
        imlists.append([pjoin(d,f) for f in os.listdir(d) if f.endswith(suf) and pexists(pjoin(d,f)) ])
    else:
        imlists.append([pjoin(d,f) for f in os.listdir(d)] and pexists(pjoin(d,f)) )
    if recurse:
        subd=[sd for sd in os.listdir(d) if os.path.isdir(pjoin(d,sd))]
        imlists.append(sum([imlist(pjoin(d,sd),suf=suf,recurse=recurse) for sd in subd],[]))

    return sum(imlists,[])


def load_imlst(imlist, newdims=None, mode='RGB', preproc=None,**preprocargs):
    return [load_image(fn,newdims=newdims,mode=mode,preproc=preproc,**preprocargs) for fn in imlist]

def labelimlist(fn,join=False):
    with open(fn) as f:
        _lst_ = [line.strip().strip(' ') for line in f]
        labels = [int(x.split(' ')[-1].strip()) for x in _lst_ if len(x.split(' '))>1]    
        im_list = [' '.join(x.split(' ')[:-1]).strip() for x in _lst_ if len(x.split(' '))>1]
    if join:
        return zip(im_list,labels)
    return (im_list,labels)

def calculate_image_mean(imlist,mode='RGB'): 
    mean = np.zeros(3).astype(np.float32)
    for imname in imlist:
        im = PIL.Image.open(imname)
        im = im.convert(mode)
        im = np.array(im)
        if len(im.shape) == 2:
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            mean = mean + im
        else:   
            im = im[:, :, ::-1] #change to BGR
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            mean = mean + im
            mean=mean
    mean /= len(imlist)
    return mean

def calculate_h5_mean(imlist,data='data'): 
    mean = None
    for imname in imlist:
        with h5py.File(imname, 'r') as f:
            im=np.squeeze(npa(f[data]))
        if len(im.shape) == 2:
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            if mean is None:
                mean = im
            else:
                mean = mean + im
        else:
            raise
    mean /= len(imlist)
    return mean

def write_h5_set(img_dict,h5_file,txtfile,metadata_dict=None):
    with h5py.File(h5_file, 'w') as f:
        for k,v in img_dict.iteritems():
            f[k] = npa(v).astype(float)
        if metadata_dict is not None:
            for k,v in metadata_dict.iteritems():
                f[k] = v
    with open(txtfile, 'a') as f:
        f.write(h5_file + '\n')


def calculate_mean_image_h5(train_source,data_key='data'):
    with open(train_source,'r') as f:
        files=[line.strip() for line in f]
    print(len(files));sys.stdout.flush()
    img_mean=None
    num_im=0
    for fil in files:
        with h5py.File(fil,'r') as f:
            img=npa(f[data_key])
        num_im+=1
        if img_mean is None:
            img_mean=img
        else:
            img_mean+=img
        
    img_mean=img_mean/num_im
    if len(img_mean.shape)==4:
        img_mean=img_mean[0,...]
    print(num_im)
    return img_mean


