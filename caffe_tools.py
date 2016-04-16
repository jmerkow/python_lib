from __future__ import print_function
import scipy.io, os, pandas, sys
import numpy as np
from PIL import Image
import time
import caffe
from caffe import layers as L, params as P
from caffe import net_spec
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import tempfile
from sklearn import metrics
from helper_tools import *
from image_tools import *
from dcm_lib import *
from collections import OrderedDict
import glob, re

from caffe_netspec_lib import *


def convert_camel(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

SP=caffe_pb2.SolverParameter()
solver_enum_dict={atrb:convert_camel(atrb) for atrb in dir(SP) if "EnumTypeWrapper" in str(type(getattr(SP,atrb)))}
solver_enum_dict2={convert_camel(atrb):atrb for atrb in dir(SP) if "EnumTypeWrapper" in str(type(getattr(SP,atrb)))}

def solverdict_from_file(solverFile):
    solver_proto=caffe_pb2.SolverParameter()
    text_format.Merge(open(solverFile).read(),solver_proto)
    return byteify(dict([(f[0].name,f[1]) for f in solver_proto.ListFields()]))

def solver_fixup(k,v):

    if k in solver_enum_dict2.keys():
        return getattr(SP,solver_enum_dict2[k]).keys()[v]
    elif k in solver_enum_dict.keys():
        return v
    else:
        if isinstance(v,bool):
            return str(v).lower()
        elif isinstance(v,str) or isinstance(v,unicode):
            v=v.encode('utf-8')
            v=v.strip('"')
            return '"'+v+'"'
        else:
            return(v)

class CaffeSolver(OrderedDict):
    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default values and can export a solver parameter file.
    Note that all parameters are stored as strings. For technical reasons, the strings are stored as strings within strings.
    """

    def __init__(self, d=None, default=True,
                testnet_prototxt_path = None, 
                trainnet_prototxt_path = None,
                debug = False, fn = None):
        OrderedDict.__init__(self)
        if d is not None:
            self.update(d)
        if fn is not None:
            self.add_from_file(fn)

        if d is None and fn is None and default:
            # critical:
            self['base_lr'] = 0.001
            self['momentum'] = 0.9

            # speed:
            self['test_iter'] = [100]
            self['test_interval'] = 250

            # looks:
            self['display'] = 25
            self['snapshot'] = 2500
            self['snapshot_prefix'] = 'snapshot' # string withing a string!

            # learning rate policy
            self['lr_policy'] = 'fixed'

            # important, but rare:
            self['gamma'] = 0.1
            self['weight_decay'] = 0.0005


            # pretty much never change these.
            self['max_iter'] = 100000
            self['test_initialization'] = False
            self['average_loss'] = 25 # this has to do with the display.
            self['iter_size'] = 1 #this is for accumulating gradients

            if (debug):
                self['max_iter'] = 12
                self['test_iter'] = [1]
                self['test_interval'] = 4
                self['display'] = 1

        if trainnet_prototxt_path is not None:   
            self['train_net'] = trainnet_prototxt_path

        if testnet_prototxt_path is not None:
                if hasattr(value,'__getitem__') and not isinstance(value,str):
                    self['test_net'] = testnet_prototxt_path
                else:
                    self['test_net'] = [testnet_prototxt_path]

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver instance parameters.
        """
        self.update(solverdict_from_file(filepath))


    def print(self, file=None):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        if file is not None:
            f=open(file, 'w')
        else:
            f=sys.stdout

        for key, value in sorted(self.items()):
            if hasattr(value,'__getitem__') and not isinstance(value,str):
                for v in value:
                    print(key,": ",solver_fixup(key,v), file=f ,sep='')
            else:
                print(key,": ",solver_fixup(key,value),file=f,sep='')
        
        if file is not None:
            f.close()

    def write(self,file):
        self.print(file=file)
    
def get_snapshots(snapstr):
    snapshots = glob.glob(os.path.join(snapstr))
    _iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in snapshots]
    return OrderedDict(sorted(zip(_iter,snapshots)))


def load_net(net_proto,caffeModel=None,mode=caffe.TRAIN):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    if caffeModel is None:
        return caffe.Net(f.name, mode)
    else:
        return caffe.Net(f.name, caffeModel, mode)

def load_proto(netFile):
    net_proto = caffe_pb2.NetParameter()
    text_format.Merge(open(netFile).read(),net_proto)
    return net_proto


def net_to_pic(net_proto):
    png = load_net_pic(net_proto)
    sio = StringIO()
    sio.write(png)
    sio.seek(0)
    return mpimg.imread(sio)


def run2(workdir=None, caffemodel = None, gpuid = 1, 
        solverFile = 'solver.prototxt', 
        log = None, logdir = None,
        snapshot_prefix = 'snapshot', debug=False,
        caffepath = None, restart = False, nbr_iters = None,
        do_run=False, byobu_rename=True, change_dir=True):


    """
    run is a simple caffe wrapper for training nets. It basically does two things. (1) ensures that training continues from the most recent model, and (2) makes sure the output is captured in a log file.
    Takes
    workdir: directory where the net prototxt lives.
    caffemodel: name of a stored caffemodel.
    solverFile: name of solver.prototxt [this refers, in turn, to net.prototxt]
    log: name of log file
    snapshot_prefix: snapshot prefix. 
    caffepath: path the caffe binaries. This is required since we make a system call to caffe.
    restart: determines whether to restart even if there are snapshots in the directory.
    """

    if debug:
        workdir_old=workdir
        workdir=pjoin(workdir,'dbg')
        solver=CaffeSolver(fn=solverFile)
        solver['snapshot_prefix']='snapshots/dbg_train'
#         solver['debug_info'] = True
        solver['display'] = 1
        solver['test_interval']=10
        solverFile = solverFile.replace(workdir_old,workdir)
        trained_fulldir=pjoin(workdir,'snapshots')
        if not pexists(workdir): mkdir(workdir)
        if not pexists(trained_fulldir): mkdir(trained_fulldir)
        print(solverFile)
        solver.print()
        solver.write(solverFile)
    # find initial caffe model
    if not caffemodel:
        caffemodel = glob.glob(os.path.join(workdir, "*initial.caffemodel"))
        if caffemodel:
            caffemodel = os.path.basename(caffemodel[0])

    solver=CaffeSolver(fn=solverFile)
    
    # finds the latest snapshots
    snapshots = glob.glob(os.path.join(workdir+'/', "{}*.solverstate".format(solver['snapshot_prefix'])))
    print()
    if snapshots:
        _iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in snapshots]
        max_iter = np.max(_iter)
        latest_snapshot = snapshots[np.argmax(_iter)]
    else:
        max_iter = 0

    # update solver with new max_iter parameter (if asked for)
    if nbr_iters is not None: 
        solver = CaffeSolver()
        solver.add_from_file(os.path.join(workdir, solverFile))
        solver['max_iter'] = str(max_iter + nbr_iters)
        solver['snapshot'] = str(1000000) #disable this, don't need it.
        solver.write(os.path.join(workdir, solverFile))
        solverFile=os.path.join(workdir, solverFile)
    
    if solverFile is not None:
        solver=CaffeSolver(fn=solverFile)
    # if
    
    if logdir is None:
        logdir=workdir
    
    if log is None:
        log = '{}_{}.{}.{:d}.log'.format(pbasename(os.path.split(solverFile)[0]),
                                        pbasename(solverFile).replace('solver_',''),max_iter,
                                       int(solver['max_iter']))
        log=os.path.join(logdir,log)
    if debug:
        log=''
    log2='caffe_log.log'

    if pexists(log):
        for n in range(0,100):
            if not pexists(log+'.'+str(n)):
                log=log+'.'+str(n)
                break

    if pexists(log2):
        for n in range(0,100):
            if not pexists(log2+'.'+str(n)):
                log2=log2+'.'+str(n)
                break

    byobu_name=pbasename(log)
    
    print ("caffepath",caffepath)
    print ("log",log)

    # by default, start from the most recent snapshot
    if snapshots and not(restart): 
        print ("Running {} from iter {}.".format(workdir, np.max(_iter)))
        runstring  = '{} train -solver {} -snapshot {} -gpu {} 2>&1 | tee {} {}'.format(caffepath, solverFile, latest_snapshot, gpuid, log, log2)

    # else, start from a pre-trained net defined in caffemodel
    elif(caffemodel): 
        if(os.path.isfile(os.path.join(workdir, caffemodel))):
            print ("Fine tuning {} from {}.".format(workdir, caffemodel))
            runstring  = '{} train -solver {} -weights {} -gpu {} 2>&1 | tee {} {}'.format(caffepath, solverFile, caffemodel, gpuid, log, log2)

        else:
            raise IOError("Can't fine intial weight file: " + os.path.join(workdir, caffemodel))

    # Train from scratch. Not recommended for larger nets.
    else: 
        print ("No caffemodel specified. Running {} from scratch!!".format(workdir))
        runstring  = '{} train -solver {} -gpu {} 2>&1 | tee {} {}'.format(caffepath, solverFile, gpuid, log, log2) 

    if change_dir:
        runstring="cd {}; ".format(workdir) + runstring 
    if byobu_rename:
        runstring="byobu rename-window {}; ".format(byobu_name) + runstring
    if do_run:
        os.system(runstring)
    else:
        print()
        print(runstring)

def run(workdir="./", caffemodel = None, gpuid = 1, 
        solverFile = 'solver.prototxt', 
        log = None,
        snapshot_prefix = 'snapshot', debug=False,
        caffepath = None, restart = False, nbr_iters = None,
        do_run=True, byobu_rename=True, change_dir=True):


    """
    run is a simple caffe wrapper for training nets. It basically does two things. (1) ensures that training continues from the most recent model, and (2) makes sure the output is captured in a log file.
    Takes
    workdir: directory where the net prototxt lives.
    caffemodel: name of a stored caffemodel.
    solverFile: name of solver.prototxt [this refers, in turn, to net.prototxt]
    log: name of log file
    snapshot_prefix: snapshot prefix. 
    caffepath: path the caffe binaries. This is required since we make a system call to caffe.
    restart: determines whether to restart even if there are snapshots in the directory.
    """

    if caffepath is None:
       raise ValueError("caffepath cannot be none")

    workdir=os.path.abspath(workdir)
    print()
    print("workdir:", workdir)
    if debug:
        workdir_old=workdir
        workdir=pjoin(workdir,'dbg')
        solver=CaffeSolver(fn=solverFile)
        solver['snapshot_prefix']='snapshots/dbg_train'
#         solver['debug_info'] = True
        solver['display'] = 1
        solver['test_interval']=10
        solverFile = solverFile.replace(workdir_old,workdir)
        trained_fulldir=pjoin(workdir,'snapshots')
        if not pexists(workdir): mkdir(workdir)
        if not pexists(trained_fulldir): mkdir(trained_fulldir)
        print(solverFile)
        solver.print()
        solver.write(solverFile)
    # find initial caffe model
    if not caffemodel:
        caffemodel = glob.glob(os.path.join(workdir, "*initial.caffemodel"))
        if caffemodel:
            caffemodel = os.path.basename(caffemodel[0])

    solver=CaffeSolver(fn=solverFile)
    
    # finds the latest snapshots
    snapshots = glob.glob(os.path.join(workdir+'/', "{}*.solverstate".format(solver['snapshot_prefix'])))
    if snapshots:
        _iter = [f[f.index('iter_')+5:f.index('.s')] for f in snapshots]
        _iter= map(int,_iter)
        max_iter = np.max(_iter)
        latest_snapshot = snapshots[np.argmax(_iter)]
    else:
        max_iter = 0
    # update solver with new max_iter parameter (if asked for)
    if nbr_iters is not None: 
        solver = CaffeSolver()
        solver.add_from_file(os.path.join(workdir, solverFile))
        solver['max_iter'] = str(max_iter + nbr_iters)
        solver['snapshot'] = str(1000000) #disable this, don't need it.
        solver.write(os.path.join(workdir, solverFile))
        solverFile=os.path.join(workdir, solverFile)
    
    if solverFile is not None:
        solver=CaffeSolver(fn=solverFile)
    # if
    logdir=workdir

    # if logdir is None:
    #     logdir=workdir
    
    # if log is None:
    #     log = '{}_{}.{}.{:d}.log'.format(pbasename(os.path.split(solverFile)[0]),
    #                                     pbasename(solverFile).replace('solver_',''),max_iter,
    #                                    int(solver['max_iter']))
    log=os.path.join(logdir,'caffe_log.log')

    if not debug:
        log=check_file(log)


    # if pexists(log):
    #     for n in range(0,100):
    #         if not pexists(log+'.'+str(n)):
    #             log=log+'.'+str(n)
    #             break

    # if pexists(log2):
    #     for n in range(0,100):
    #         if not pexists(log2+'.'+str(n)):
    #             log2=log2+'.'+str(n)
    #             break

    byobu_name=pbasename(workdir)
    
    print ("caffepath",caffepath)
    print ("log",log)
    print()

    # by default, start from the most recent snapshot
    if snapshots and not(restart): 
        print ("Running {} from iter {}.".format(workdir, np.max(_iter)))
        runstring  = '{} train -solver {} -snapshot {} -gpu {} 2>&1 | tee {}'.format(caffepath, solverFile, latest_snapshot, gpuid, log)

    # else, start from a pre-trained net defined in caffemodel
    elif(caffemodel): 
        if(os.path.isfile(os.path.join(workdir, caffemodel))):
            print ("Fine tuning {} from {}.".format(workdir, caffemodel))
            runstring  = '{} train -solver {} -weights {} -gpu {} 2>&1 | tee {}'.format(caffepath, solverFile, caffemodel, gpuid, log)

        else:
            raise IOError("Can't fine intial weight file: " + os.path.join(workdir, caffemodel))

    # Train from scratch. Not recommended for larger nets.
    else: 
        print ("No caffemodel specified. Running {} from scratch!!".format(workdir))
        runstring  = '{} train -solver {} -gpu {} 2>&1 | tee {}'.format(caffepath, solverFile, gpuid, log) 

    if change_dir:
        runstring="cd {}; ".format(workdir) + runstring 
    byoburen=''
    if byobu_rename:
        byoburen="byobu rename-window {}; ".format(byobu_name)
    print()
    print(byoburen)
    print(runstring)

    if do_run:
        os.system(byoburen)
    else:
        runstring=byoburen + runstring

    time.sleep(10)

    if do_run:
        os.system(runstring)


def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer
    Arguments:
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

# def backward_pass(scores, net, transformer, batch_size=1):
#     """
#     Returns scores for each image as an np.ndarray (nImages x nClasses)
#     Arguments:
#     images -- a list of np.ndarrays
#     net -- a caffe.Net
#     transformer -- a caffe.io.Transformer
#     Keyword arguments:
#     batch_size -- how many images can be processed at once
#         (a high value may result in out-of-memory errors)
#     """
#     net.blobs[net.outputs[-1]].diff[...] = scores
#     net.backward(

# #         print 'Processed %s/%s images ...' % (len(scores), len(caffe_images))

#     return scores

def forward_pass(images, net, transformer, batch_size=1,return_backprop=False):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    caffe_images = np.array(caffe_images)

    dims = transformer.inputs['data'][1:]

    scores,backprops = None,None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        output = net.forward()[net.outputs[-1]]
        if return_backprop:
                net.blobs[net.outputs[-1]].diff[...]=output[...]
                net.backward()
                back=net.blobs[net.inputs[0]].diff
                if backprops is None:
                    backprops = np.copy(back)
                else:
                    backprops = np.vstack((backprops, back))

        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))

#         print 'Processed %s/%s images ...' % (len(scores), len(caffe_images))
    if return_backprop:
        return scores,backprops

    return scores

def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net
    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

def classify(caffemodel, deploy_file, image_files, imgpreproc=None,
        mean_file=None, mean_val=0, labels_file=None, gpuid=0,batch_size=None,return_backprop=False,**preprocargs):
    """
    Classify some images against a Caffe model and print the results
    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images
    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    if gpuid>-1:
        caffe.set_mode_gpu()
        caffe.set_device(gpuid)
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    # net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    bs, channels, height, width = transformer.inputs['data']
    if batch_size is None:
        batch_size=bs
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = [load_image(image_file, (height, width), mode, imgpreproc,**preprocargs)-np.array(mean_val) for image_file in image_files]
    if labels_file:
        labels = read_labels(labels_file)

    # Classify the image
    classify_start_time = time.time()
    if return_backprop:
        scores,backprops = forward_pass(images, net, transformer,batch_size=batch_size,return_backprop=True)
    else:
        scores = forward_pass(images, net, transformer,batch_size=batch_size)
#     print ('Classification took %s seconds.' % (time.time() - classify_start_time,)
# )
    ### Process the results

    indices = (-scores).argsort()[:, :]
    if return_backprop:
        return indices,scores,backprops

    return indices,scores

def classify_dcm(image_files, caffemodel=None, deploy_file=None, imgpreproc=None,
        mean_file=None, mean_val=0, gpuid=0, batch_size=None, return_backprop=False,**preprocargs):
    """
    Classify some images against a Caffe model and print the results
    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images
    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    if caffemodel is None:
        raise ValueError('caffemodel cannot be none')
    if deploy_file is None:
        raise ValueError('deploy_file cannot be none')
    # Load the model and images
    if gpuid>-1:
        caffe.set_mode_gpu()
        caffe.set_device(gpuid)
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    # net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    bs, channels, height, width = transformer.inputs['data']
    if batch_size is None:
        batch_size=bs
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    # print(imgpreproc)
    # print(preprocargs);sys.stdout.flush()
    # preprocargs.pop('newdims')
    assert(all(a==b for a,b in zip((height, width),preprocargs['newdims'])))
    images = [load_dcm2d(image_file, preproc=imgpreproc,**preprocargs) for image_file in image_files]
    # images = [image-np.array(mean_val) for image in images if image is not None]

    # print(npa(images).mean())
    # print(npa(images).shape)
    # return images

    # Classify the image
    classify_start_time = time.time()
    if return_backprop:
        scores,backprops = forward_pass(images, net, transformer,batch_size=batch_size,return_backprop=False)
    else:
        scores = forward_pass(images, net, transformer,batch_size=batch_size)
#     print ('Classification took %s seconds.' % (time.time() - classify_start_time,)
# )
    ### Process the results

    indices = (-scores).argsort()[:, :]
    if return_backprop:
        return indices,scores,backprops

    return indices,scores


def classify_data(images, caffemodel=None, deploy_file=None, imgpreproc=None,
        mean_file=None, mean_val=0, gpuid=0, batch_size=None, return_backprop=False,**preprocargs):
    """
    Classify some images against a Caffe model and print the results
    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images
    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    if caffemodel is None:
        raise ValueError('caffemodel cannot be none')
    if deploy_file is None:
        raise ValueError('deploy_file cannot be none')
    # Load the model and images
    if gpuid>-1:
        caffe.set_mode_gpu()
        caffe.set_device(gpuid)
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    # net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    bs, channels, height, width = transformer.inputs['data']
    if batch_size is None:
        batch_size=bs
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    # print(imgpreproc)
    # print(preprocargs);sys.stdout.flush()
    # preprocargs.pop('newdims')
    # print(npa(images).mean())
    assert(all(a==b for a,b in zip((height, width),preprocargs['newdims'])))
    images = [preproc_image(image, preproc=imgpreproc,**preprocargs) for image in images]

    # print(npa(images).mean())
    # print(npa(images).shape)
    # return images

    # Classify the image
    classify_start_time = time.time()
    if return_backprop:
        scores,backprops = forward_pass(images, net, transformer,batch_size=batch_size,return_backprop=False)
    else:
        scores = forward_pass(images, net, transformer,batch_size=batch_size)
#     print ('Classification took %s seconds.' % (time.time() - classify_start_time,)
# )
    ### Process the results

    indices = (-scores).argsort()[:, :]
    if return_backprop:
        return indices,scores,backprops

    return indices,scores




def net_table(net_proto,proto=True):
    """
    net_table genererates a pandas table of the important information contained in a net proto.
    net_proto can also work on net files, if the proto flag is set to false.
    """
    param_dict=net_spec.param_name_dict()

    if not proto:
        net_proto=load_proto(net_proto)


    net = load_net(net_proto)
    layer_lst = list()
    layer_names=list()
    skipped_layers=list()
    loss_layers=list()
    for l,layer in enumerate(net_proto.layer):
        d=dict()
        param_type = layer.type
        d['type']=param_type
        use_bottom_scale = False
        layer_names.append(layer.name)
        d['bottoms']=', '.join([str(v) for v in layer.bottom._values])
        d['layer_index']=l
        d['name']=layer.name
        if not param_type in param_dict.keys():
            if param_type == 'Deconvolution':
                param_type = 'Convolution'
                use_bottom_scale = True
            elif 'Loss' in param_type:
                d['loss_weight']= 1 if not len(layer.loss_weight) else layer.loss_weight[0]
                layer_lst.append(d)
                continue
            else:
                skipped_layers.append(layer.name)
                continue
        top_name=layer.top._values[0]
        bottom_name = layer.bottom._values[0] if len(layer.bottom._values) else ''
        d['tops']=', '.join([str(v) for v in layer.top._values])
        d['param_type']=param_type
        d['param_name']=param_dict[param_type]+'_param'

        param = getattr(layer,param_dict[param_type]+'_param')

        if 'Loss' in param_type:
            d['loss_weight']= 1 if not len(layer.loss_weight) else layer.loss_weight[0]
        
        for i,param_sp in enumerate(layer.param._values):
            for field in param_sp.ListFields():
                fn=field[0].name
                if fn not in d.keys():
                    d[fn]=list()
                d[fn].append(field[1])
        
        for field in param.ListFields():
            fname=field[0].name
            ff = field[1]
            if 'weight_filler' in fname:
                ff = {each.split(':')[0].strip():try_float(each.split(':')[1].strip()) for each in str(ff).strip().split('\n')}.items()
            else:
                ff = ff[0] if hasattr(ff, '__iter__') and len(ff)==1 else ff
            d[fname] = ff
        
        try:
            d['output_size'] = net.blobs[top_name].data.shape[1:]
        except:
            d['output_size'] = ''

        param_shape = [p.data.shape for p in net.params[layer.name]] if layer.name in net.params.keys() else ''
        d['param_shape'] = param_shape
        d['total_params'] = np.sum([np.prod(p) for p in param_shape]) if param_shape is not '' else 0
        layer_lst.append(d)
    print("skipped:",', '.join([str(s) for s in skipped_layers]))
    layer_tab = pandas.DataFrame(layer_lst).replace(np.nan,' ', regex=True).set_index(['name'])
    return layer_tab

conv_cols=['param_shape','tops','bottoms','kernel_size','pad','num_output','output_size','lr_mult','decay_mult']
pool_cols=['kernel_size','stride','output_size','pad','pool']
loss_cols=['type','bottoms','loss_weight']
other_cols=[['bottoms','type','param_shape','output_size']]

 
