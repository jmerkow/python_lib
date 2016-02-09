from __future__ import print_function
import scipy.io, Image, os, pandas, sys
import numpy as np

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
from collections import OrderedDict
import glob, re


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
            open(file, 'w').close()
        for key, value in sorted(self.items()):
            if file is not None:
                with open(file, 'a') as f:
                    if hasattr(value,'__getitem__') and not isinstance(value,str):
                        for v in value:
                            print(key,": ",solver_fixup(key,v), file=f ,sep='')
                    else:
                        print(key,": ",solver_fixup(key,value),file=f,sep='')
            else:
                if hasattr(value,'__getitem__') and not isinstance(value,str):
                    for v in value:
                        print(key,": ",solver_fixup(key,v),sep='')
                else:
                    print(key,": ",solver_fixup(key,value),sep='')
    def write(self,file):
        self.print(file=file)
    



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


def run(workdir=None, caffemodel = None, gpuid = 1, 
        solverFile = 'solver.prototxt', 
        log = None, logdir = None,
        snapshot_prefix = 'snapshot', 
        caffepath = None, restart = False, nbr_iters = None,
        do_run=False, byobu_rename=True, change_dir=False):


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

    # find initial caffe model
    if not caffemodel:
        caffemodel = glob.glob(os.path.join(workdir, "*initial.caffemodel"))
        if caffemodel:
            caffemodel = os.path.basename(caffemodel[0])

    # finds the latest snapshots
    snapshots = glob.glob(os.path.join(workdir, "{}*.solverstate".format(snapshot_prefix)))
    if snapshots:
        _iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in snapshots]
        max_iter = np.max(_iter)
        latest_snapshot = os.path.basename(snapshots[np.argmax(_iter)])
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

    if log is None:
        log = '{}_{}.0.{:d}.log'.format(pbasename(os.path.split(solverFile)[0]),
                                        pbasename(solverFile).replace('solver_',''),
                                       int(solver['max_iter']))
        log=os.path.join(logdir,log)
    
    if pexists(log):
        for n in range(0,100):
            if not pexists(log+'.'+str(n)):
                log=log+'.'+str(n)
                break

    byobu_name=pbasename(log)
    print (caffepath)
    print (log)
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
        runstring="cd {};".format(workdir) + runstring 
    if byobu_rename:
        runstring="byobu rename-window {}; ".format(byobu_name) + runstring
    if do_run:
        os.system(runstring)
    else:
        print(runstring)

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

def classify_dcm(caffemodel, deploy_file, image_files, imgpreproc=None,
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

def get_top(n,k):
    return n.__dict__['tops'][k]
def get_bottom(n):
    return n.__dict__['tops'][n.__dict__['tops'].keys()[-1]]
def dict2net(n,ns):
    for k,v in ns.iteritems():
        setattr(n,k,v)
    return get_bottom(n)

def max_pool(bottom, ks=2, stride=2,pad=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride,pad=pad)

def ave_pool(bottom, ks=2, stride=2,pad=1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride,pad=pad)

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, learn=True, 
              param=None, wf=None,bf=None):
    if learn and param is None:
        param = [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    
    if wf is None:
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, param=param)
    else:
        if bf is None:
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                num_output=nout, pad=pad, param=param,weight_filler=wf)
        else:
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=param,
                                 weight_filler=wf,bias_filler=bf)
    return conv, L.ReLU(conv, in_place=True)

def conv_bn_relu(bottom, nout, ks = 3, stride=1, pad = 0, learn = True, 
                 param=None, bn_learn=False, wf=None, bf=None, bn_first=True, bn_global=False):
    if learn and param is None:
        param = [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
        
    if bn_learn:
        #bn params:  mean=0, var=1, scale=2
        bn_learn = [dict(lr_mult=1.0, decay_mult=1.0),]*bn_learn+ [dict(lr_mult=0, decay_mult=0),]*(3-bn_learn)
    else:
        bn_learn = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)]
    
    if wf is None:
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, param=param)
    else:
        if bf is None:
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                num_output=nout, pad=pad, param=param,weight_filler=wf)
        else:
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=param,
                               weight_filler=wf,bias_filler=bf)
    if bn_first:
        bn = L.BatchNorm(conv, param=bn_learn)
        relu = L.ReLU(bn, in_place=True)
    else:
        relu = L.ReLU(conv, in_place=True)
        bn = L.BatchNorm(relu, param=bn_learn)
        
    return conv, bn, relu

def ip_relu(bottom, nout, learn=True, param=None,
              wf=None,bf=None):
    if learn and param is None:
        param = [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    
    if wf is None:
        ip = L.InnerProduct(bottom, num_output=nout, param=param)
    else:
        if bf is None:
            ip = L.InnerProduct(bottom, num_output=nout, 
                                param=param, weight_filler=wf)
        else:
            ip = L.InnerProduct(bottom,num_output=nout, 
                                param=param, weight_filler=wf, bias_filler=bf)
    return ip, L.ReLU(ip, in_place=True)

def ip_bn_relu(bottom, nout, learn=True, param=None,  bn_learn=False,
              wf=None, bf=None, bn_first=True):
    if learn and param is None:
        param = [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    
    if bn_learn:
        #bn params:  mean=0, var=1, scale=2
        bn_learn = [dict(lr_mult=1.0, decay_mult=1.0),]*bn_learn + [dict(lr_mult=0, decay_mult=0),]*(3-bn_learn)
    else:
        bn_learn = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)]
    
    if wf is None:
        ip = L.InnerProduct(bottom, num_output=nout, param=param)
    else:
        if bf is None:
            ip = L.InnerProduct(bottom, num_output=nout, 
                                param=param, weight_filler=wf)
        else:
            ip = L.InnerProduct(bottom,num_output=nout, 
                                param=param, weight_filler=wf, bias_filler=bf)
    
    if bn_first:
        bn = L.BatchNorm(ip, param=bn_learn)
        relu = L.ReLU(bn, in_place=True)
    else:
        relu = L.ReLU(ip, in_place=True)
        bn = L.BatchNorm(relu, param=bn_learn)
        
    return ip, bn, relu

def add_loss(n,s,bottom,label,losslayer=L.SoftmaxWithLoss,**lossargs):
    return dict2net(n,{s:losslayer(bottom,label,**lossargs)})
def add_acc(n,s,bottom,label,top_k=None):
    if top_k is None:
        dict2net(n,{s:L.Accuracy(bottom,label)})
    else:
        dict2net(n,{s+'-top-'+str(top_k):L.Accuracy(bottom,label,top_k=top_k)})     


def inception_unit_bn(n,s,bottom=None,
                   nouts={'1x1':64,'3x3':64,'5x5':32,'poolproj':32},
                   reduces={'3x3':96,'5x5':16}):
    
    if bottom is None:
        bottom = get_bottom(n)
    
    ns=OrderedDict()
    concat=[]
    # 1x1
    ns[s+'/1x1'],ns[s+'/bn_1x1'], ns[s+'/relu_1x1'] = conv_bn_relu(bottom,nouts['1x1'],ks=1,pad=0,
                                               wf=dict(type="xavier"))
    concat.append(ns[s+'/relu_1x1'])
    
    #3x3
    ns[s+'/3x3_reduce'],ns[s+'/bn_3x3_reduce'],ns[s+'/relu_3x3_reduce'] = conv_bn_relu(bottom,reduces['3x3'],ks=1,pad=0,
                                               wf=dict(type="xavier"))
    ns[s+'/3x3'],ns[s+'/bn_3x3'],ns[s+'/relu_3x3'] = conv_bn_relu(ns[s+'/relu_3x3_reduce'],nouts['3x3'],ks=3,pad=1,
                                               wf=dict(type="xavier"))
    concat.append(ns[s+'/relu_3x3'])
    
    #5x5
    ns[s+'/5x5_reduce'],ns[s+'/bn_5x5_reduce'],ns[s+'/relu_5x5_reduce'] = conv_bn_relu(bottom,reduces['5x5'],ks=1,pad=0,
                                               wf=dict(type="xavier"))
    ns[s+'/5x5'],ns[s+'/bn_5x5'],ns[s+'/relu_5x5'] = conv_bn_relu(ns[s+'/relu_5x5_reduce'],nouts['5x5'],ks=5,pad=2,
                                               wf=dict(type="xavier"))
    concat.append(ns[s+'/relu_5x5'])
    
    #max
    ns[s + '/pool'] = max_pool(bottom,ks=3,stride=1,pad=1)
    ns[s+'/pool_proj'],ns[s+'/bn_pool_proj'],ns[s+'/relu_pool_proj'] = conv_bn_relu(ns[s + '/pool'],nouts['poolproj'],ks=1,pad=0,
                                               wf=dict(type="xavier"))
    concat.append(ns[s+'/relu_pool_proj'])

    #concat
    ns[s + '/output'] = L.Concat(*concat)
    
    return dict2net(n,ns)


def watchtower_bn(n,bottom,s,nclass,nconv=128,nfc=1024,dropout=0.2,
    conv_lr=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
    fc_lr=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
    ip_lr=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)]):
    
    ns=dict()
    ns[s+'/ave_pool'] = ave_pool(bottom, ks=5, stride=3, pad=0)
    ns[s+'/conv'],ns[s+'/bn_conv'],ns[s+'/relu_conv'] = conv_bn_relu(ns[s+'/ave_pool'],
                                                 nconv,ks=1,pad=0,param=conv_lr,
                                                 wf=dict(type="xavier"))                                           
    ns[s+'/fc'],ns[s+'/bn_fc'],ns[s+'/relu_fc'] = ip_bn_relu(ns[s+'/relu_conv'],
                                                 nfc,param=fc_lr,
                                                 wf=dict(type="xavier"))
    ns[s+'/drop_fc'] = L.Dropout(ns[s+'/relu_fc'],in_place=True,dropout_ratio=dropout)
    ns[s+'/classifier'] = L.InnerProduct(ns[s+'/drop_fc'],num_output=nclass,
                                             param= ip_lr,
                                             inner_product_param=dict(weight_filler=dict(type="xavier",std=0.0009765625),
                                                                     bias_filler=dict(type="constant",value=0.0))
                                             ) 
    
    dict2net(n,ns)
    return ns[s+'/classifier']

def inception_vanila_table():
    arch_table=OrderedDict()
    arch_table['inception_3a']={
        'nouts':{'1x1':64,'3x3':128,'5x5':32,'poolproj':32},
        'reduces':{'3x3':96,'5x5':16}}
    arch_table['inception_3b']={
        'nouts':{'1x1':128,'3x3':192,'5x5':96,'poolproj':64},
        'reduces':{'3x3':128,'5x5':32}}

    arch_table['inception_4a']={
        'nouts':{'1x1':192,'3x3':208,'5x5':48,'poolproj':64},
        'reduces':{'3x3':96,'5x5':16}}
    arch_table['inception_4b']={
        'nouts':{'1x1':160,'3x3':224,'5x5':64,'poolproj':64},
        'reduces':{'3x3':112,'5x5':24}}
    arch_table['inception_4c']={
        'nouts':{'1x1':128,'3x3':256,'5x5':64,'poolproj':64},
        'reduces':{'3x3':128,'5x5':24}}
    arch_table['inception_4d']={
        'nouts':{'1x1':112,'3x3':288,'5x5':64,'poolproj':64},
        'reduces':{'3x3':144,'5x5':32}}
    arch_table['inception_4e']={
        'nouts':{'1x1':256,'3x3':320,'5x5':128,'poolproj':128},
        'reduces':{'3x3':160,'5x5':32}}

    arch_table['inception_5a']={
        'nouts':{'1x1':256,'3x3':320,'5x5':128,'poolproj':128},
        'reduces':{'3x3':160,'5x5':32}}
    arch_table['inception_5b']={
        'nouts':{'1x1':384,'3x3':384,'5x5':128,'poolproj':128},
        'reduces':{'3x3':192,'5x5':48}}

    return arch_table

def inception_vanilla_bn(n):
    conv1 = dict2net(n,dict(zip(['conv1/7x7','conv1/bn_7x7','conv1/relu_7x7'],
                            conv_bn_relu(n.data,64,ks=7,pad=3,stride=2,
                                      wf=dict(type="xavier")))))

    pool_1 = dict2net(n,{'pool1/3x3_s2':max_pool(get_top(n,'conv1/relu_7x7'),ks=3,stride=2,pad=0)})
    pooln1 = dict2net(n,{'pool1/norm1':L.LRN(pool_1,local_size=5,alpha=0.0001,beta=0.75)})

    conv2r = dict2net(n,dict(zip(['conv2/3x3_reduce','conv2/bn_3x3_reduce','conv2/relu_3x3_reduce'], 
                                 conv_bn_relu(pooln1,64,ks=1,pad=0,
                                           wf=dict(type="xavier")))))
    conv2 = dict2net(n,dict(zip(['conv2/3x3','conv2/bn_3x3','conv2/relu_3x3'],
                                conv_bn_relu(get_top(n,'conv2/relu_3x3_reduce'),192,ks=3,pad=1,
                                          wf=dict(type="xavier")))))

    pool_2= dict2net(n,{'pool2/3x3_s2':max_pool(get_top(n,'conv2/relu_3x3'),ks=3,stride=2,pad=0)})



    incp3a = inception_unit_bn(n,'inception_3a',**arch_table['inception_3a'])
    incp3b = inception_unit_bn(n,'inception_3b',**arch_table['inception_3b'])
    pool_3 = dict2net(n,{'pool3/3x3_s2':max_pool(get_bottom(n),ks=3,stride=2,pad=0)})
    incp4a = inception_unit_bn(n,'inception_4a',**arch_table['inception_4a'])
    clas_1 = watchtower_bn(n,incp4a,'loss1',nclass)

    incp4b = inception_unit_bn(n,'inception_4b',bottom=incp4a,**arch_table['inception_4b'])
    incp4c = inception_unit_bn(n,'inception_4c',**arch_table['inception_4c'])
    incp4d = inception_unit_bn(n,'inception_4d',**arch_table['inception_4d'])
    clas_2 = watchtower_bn(n,incp4d,'loss2',nclass)

    incp4e = inception_unit_bn(n,'inception_4e',bottom=incp4d,**arch_table['inception_4e'])
    pool_4 = dict2net(n,{'pool4/3x3_s2':max_pool(get_bottom(n),ks=3,stride=2)})

    incp5a = inception_unit_bn(n,'inception_5a',**arch_table['inception_5a'])
    incp5b = inception_unit_bn(n,'inception_5b',**arch_table['inception_5b'])

    pool_5 = dict2net(n,{'pool5/7x7_s1':max_pool(incp5b,ks=7,stride=1)})
    poold5 = dict2net(n,{'pool5/drop_7x7_s1':L.Dropout(pool_5,in_place=True,dropout_ratio=0.4)})
    clas_3 = dict2net(n,{'loss3/classifier':L.InnerProduct(poold5,
                                                 param= [dict(lr_mult=1.0, decay_mult=1.0), 
                                                         dict(lr_mult=2.0, decay_mult=0)],
                                                 inner_product_param=dict(num_output=nclass,
                                                                         weight_filler=dict(type="xavier"))
                                                 )})

    loss_1 = dict2net(n,{'loss1/loss':L.SoftmaxWithLoss(clas_1,n.label,loss_weight=0.3)})
    loss_2 = dict2net(n,{'loss2/loss':L.SoftmaxWithLoss(clas_2,n.label,loss_weight=0.3)})
    loss_3 = dict2net(n,{'loss':L.SoftmaxWithLoss(clas_3,n.label,loss_weight=1.0)})

    if acc:
        dict2net(n,{'loss1/accuracy':L.Accuracy(clas_1,n.label,
                                             include=dict(phase=caffe.TEST))})
        dict2net(n,{'loss1/accuracy-top-'+str(topk):L.Accuracy(clas_1,n.label,
                                             include=dict(phase=caffe.TEST),top_k=topk)})
        dict2net(n,{'loss2/accuracy':L.Accuracy(clas_2,n.label,
                                             include=dict(phase=caffe.TEST))})
        dict2net(n,{'loss2/accuracy-top-'+str(topk):L.Accuracy(clas_2,n.label,
                                             include=dict(phase=caffe.TEST),top_k=topk)})
        
        dict2net(n,{'accuracy':L.Accuracy(clas_3,n.label,
                                             include=dict(phase=caffe.TEST))})
        dict2net(n,{'accuracy-top-'+str(topk):L.Accuracy(clas_3,n.label,
                                             include=dict(phase=caffe.TEST),top_k=topk)})
def inception_unit(n,s,bottom=None,
                   nouts={'1x1':64,'3x3':64,'5x5':32,'poolproj':32},
                   reduces={'3x3':96,'5x5':16}):
    
    if bottom is None:
        bottom = get_bottom(n)
    
    
    ns=OrderedDict()
    concat=[]
    # 1x1
    ns[s+'/1x1'],ns[s+'/relu_1x1'] = conv_relu(bottom,nouts['1x1'],ks=1,pad=0,
                                               wf=dict(type="xavier",std=0.03),
                                               bf=dict(type="constant",value=0.2))
    concat.append(ns[s+'/relu_1x1'])
    
    #3x3
    ns[s+'/3x3_reduce'],ns[s+'/relu_3x3_reduce'] = conv_relu(bottom,reduces['3x3'],ks=1,pad=0,
                                               wf=dict(type="xavier",std=0.09),
                                               bf=dict(type="constant",value=0.2))
    ns[s+'/3x3'],ns[s+'/relu_3x3'] = conv_relu(ns[s+'/relu_3x3_reduce'],nouts['3x3'],ks=3,pad=1,
                                               wf=dict(type="xavier",std=0.03),
                                               bf=dict(type="constant",value=0.2))
    concat.append(ns[s+'/relu_3x3'])
    
    #5x5
    ns[s+'/5x5_reduce'],ns[s+'/relu_5x5_reduce'] = conv_relu(bottom,reduces['5x5'],ks=1,pad=0,
                                               wf=dict(type="xavier",std=0.2),
                                               bf=dict(type="constant",value=0.2))
    ns[s+'/5x5'],ns[s+'/relu_5x5'] = conv_relu(ns[s+'/relu_5x5_reduce'],nouts['5x5'],ks=5,pad=2,
                                               wf=dict(type="xavier",std=0.03),
                                               bf=dict(type="constant",value=0.2))
    concat.append(ns[s+'/relu_5x5'])
    
    #max
    ns[s + '/pool'] = max_pool(bottom,ks=3,stride=1,pad=1)
    ns[s+'/pool_proj'],ns[s+'/relu_pool_proj'] = conv_relu(ns[s + '/pool'],nouts['poolproj'],ks=1,pad=0,
                                               wf=dict(type="xavier",std=0.1),
                                               bf=dict(type="constant",value=0.2))
    concat.append(ns[s+'/relu_pool_proj'])

    #concat
    ns[s + '/output'] = L.Concat(*concat)
    
    dict2net(n,ns)
    return ns[s + '/output']


def watchtower(n,bottom,s,nclass):
    
    ns=dict()
    ns[s+'/ave_pool'] = ave_pool(bottom, ks=5, stride=3, pad=0)
    
    ns[s+'/conv'],ns[s+'/relu_conv'] = conv_relu(ns[s+'/ave_pool'],128,ks=1,pad=0,
                                                 wf=dict(type="xavier",std=0.08),
                                                 bf=dict(type="constant",value=0.2))
                                                    
    ns[s+'/fc'],ns[s+'/relu_fc'] = ip_relu(ns[s+'/relu_conv'],1024,
                                                 wf=dict(type="xavier",std=0.08),
                                                 bf=dict(type="constant",value=0.2))
        
    
    ns[s+'/drop_fc'] = L.Dropout(ns[s+'/relu_fc'],in_place=True,dropout_ratio=0.7)
    ns[s+'/classifier'] = L.InnerProduct(ns[s+'/drop_fc'],
                                             param= [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
                                             inner_product_param=dict(num_output=nclass,
                                                                     weight_filler=dict(type="xavier",std=0.0009765625),
                                                                     bias_filler=dict(type="constant",value=0.0))
                                             ) 
    
    dict2net(n,ns)
    return ns[s+'/classifier']


def googlenet(n,nclass,data_top,add_watchtower=True,arch_table=None):
    
    if arch_table is None:
        arch_table = inception_vanila_table()
    conv1 = dict2net(n,dict(zip(['conv1/7x7','conv1/relu_7x7'],
                            conv_relu(data_top,64,ks=7,pad=3,stride=2,
                                      wf=dict(type="xavier")))))

    pool_1 = dict2net(n,{'pool1/3x3_s2':max_pool(get_bottom(n),ks=3,stride=2,pad=0)})
    pooln1 = dict2net(n,{'pool1/norm1':L.LRN(pool_1,local_size=5,alpha=0.0001,beta=0.75)})

    conv2r = dict2net(n,dict(zip(['conv2/3x3_reduce','conv2/relu_3x3_reduce'], 
                                 conv_relu(pooln1,64,ks=1,pad=0,
                                           wf=dict(type="xavier")))))
    conv2 = dict2net(n,dict(zip(['conv2/3x3','conv2/relu_3x3'],
                                conv_relu(conv2r,192,ks=3,pad=1,
                                          wf=dict(type="xavier")))))
    
    pool_2=dict2net(n,{'pool2/3x3_s2':max_pool(conv2,ks=3,stride=2,pad=0)})



    incp3a = inception_unit(n,'inception_3a',**arch_table['inception_3a'])
    incp3b = inception_unit(n,'inception_3b',**arch_table['inception_3b'])
    pool_3 = dict2net(n,{'pool3/3x3_s2':max_pool(incp3b,ks=3,stride=2,pad=0)})
    incp4a = inception_unit(n,'inception_4a',**arch_table['inception_4a'])
    if add_watchtower:
        clas_1 = watchtower(n,incp4a,'loss1',nclass)

    incp4b = inception_unit(n,'inception_4b',bottom=incp4a,**arch_table['inception_4b'])
    incp4c = inception_unit(n,'inception_4c',**arch_table['inception_4c'])
    incp4d = inception_unit(n,'inception_4d',**arch_table['inception_4d'])
    if add_watchtower:
        clas_2 = watchtower(n,incp4d,'loss2',nclass)

    incp4e = inception_unit(n,'inception_4e',bottom=incp4d,**arch_table['inception_4e'])
    pool_4 = dict2net(n,{'pool4/3x3_s2':max_pool(get_bottom(n),ks=3,stride=2)})

    incp5a = inception_unit(n,'inception_5a',**arch_table['inception_5a'])
    incp5b = inception_unit(n,'inception_5b',**arch_table['inception_5b'])

    pool_5 = dict2net(n,{'pool5/7x7_s1':max_pool(incp5b,ks=7,stride=1)})
    poold5 = dict2net(n,{'pool5/drop_7x7_s1':L.Dropout(pool_5,in_place=True,dropout_ratio=0.4)})
    clas_3 = dict2net(n,{'loss3/classifier':L.InnerProduct(poold5,
                                                 param= [dict(lr_mult=1.0, decay_mult=1.0), 
                                                         dict(lr_mult=2.0, decay_mult=0)],
                                                 inner_product_param=dict(num_output=nclass,
                                                                         weight_filler=dict(type="xavier"))
                                                 )})    
    if add_watchtower:
        return clas_1,clas_2,clas_3
    
    return clas_3
        

def vgg16(nclasses, source, transform_param=None, batch_size=32, acclayer = False, learn = True,is_color=False):
    n = caffe.NetSpec()
    if transform_param is None:
        transform_param=dict()
    n.data,n.label=L.ImageData(ntop=2,transform_param=transform_param,
                           source=source,
                           new_width=224,new_height=224,is_color=is_color,
                           shuffle=True)

    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, learn = learn, wf=dict(type="xavier"))
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, learn = learn, wf=dict(type="xavier"))
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, learn = learn, wf=dict(type="xavier"))
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, learn = learn, wf=dict(type="xavier"))
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, learn = learn, wf=dict(type="xavier"))
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, learn = learn, wf=dict(type="xavier"))
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, learn = learn, wf=dict(type="xavier"))
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, learn = learn, wf=dict(type="xavier"))
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, learn = learn, wf=dict(type="xavier"))
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, learn = learn, wf=dict(type="xavier"))
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, learn = learn, wf=dict(type="xavier"))
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, learn = learn, wf=dict(type="xavier"))
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, learn = learn, wf=dict(type="xavier"))
    n.pool5 = max_pool(n.relu5_3)

    if learn:
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)]


    n.fc6 = L.InnerProduct(n.pool5, num_output=4096,
        param = param, weight_filler=dict(type="xavier"))

    n.relu6 = L.ReLU(n.fc6, in_place=True)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7 = L.InnerProduct(n.fc6, num_output=4096,
        param = param, weight_filler=dict(type="xavier"))

    n.relu7 = L.ReLU(n.fc7, in_place=True)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.score = L.InnerProduct(n.fc7, num_output=nclasses,
        param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)], weight_filler=dict(type="xavier")) #always learn this layer. Else it's no fun!

    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    
    if acclayer:
        n.accuracy = L.Accuracy(n.score, n.label)
    return n


def residual_standard_unit(n, nout, s, newdepth = False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    bottom = n.__dict__['tops'][n.__dict__['tops'].keys()[-1]] #find the last layer in netspec
    stride = 2 if newdepth else 1

    n[s + 'conv1'], n[s + 'bn1'], n[s + 'lrn1'] = conv_bn(bottom, ks = 3, stride = stride, nout = nout, pad = 1)
    n[s + 'relu1'] = L.ReLU(s + 'lrn1', in_place=True)
    n[s + 'conv2'], n[s + 'bn2'], n[s + 'lrn2'] = conv_bn(s + 'relu1', ks = 3, stride = 1, nout = nout, pad = 1)
   
    if newdepth: 
        n[s + 'conv_expand'], n[s + 'bn_expand'], n[s + 'lrn_expand'] = conv_bn(bottom, ks = 1, stride = 2, nout = nout, pad = 0)
        n[s + 'sum'] = L.Eltwise(s + 'lrn2', s + 'lrn_expand')
    else:
        n[s + 'sum'] = L.Eltwise(s + 'lrn2', bottom)

    n[s + 'relu2'] = L.ReLU(s + 'sum', in_place=True)
    

def residual_bottleneck_unit(n, nout, s, newdepth = False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    
    bottom = n.__dict__['tops'].keys()[-1] #find the last layer in netspec
    stride = 2 if newdepth and nout > 64 else 1

    n[s + 'conv1'], n[s + 'bn1'], n[s + 'lrn1'] = conv_bn(n[bottom], ks = 1, stride = stride, nout = nout, pad = 0)
    n[s + 'relu1'] = L.ReLU(n[s + 'lrn1'], in_place=True)
    n[s + 'conv2'], n[s + 'bn2'], n[s + 'lrn2'] = conv_bn(n[s + 'relu1'], ks = 3, stride = 1, nout = nout, pad = 1)
    n[s + 'relu2'] = L.ReLU(n[s + 'lrn2'], in_place=True)
    n[s + 'conv3'], n[s + 'bn3'], n[s + 'lrn3'] = conv_bn(n[s + 'relu2'], ks = 1, stride = 1, nout = nout * 4, pad = 0)
   
    if newdepth: 
        n[s + 'conv_expand'], n[s + 'bn_expand'], n[s + 'lrn_expand'] = conv_bn(n[bottom], ks = 1, stride = stride, nout = nout * 4, pad = 0)
        n[s + 'sum'] = L.Eltwise(n[s + 'lrn3'], n[s + 'lrn_expand'])
    else:
        n[s + 'sum'] = L.Eltwise(n[s + 'lrn3'], n[bottom])

    n[s + 'relu3'] = L.ReLU(n[s + 'sum'], in_place=True)

def residual_net(total_depth, data_layer_params, num_classes = 1000, acclayer = True):
    """
    Generates nets from "Deep Residual Learning for Image Recognition". Nets follow architectures outlined in Table 1. 
    """
    # figure out network structure
    net_defs = {
        18:([2, 2, 2, 2], "standard"),
        34:([3, 4, 6, 3], "standard"),
        50:([3, 4, 6, 3], "bottleneck"),
        101:([3, 4, 23, 3], "bottleneck"),
        152:([3, 8, 36, 3], "bottleneck"),
    }
    assert total_depth in net_defs.keys(), "net of depth:{} not defined".format(total_depth)

    nunits_list, unit_type = net_defs[total_depth] # nunits_list a list of integers indicating the number of layers in each depth.
    nouts = [64, 128, 256, 512] # same for all nets

    # setup the first couple of layers
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'beijbom_caffe_data_layers', layer = 'ImageNetDataLayer',
                ntop = 2, param_str=str(data_layer_params))
    n.conv1, n.bn1, n.lrn1 = conv_bn(n.data, ks = 7, stride = 2, nout = 64, pad = 3)
    n.relu1 = L.ReLU(n.lrn1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, stride = 2, kernel_size = 3)
    
    # make the convolutional body
    for nout, nunits in zip(nouts, nunits_list): # for each depth and nunits
        for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
            s = str(nout) + '_' + str(unit) + '_' # layer name prefix
            if unit_type == "standard":
                residual_standard_unit(n, nout, s, newdepth = unit is 1 and nout > 64)
            else:
                residual_bottleneck_unit(n, nout, s, newdepth = unit is 1)
                
    # add the end layers                    
    n.global_pool = L.Pooling(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], pooling_param = dict(pool = 1, global_pooling = True))
    n.score = L.InnerProduct(n.global_pool, num_output = num_classes,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    if acclayer:
        n.accuracy = L.Accuracy(n.score, n.label)

    return n    
