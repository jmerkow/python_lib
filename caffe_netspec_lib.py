from __future__ import print_function
from helper_tools import *
from image_tools import *
from caffe import layers as L, params as P
from caffe import net_spec
from caffe.proto import caffe_pb2
from collections import OrderedDict

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
              param=None, group=1, wf=None,bf=None):
    if learn and param is None:
        param = [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    
    if wf is None:
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, param=param, group=group)
    else:
        if bf is None:
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                num_output=nout, pad=pad, param=param, group=group ,weight_filler=wf)
        else:
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=param, group=group,
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

def conv_bn_scale(bottom, nout, ks = 3, stride=1, pad = 0,
                       learn = True, bn_learn = True,
                       use_global_stats=False, add_relu=False, **convargs):
    if learn:
        param = [dict(lr_mult=1, decay_mult=1)]
    else:
        param = [dict(lr_mult=0, decay_mult=0)]
        
    if bn_learn:
        bn_param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)]
    else:
        bn_param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),]
    
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, param = param, bias_term=False,
                         weight_filler=dict(type="msra"),**convargs)
    
    bn= L.BatchNorm(conv,in_place=True,use_global_stats=use_global_stats)
    scale = L.Scale(bn, in_place=True, bias_term=True)
    
    if add_relu:
        relu = L.ReLU(scale,in_place=True)
        return conv, bn, scale, relu
    else:
        return conv, bn, scale
    

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
    if top_k is None or top_k==0:
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


def googlenet(n,nclass,data_top,add_watchtower=True,arch_table=None, dropout=0.4,
              return_layer=None):
    
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
    if dropout:
        pool_5 = dict2net(n,{'pool5/drop_7x7_s1':L.Dropout(pool_5,in_place=True,dropout_ratio=dropout)})
    clas_3 = dict2net(n,{'loss3/classifier-'+str(nclass):L.InnerProduct(pool_5,
                                                 param= [dict(lr_mult=1.0, decay_mult=1.0), 
                                                         dict(lr_mult=2.0, decay_mult=0)],
                                                 inner_product_param=dict(num_output=nclass,
                                                                         weight_filler=dict(type="xavier"))
                                                 )})    
    if add_watchtower:
        if return_layer is None:
            return clas_1,clas_2,clas_3
        else:
            return clas_1,clas_2,clas_3,getattr(n,return_layer)
    if return_layer is None:
        return clas_3
    else:
        return getattr(n,return_layer)

def googlenet2(n,nclass,data_top,arch_table=None):
    
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
    incp4b = inception_unit(n,'inception_4b',bottom=incp4a,**arch_table['inception_4b'])
    incp4c = inception_unit(n,'inception_4c',**arch_table['inception_4c'])
    incp4d = inception_unit(n,'inception_4d',**arch_table['inception_4d'])
    incp4e = inception_unit(n,'inception_4e',bottom=incp4d,**arch_table['inception_4e'])
    pool_4 = dict2net(n,{'pool4/3x3_s2':max_pool(get_bottom(n),ks=3,stride=2)})

    incp5a = inception_unit(n,'inception_5a',**arch_table['inception_5a'])
    incp5b = inception_unit(n,'inception_5b',**arch_table['inception_5b'])

    pool_5 = dict2net(n,{'pool5/7x7_s1':max_pool(incp5b,ks=7,stride=1)})
    if dropout:
        pool_5 = dict2net(n,{'pool5/drop_7x7_s1':L.Dropout(pool_5,in_place=True,dropout_ratio=dropout)})

    
        

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

def residual_standard_unit(n, bottom, nout, s, newdepth = False, use_global_stats=False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
#     bottom = n.__dict__['tops'][n.__dict__['tops'].keys()[-1]] #find the last layer in netspec
    ns=dict()
    stride = newdepth if newdepth else 1

    ns[s + '_branch2conv1'], ns[s + '_branch2bn1'], ns[s + '_branch2scale1']  = conv_bn_scale(bottom, ks = 3, 
                                                                                           stride = stride, nout = nout, pad = 1, 
                                                                                           use_global_stats=use_global_stats)
    ns[s + '_branch2relu1'] = L.ReLU(ns[s + '_branch2scale1'], in_place=True)
    ns[s + '_branch2conv2'], ns[s + '_branch2bn2'], ns[s + '_branch2scale2'] = conv_bn_scale(ns[s + '_branch2relu1'], ks = 3,
                                                                                          stride = 1, nout = nout, pad = 1,
                                                                                          use_global_stats=use_global_stats)
   
    if newdepth:
        ns[s + '_branch1conv'], ns[s + '_branch1bn1'], ns[s + '_branch1scale1'] = conv_bn_scale(bottom, ks = 1, 
                                                                                            stride = stride, nout = nout, pad = 0,
                                                                                            use_global_stats=use_global_stats)
        ns[s] = L.Eltwise(ns[s + '_branch1scale1'],ns[s + '_branch2scale2'])
    else:
        ns[s] = L.Eltwise(bottom, ns[s + '_branch2scale2'])

    ns[s + '_relu'] = L.ReLU(ns[s], in_place=True)
    
    dict2net(n,ns)
    return ns[s + '_relu']
    
def residual_bottleneck_unit(n,bottom, nout, s, newdepth = False, use_global_stats=False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    
#     bottom = n.__dict__['tops'][n.__dict__['tops'].keys()[-1]]
    
    ns=dict()
    stride = newdepth if newdepth else 1

    ns[s + '_branch2conv1'], ns[s + '_branch2bn1'], ns[s + '_branch2scale1'] = conv_bn_scale(bottom, ks = 1, 
                                                                                           stride = stride, nout = nout, pad = 0,
                                                                                           use_global_stats=use_global_stats)
    ns[s + '_branch2relu1'] = L.ReLU(ns[s + '_branch2scale1'], in_place=True)
    ns[s + '_branch2conv2'], ns[s + '_branch2bn2'], ns[s + '_branch2scale2'] = conv_bn_scale(ns[s + '_branch2relu1'], ks = 3,
                                                                                          stride = 1, nout = nout, pad = 1,
                                                                                          use_global_stats=use_global_stats)
    ns[s + '_branch2relu2'] = L.ReLU(ns[s + '_branch2scale2'], in_place=True)
    ns[s + '_branch2conv3'], ns[s + '_branch2bn3'], ns[s + '_branch2scale3'] = conv_bn_scale(ns[s + '_branch2relu2'], ks = 1,
                                                                                          stride = 1, nout = nout*4, pad = 0,
                                                                                          use_global_stats=use_global_stats)
   
    if newdepth:
        ns[s + '_branch1conv'], ns[s + '_branch1bn1'], ns[s + '_branch1scale1'] = conv_bn_scale(bottom, ks = 1, 
                                                                                            stride = stride, nout = nout*4, pad = 0,
                                                                                            use_global_stats=use_global_stats)
        ns[s] = L.Eltwise(ns[s + '_branch1scale1'],ns[s + '_branch2scale3'])
    else:
        ns[s] = L.Eltwise(bottom, ns[s + '_branch2scale3'])

    ns[s + '_relu'] = L.ReLU(ns[s], in_place=True)
    
    dict2net(n,ns)
    return ns[s + '_relu']
    
def residual_net(n,bottom,total_depth, nclasses, use_global_stats=False,return_layer=None):
    """
    Generates nets from "Deep Residual Learning for Image Recognition". 
    Nets follow architectures outlined in Table 1. 
    """
    # figure out network structure
    net_defs = {
        18:([2, 2, 2, 2], "standard"),
        34:([3, 4, 6, 3], "standard"),
        50:([3, 4, 6, 3], "bottleneck"),
        101:([3, 4, 23, 3], "bottleneck"),
        152:([3, 8, 36, 3], "bottleneck"),
    }
    alpha = string.ascii_lowercase
    assert total_depth in net_defs.keys(), "net of depth:{} not defined".format(total_depth)

    # nunits_list a list of integers indicating the number of layers in each depth.
    nunits_list, unit_type = net_defs[total_depth] 
    nouts = [64, 128, 256, 512] # same for all nets
    
    n.conv1, n.bn1, n.scale1 = conv_bn_scale(bottom, ks = 7, 
                                           stride = 2, nout = 64, pad = 3,
                                           use_global_stats=use_global_stats)
    n.conv1_relu = L.ReLU(n.scale1, in_place=True)
    n.pool1 = L.Pooling(n.conv1_relu, stride = 2, kernel_size = 3, pool=P.Pooling.MAX)
    
    U=n.pool1
    
    # make the convolutional body
    for i,(nout, nunits) in enumerate(zip(nouts, nunits_list)): # for each depth and nunits
        for unit,a in zip(range(1, nunits + 1),alpha): # for each unit. Enumerate from 1.
#             s = str(nout) + '_' + str(unit) + '_' # layer name prefix
            s= 'res{}{}'.format(i+2,a)
#             print(s)
            newdepth = 2 if unit is 1 else 0
            if i is 0 and newdepth:
                newdepth=1
            if unit_type == "standard":

                U=residual_standard_unit(n,U, nout, s, newdepth = newdepth, use_global_stats=use_global_stats)
            else:
                U=residual_bottleneck_unit(n,U, nout, s, newdepth = newdepth, use_global_stats=use_global_stats)

    # add the end layers                    
    n.global_pool = L.Pooling(U, pooling_param = dict(pool = 1, global_pooling = True))
    setattr(n,'fc'+str(nclasses),  L.InnerProduct(n.global_pool, num_output = nclasses,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]))
    
    if return_layer is None:
        return getattr(n,'fc'+str(nclasses))
    else:
        return getattr(n,return_layer)


def residual_standard_unit_old(n, nout, s, newdepth = False):
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
    

def residual_bottleneck_unit_old(n, nout, s, newdepth = False):
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

def residual_net_old(total_depth, data_layer_params, num_classes = 1000, acclayer = True):
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
                residual_standard_unit_old(n, nout, s, newdepth = unit is 1 and nout > 64)
            else:
                residual_bottleneck_unit_old(n, nout, s, newdepth = unit is 1)
                
    # add the end layers                    
    n.global_pool = L.Pooling(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], pooling_param = dict(pool = 1, global_pooling = True))
    n.score = L.InnerProduct(n.global_pool, num_output = num_classes,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    if acclayer:
        n.accuracy = L.Accuracy(n.score, n.label)

    return n   