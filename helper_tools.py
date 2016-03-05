import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt

pjoin = os.path.join
psplit = os.path.split
psplitext = os.path.splitext
pexists = os.path.exists
mkdir = os.makedirs
npa = np.array

def cat_file(file):
    if not pexists(file):
        print("File {} does not exist".format(str(file)))
        return
    with open(file, "r") as f:
        text = f.read()
        print(text)

def is_int(s):
    try:
        int(float(s))
        return True
    except:
        return False

def try_int(s,default=None):
    try:
        return int(float(s))
    except:
        if default is None:
            return s
        else:
            return default
        
def try_float(s,default=None):
    try:
        return float(s)
    except:
        if default is None:
            return s
        else:
            return default

def try_int_float(s,default=None):
    try:
        if int(float(s))==float(s):
            return int(float(s))
        else:
            return try_float(s,default=default)
    except:
        if default is None:
            return s
        else:
            return default
    
def try_bool_int_float(s,default=None):
    try:
        if float(s)==0:
            return False

        if float(s)==1:
            return True
        return try_int_float(s,default=default)
    except:
        return try_int_float(s,default=default)

def pbasename(fn):
    return os.path.splitext(os.path.split(fn)[1])[0]

def listdir(d,suf):
    return [f for f in os.listdir(d) if d.endswith(suf)]

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key,value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def check_file(fn,sep='.{}',end=100,loc='append'):
    if loc is 'append':
        tfn=fn+sep
    elif loc is 'newdir':
        p,f=os.path.split(fn)
        tfn=pjoin(p,sep,f)
    else:
        raise ValueError("Unrecognized loc: {}".format(loc))
    if pexists(fn):
        for n in range(0,100):
            if not pexists(tfn.format(n)):
                return tfn.format(n)
        raise Exception('Hit the end ({})'.format(end))
    else:
        return fn

def check_file_replace(fn,return_new=False):
    fn2=check_file(fn)
    if fn2!=fn:
        print("File Exists, backing up old copy to:")
        print(fn2)
        os.rename(fn,fn2)
        if return_new:
            return False,fn2
        else:
            return False
    if return_new:
        return True,fn
    else:
        return True

