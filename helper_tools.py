import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt

pjoin = os.path.join
psplit = os.path.split
psplitext = os.path.splitext
pexists = os.path.exists
mkdir = os.makedirs
npa = np.array

def try_float(s):
    try:
        s = float(s)
    except:
        return s
    return s

def try_int(s):
    try:
        s=int(float(s))
    except:
        return s

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

