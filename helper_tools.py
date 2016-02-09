import os
import numpy as np

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
        s= s
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