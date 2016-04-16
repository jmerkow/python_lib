#!/usr/bin/env python
from settings import *

sys.path.insert(0,CAFFEPYTHONPATH)
sys.path.insert(0,CAFFEEXTRASPATH)

import caffe_tools as ct
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("-dbg",action="store_true")
    parser.add_argument("-ft",default=None)

    args = parser.parse_args()


    ct.run(caffepath=CAFFEEXEPATH,gpuid=args.gpu,debug=args.dbg,caffemodel=args.ft)