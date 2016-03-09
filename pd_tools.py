from __future__ import print_function
import pandas as pd
from pandas import HDFStore
import os
import time
import errno




class SafeHDFStore(HDFStore):
    def __init__(self, *args, **kwargs):
        probe_interval = kwargs.pop("probe_interval", 1)
        self._lock = "%s.lock" % args[0]
        while True:
            try:
                self._flock = os.open(self._lock, os.O_CREAT |
                                                  os.O_EXCL |
                                                  os.O_WRONLY)
                break
            except OSError as e:
                if e.errno == errno.EEXIST:
                    time.sleep(probe_interval)
                else:
                    raise

        HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        HDFStore.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)

def hdfstore_append(storefile,key,df,format="t",columns=None,data_columns=None,probe_interval=1,print_error=False):
    if df is None:
        return
    if key[0]!='/':
        key='/'+key
    with SafeHDFStore(storefile,probe_interval=probe_interval) as store:
        if key not in store.keys():
            store.put(key,df,format=format,columns=columns,data_columns=data_columns)
        else:
            try:
                dtypes=store[key].dtypes
                for col,dtype in zip(dtypes.index,dtypes):
                    df[col] = df[col].astype(dtype)
                store.append(key,df)
            except Exception as inst:
                if print_error:
                    print("rebuilding", key)
                    print(inst)
                df = pd.concat([store.get(key),df])
                store.put(key,df,format=format,columns=columns,
                          data_columns=data_columns)



def infer_key_dtype(keys):
    time_keys=[k for k in keys if "time" in k.lower()]
    str_keys=[k for k in keys if "str" in k.lower() or "dir" in k.lower()]
    numeric_keys=[k for k in keys if k not in time_keys and k not in str_keys ]
    return numeric_keys,str_keys,time_keys

def apply_inferred_dypes(df):
    cols=df.columns
    strstr="md_|str|dir|filename"
    timestr="time|date"

    str_cols=cols[df.columns.str.lower().str.contains(strstr)]
    time_cols=cols[(df.columns.str.lower().str.contains(timestr))].difference(str_cols)
    numeric_cols=cols.difference(str_cols).difference(time_cols)
    for c in str_cols:
        df[c]=df[c].apply(str)
    for c in time_cols:
        df[c]=pd.to_datetime(df[c],errors="coerce")
    for c in numeric_cols:
        df[c]=pd.to_numeric(df[c],errors="coerce").astype(float)
        
    return df