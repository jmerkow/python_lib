import pandas as pd
from pandas import HDFStore
import os
import time

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
            except FileExistsError:
                time.sleep(probe_interval)

        HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        HDFStore.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)

def hdfstore_append(storefile,key,df,format="t",columns=None,data_columns=None,probe_interval=1):
    if df is None:
        return
    if key[0]!='/':
        key='/'+key
    with SafeHDFStore(storefile,probe_interval=probe_interval) as store:
        if key not in store.keys():
            store.put(key,df,format=format,columns=columns,data_columns=data_columns)
        else:
            try:
                store.append(key,df)
            except Exception as inst:
                df = pd.concat([store.get(key),df])
                store.put(key,df,format=format,columns=columns,
                          data_columns=data_columns)