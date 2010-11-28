#!/usr/bin/env python

from __future__ import division

import sys
sys.path.insert(0, "../pylib")

from time import time
from mpi4py import MPI
import numpy as np

from parutils import pprint

sizes = [ 2**n for n in xrange(1,24) ]
runs  = 20

comm = MPI.COMM_WORLD


pprint("Benchmarking reduce performance on %d parallel MPI processes..." % comm.size)
pprint()
pprint("%15s | %12s | %12s" % 
    ("Size (bytes)", "Time (msec)", "Bandwidth (MiBytes/s)"))

for s in sizes:
    data = np.ones(s)
    res = np.empty_like(data)

    comm.Barrier()
    t0 = time()
    for i in xrange(runs):
        comm.Reduce( [data, MPI.DOUBLE], [res, MPI.DOUBLE] ) 
    comm.Barrier()
    t = (time()-t0) / runs
    
    pprint("%15d | %12.3f | %12.3f" % 
        (data.nbytes, t*1000, data.nbytes/t/1024/1024) )
