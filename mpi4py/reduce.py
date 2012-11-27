#!/usr/bin/env python

from __future__ import division

import sys
sys.path.insert(0, "../pylib")

from time import time
from mpi4py import MPI
import numpy as np

from parutils import pprint

sizes = [ 2**n for n in xrange(1,24) ]
runs  = 50

comm = MPI.COMM_WORLD


pprint("Benchmarking Reduce performance on %d parallel MPI processes..." % comm.size)
pprint()
pprint("%15s | %12s | %12s" % 
    ("Size (bytes)", "Time (msec)", "Bandwidth (MiBytes/s)"))

for s in sizes:
    data = np.ones(s)
    res = np.empty_like(data)

    comm.Barrier()
    t_min = np.inf
    for i in xrange(runs):
        t0 = time()
        comm.Reduce( [data, MPI.DOUBLE], [res, MPI.DOUBLE] ) 
        t = time()-t0
        t_min = min(t, t_min)
    comm.Barrier()
    
    pprint("%15d | %12.3f | %12.3f" % 
        (data.nbytes, t_min*1000, data.nbytes/t_min/1024/1024) )

pprint("Benchmarking AllReduce performance on %d parallel MPI processes..." % comm.size)
pprint()
pprint("%15s | %12s | %12s" % 
    ("Size (bytes)", "Time (msec)", "Bandwidth (MiBytes/s)"))

for s in sizes:
    data = np.ones(s)
    res = np.empty_like(data)

    comm.Barrier()
    t_min = np.inf
    for i in xrange(runs):
        t0 = time()
        comm.Allreduce( [data, MPI.DOUBLE], [res, MPI.DOUBLE] ) 
        t = time()-t0
        t_min = min(t, t_min)
    comm.Barrier()
    
    pprint("%15d | %12.3f | %12.3f" % 
        (data.nbytes, t_min*1000, data.nbytes/t_min/1024/1024) )

