#!/usr/bin/env python

from time import time
from mpi4py import MPI
from numpy.random import randn
import numpy as np

sizes = [ 2**n for n in xrange(1,24) ]
runs  = 20

WORLD = MPI.COMM_WORLD
for s in sizes:
    data = np.ones(s)

    WORLD.Barrier()
    t0 = time()
    for i in xrange(runs):
        WORLD.Bcast( (data, MPI.DOUBLE), 0)
        #data = WORLD.Bcast( data, 0)
    WORLD.Barrier()
    t = (time()-t0) / runs
    
    #print data.sum()

    if WORLD.rank == 0:
        print "%d , %f" % (s, 1000*t)
