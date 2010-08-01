#!/usr/bin/env python

from __future__ import division
import sys
sys.path.insert(0, "../pylib")

import numpy as np

from mpi4py import MPI
from parutils import pprint


#=============================================================================
# Main

comm = MPI.COMM_WORLD

pprint("-"*78)
pprint(" Running %d parallel processes..." % comm.size)
pprint("-"*78)

my_N = 10 + comm.rank

my_a = comm.rank * np.ones(my_N)

N = comm.allreduce(my_N)

#a = np.empty(N)
a = comm.gather(my_a)

pprint("Gathered array: %s" % a)
