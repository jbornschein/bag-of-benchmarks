#!/usr/bin/env python


from __future__ import division

import sys
sys.path.insert(0, "../pylib")

from timeit import Timer
from time import time
from mpi4py import MPI
import numpy as np

from parutils import pprint


#=============================================================================
# Main

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--nbytes", dest="nbytes", type="int", default=64,
                    help="Size of input array in MiB [default=64]")
parser.add_option("-t", "--dtype", dest="dtype", default="float32",
                    help="Datatype to be benchmarked")


(options, args) = parser.parse_args()

# Parse dtype argument
if options.dtype == "float32":
    dtype_str = "np.float32"
    dtype = np.float32
elif options.dtype == "float64":
    dtype_str = "np.float64"
    dtype = np.float64
else:
    print "[FATAL] Unknown type %s" % options.dtype

# Calculate sizes
nbytes = options.nbytes * 1024 * 1024
size = nbytes // np.dtype(dtype).itemsize
shape = (size,)

comm = MPI.COMM_WORLD
if comm.size != 1:
    pprint()
    pprint("Running %d parallel MPI processes: Results display collective performance")
    pprint()


##############################################################################
# linear benchmarks

linear_benchcodes = (
    ("x = 1 * a"             ,  1  , 2  ),
    ("x = a * a"             ,  1  , 3  ),
    ("x = a * b"             ,  1  , 3  ),
    ("x = a * b * c"         ,  2  , 6  ),
    ("x = a[::2] * b[::2]"   ,  0.5, 1.5),
    ("x = np.exp(a)"         ,  1  , 2  ),
    ("x = np.log(a)"         ,  1  , 2  ),
    ("x = np.sin(a)"         ,  1  , 2  ),
    ("x = np.cos(a)"         ,  1  , 2  ),
)

np_setup = """
from mpi4py import MPI
import numpy as np

gc.enable();

nbytes = %d
dtype = %s
size = nbytes // np.dtype(dtype).itemsize
shape = (size,)

a = 1*np.ones(shape, dtype)
b = 2*np.ones(shape, dtype)
c = 3*np.ones(shape, dtype)
x = np.empty(shape, dtype)
y = np.empty(shape, dtype)
z = np.empty(shape, dtype)


comm = MPI.COMM_WORLD

comm.Barrier()
"""



def bench_linear(src, nbytes, dtype_str, runs=3, number=10):
    number_cl = 1
    number_np = 1

    timer_np = Timer(src+"; comm.Barrier()", np_setup % (nbytes, dtype_str))  
    t_np = min( timer_np.repeat(runs, number_np) ) / number_np

    return t_np
 
pprint()
pprint("O(n) function benchmarks")
pprint("========================")
pprint()
pprint(" Input arrays a, b and c are of dtype=%s, shape=%s, nbytes=%d MiB"  % (dtype_str, shape, nbytes/1024/1024))
pprint()
pprint(" Code                      ||      GFLOP/s |   membw GiB/s |")
pprint("---------------------------++--------------+---------------+")
for src, flopfac, memfac in linear_benchcodes:
    t_np = bench_linear(src, nbytes, dtype_str)

    gflops = comm.size * size * flopfac / t_np / 1e9
    membw = comm.size * nbytes * memfac / t_np / 1024 / 1024 /1024

    pprint(" %25s ||    %9.3f |     %9.3f |" % (src, gflops, membw,))
        

