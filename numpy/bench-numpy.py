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
# Run benchmark

o1_setup = """
from mpi4py import MPI
import numpy as np

gc.enable();

dtype = %(DTYPE)s
shape = %(SHAPE)s

a = 1*np.ones(shape, dtype)
b = 2*np.ones(shape, dtype)
c = 3*np.ones(shape, dtype)
x = np.empty(shape, dtype)
y = np.empty(shape, dtype)
z = np.empty(shape, dtype)


comm = MPI.COMM_WORLD

comm.Barrier()
"""

o2_setup = """
from mpi4py import MPI
import numpy as np

gc.enable();

dtype = %(DTYPE)s
shape1 = %(SHAPE1)s
shape2 = %(SHAPE2)s

a = 1*np.ones(shape1, dtype)
b = 2*np.ones(shape1, dtype)
c = 3*np.ones(shape1, dtype)
A = 1*np.ones(shape2, dtype)
B = 2*np.ones(shape2, dtype)
C = 3*np.ones(shape2, dtype)


comm = MPI.COMM_WORLD

comm.Barrier()
"""



def bench_code(setup, src, runs=3, number=10):
    number_cl = 1
    number_np = 1

    timer_np = Timer(src+"; comm.Barrier()", setup)  
    t_np = min( timer_np.repeat(runs, number_np) ) / number_np

    return t_np


#=============================================================================
# Main



from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--nbytes", dest="nbytes", type="int", default=64,
                    help="Size of input array in MiB [default=64]")
parser.add_option("-t", "--dtype", dest="dtype", default="float64",
                    help="Datatype to be benchmarked")
parser.add_option("-b", "--benchmarks", dest="benches", default="O1,O2",
                    help="Benchmarks to run (default: O1,O2)")


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

benches = options.benches.split(",")

comm = MPI.COMM_WORLD
pprint()
pprint("Running %d parallel MPI processes: Results display collective performance" % comm.size)
pprint()

# Calculate sizes
nbytes = options.nbytes * 1024 * 1024
size = nbytes // np.dtype(dtype).itemsize

if 'O1' in benches:
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

    shape1 = (size,)

    pprint()
    pprint("O(n) function benchmarks")
    pprint("========================")
    pprint()
    pprint(" Input arrays a, b and c are of dtype=%s, shape=%s, nbytes=%d MiB"  % (dtype_str, shape1, nbytes/1024/1024))
    pprint()
    pprint(" Code                      ||      GFLOP/s |   membw GiB/s |")
    pprint("---------------------------++--------------+---------------+")
    
    subs = {
        'SHAPE'  : shape1,
        'DTYPE'  : dtype_str
    }
    setup = o1_setup % subs
    for src, flopfac, memfac in linear_benchcodes:
        t = bench_code(setup, src)

        gflops = comm.size * size * flopfac / t / 1e9
        membw = comm.size * nbytes * memfac / t / 1024 / 1024 /1024

        pprint(" %25s ||    %9.3f |     %9.3f |" % (src, gflops, membw))
        
if 'O2' in benches:

    o2_benchcodes = (
        ("x = np.inner(A, a)"    ,  2 ),
        ("x = np.inner(A, b)"    ,  2 ),
        ("X = np.inner(A, B)"    ,  3 ),
        ("X = np.dot(A, B)"      ,  3 ),
    )

    N = int(np.sqrt(size))
    shape1 = (N,)
    shape2 = (N, N)

    pprint()
    pprint("O(n^2) function benchmarks")
    pprint("==========================")
    pprint()
    pprint(" Input arrays a, and b are of dtype=%s, shape=%s"  % (dtype_str, str(shape1)))
    pprint(" Input arrays A, and B are of shape=%s"  % str(shape2))
    pprint()
    pprint(" Code                      ||      GFLOP/s |")
    pprint("---------------------------++--------------+")

    subs = {
        'SHAPE1'  : str(shape1),
        'SHAPE2'  : str(shape2),
        'DTYPE'   : dtype_str,
    }
    setup = o2_setup % subs
    for src, flopexp in o2_benchcodes:
        t_np = bench_code(setup, src)

        gflops = comm.size * N**flopexp / t_np / 1e9

        pprint(" %25s ||    %9.3f |" % (src, gflops))

