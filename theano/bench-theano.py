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
# Templates

template_np = """
from time import time
from mpi4py import MPI

import numpy as np
import numpy as P

runs  = %(RUNS)s
dtype = %(DTYPE)s
shape = %(SHAPE)s

a = np.random.normal()*np.ones(shape, dtype)
b = np.random.normal()*np.ones(shape, dtype)
c = np.random.normal()*np.ones(shape, dtype)
x = np.empty(shape, dtype)
y = np.empty(shape, dtype)
z = np.empty(shape, dtype)

comm = MPI.COMM_WORLD

best_t = np.inf
for r in xrange(runs):
    comm.Barrier
    t0 = time()
    %(CODE)s
    comm.Barrier()
    t1 = time()
    best_t = min(best_t, t1-t0)

"""

template_th = """
from time import time
from mpi4py import MPI
import numpy as np

import theano.tensor as T
import theano.tensor as P
from theano import function

runs = %(RUNS)s
dtype = "float32" # %(DTYPE)s
shape = %(SHAPE)s

# Create and compile function
a = T.vector("a", dtype=dtype)
b = T.vector("b", dtype=dtype)
c = T.vector("c", dtype=dtype)
x = T.vector("x", dtype=dtype)
y = T.vector("y", dtype=dtype)
z = T.vector("z", dtype=dtype)

%(CODE)s

f = function([a,b,c], x)

# Run benchmark with compiled function

comm = MPI.COMM_WORLD

best_t = np.inf
for r in xrange(runs):
    a = 1*np.ones(shape, dtype)
    b = 2*np.ones(shape, dtype)
    c = 3*np.ones(shape, dtype)
    x = np.empty(shape, dtype)
    y = np.empty(shape, dtype)
    z = np.empty(shape, dtype)

    comm.Barrier
    t0 = time()
    f(a,b,c)
    comm.Barrier()
    t1 = time()
    best_t = min(best_t, t1-t0)
"""

def bench_code(code, ctx={}):
    exec code in ctx

    return ctx["best_t"]

#=============================================================================
# Main

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--nbytes", dest="nbytes", type="int", default=64,
                    help="Size of input array in MiB [default=64]")
parser.add_option("-r", "--repeat", dest="runs", type="int", default=3,
                    help="Number of times each benchmark should be repeated [3]")
parser.add_option("-t", "--dtype", dest="dtype", default="float32",
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
runs = int(options.runs)
nbytes = options.nbytes * 1024 * 1024
size = nbytes // np.dtype(dtype).itemsize

if 'O1' in benches:
    linear_benchcodes = (
        ("x = 1 * a"                  ,  1  , 2  ),
        ("x = a * a"                  ,  1  , 2  ),
        ("x = a * b"                  ,  1  , 3  ),
        ("x = a * b * c"              ,  2  , 4  ),
        ("x = a[::2] * b[::2]"        ,  0.5, 1.5),
        ("x = a+a**2+a**3+a**4+a**5"  ,  8  , 2  ),
        ("x = a[2:]+a[1:-1]+a[:-2]"   ,  3  , 2  ),
        ("x = P.exp(a)"               ,  1  , 2  ),
        ("x = P.log(a)"               ,  1  , 2  ),
        ("x = P.sin(a)"               ,  1  , 2  ),
        ("x = P.cos(a)"               ,  1  , 2  ),
    )

    shape1 = (size,)

    pprint()
    pprint("O(n) function benchmarks")
    pprint("========================")
    pprint()
    pprint(" Input arrays a, b and c are of dtype=%s, shape=%s, nbytes=%d MiB"  % (dtype_str, shape1, nbytes/1024/1024))
    pprint()
    pprint(" Code                      ||          NumPy             |         Theano             |   Speedup | ")
    pprint("                           ||     GFLOP/s |  membw GiB/s |     GFLOP/s |  membw GiB/s |           |")
    pprint("---------------------------++-------------+--------------+-------------+--------------+-----------+")
    
    for code, flopfac, memfac in linear_benchcodes:
        subs = {
            'CODE'   : code,
            'SHAPE'  : shape1,
            'DTYPE'  : dtype_str,
            'RUNS'   : str(runs)
        }

        code_np = template_np % subs
        code_th = template_th % subs
 
        t_np = bench_code(code_np)
        t_th = bench_code(code_th)
        speedup = t_np / t_th

        gflops_np = comm.size * size * flopfac / t_np / 1e9
        membw_np = comm.size * nbytes * memfac / t_np / 1024 / 1024 /1024

        gflops_th = comm.size * size * flopfac / t_th / 1e9
        membw_th = comm.size * nbytes * memfac / t_th / 1024 / 1024 /1024

        pprint(" %25s ||    %8.2f |     %8.2f |    %8.2f |     %8.2f |  %8.2f |" % 
            (code, gflops_np, membw_np, gflops_th, membw_th, speedup))
        
