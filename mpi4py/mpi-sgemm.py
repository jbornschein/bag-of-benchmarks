#!/usr/bin/env python 

"""

Implementation of a MPI parallel A*B=C matrix-matrix multiplication.

Each rank locally stores a tile of the matrices A, B and C; Communication 
is overlayed with computation. 

"""

from __future__ import division

import sys
sys.path.insert(0, "../pylib")

import numpy as np 
from mpi4py import MPI
from time import time

from parutils import pprint

#=============================================================================#

my_N = 1000
my_M = 1000

#=============================================================================#

NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    mpi_rows = int(np.floor(np.sqrt(comm.size)))
    mpi_cols = comm.size // mpi_rows
    if mpi_rows*mpi_cols > comm.size:
        mpi_cols -= 1
    if mpi_rows*mpi_cols > comm.size:
        mpi_rows -= 1

    pprint("="*78 )
    pprint("Running %d parallel processes (ranks)" % (comm.size) )
    pprint("Creating a %d x %d processor grid..." % (mpi_rows, mpi_cols) )

    # Create a 2d cartesian grid with periodic boundary conditions
    ccomm = comm.Create_cart( (mpi_rows, mpi_cols), periods=(True, True), reorder=True)

    my_mpi_row, my_mpi_col = ccomm.Get_coords( ccomm.rank ) 
    
    # Identifiy our neighbours on the grid
    neigh = [0,0,0,0]
    neigh[NORTH], neigh[SOUTH] = ccomm.Shift(0, 1)
    neigh[EAST],  neigh[WEST]  = ccomm.Shift(1, 1)

    # Create matrices
    my_A = np.random.normal(size=(my_N, my_M)).astype(np.float32)
    my_B = np.random.normal(size=(my_N, my_M)).astype(np.float32)
    my_C = np.zeros_like(my_A)

    tile_A = my_A
    tile_B = my_B
    tile_A_ = np.empty_like(my_A)
    tile_B_ = np.empty_like(my_A)
    req = [None, None, None, None]

    t0 = time()
    for r in xrange(mpi_rows-1):
        # Initiate async. tile shift communication
        req[EAST]  = ccomm.Isend(tile_A , neigh[EAST])
        req[WEST]  = ccomm.Irecv(tile_A_, neigh[WEST])
        req[SOUTH] = ccomm.Isend(tile_B , neigh[SOUTH])
        req[NORTH] = ccomm.Irecv(tile_B_, neigh[NORTH])

        # Compute on current tile
        my_C += np.dot(tile_A, tile_B)

        # Wait for communication to finish
        req[0].Waitall(req)

        # Swap buffers
        (tile_A, tile_A_) = (tile_A_, tile_A)
        (tile_B, tile_B_) = (tile_B_, tile_B)

    # Last iteration without communication
    my_C += np.dot(tile_A, tile_B)

    t_total = time()-t0
    comm.barrier()
        
    t_serial = 0.
    if comm.rank == 0:
        t0 = time()
        np.dot(tile_A, tile_B)
        t_serial = time()-t0
    t_serial = comm.bcast(t_serial)

    pprint(78*"=")
    pprint("Computed (serial) %d x %d x %d in  %6.2f seconds" % (my_M, my_M, my_N, t_serial))
    pprint(" ... expecting parallel computation to take %6.2f seconds" % (mpi_rows*mpi_rows*mpi_cols*t_serial / comm.size))
    pprint("Computed (parallel) %d x %d x %d in        %6.2f seconds" % (mpi_rows*my_M, mpi_rows*my_M, mpi_cols*my_N, t_total))
    

    comm.barrier()
    





