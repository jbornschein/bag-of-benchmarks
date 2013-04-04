#!/bin/sh
#SBATCH --mail-type ALL
#SBATCH --partition=parallel

mpirun python mpi-sgemm.py

