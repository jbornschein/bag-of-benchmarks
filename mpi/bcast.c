#include <mpi.h>
#include <stdio.h>

#define RUNS (50)
#define SIZE (1 << 23)

int main(int argc, char **argv)
{
    int r, rank, size;
    double *in;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf( "%d", SIZE );


    in = malloc( SIZE*sizeof(double) );
    
    MPI_Barrier(MPI_COMM_WORLD);
    for(r=0; r<RUNS; r++) {
        MPI_Bcast(in, SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
