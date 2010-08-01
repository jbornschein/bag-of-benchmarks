#include <mpi.h>
#include <stdio.h>

#define RUNS (1000)
#define SIZE (144*256)

int main(int argc, char **argv)
{
	int rank, size;
	double out[SIZE], in[SIZE];
    int r;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	for(r=0; r<RUNS; r++) {
		MPI_Allreduce(&out, &in, SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	}
}
