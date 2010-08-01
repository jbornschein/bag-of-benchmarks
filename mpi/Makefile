
all: bcast allreduce

bcast: bcast.c
	mpicc bcast.c -o bcast

allreduce: allreduce.c
	mpicc allreduce.c -o allreduce

clean:
	rm -f bcast allreduce

