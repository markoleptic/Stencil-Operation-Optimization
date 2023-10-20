/*
  This is the 1D stencil using MPI Collective Communictions.

  Parameters:

  m0 > 0: dimension of the original input and output vector(array) size
  k0 > 0: dimesnion of the original weights vector(array)

  float* input_sequential: pointer to original input data
  float* input_distributed: pointer to the input data that you have distributed across
  the system

  float* output_sequential:  pointer to original output data
  float* output_distributed: pointer to the output data that you have distributed across
  the system

  float* weights_sequential:  pointer to original weights data
  float* weights_distributed: pointer to the weights data that you have distributed across
  the system

  Functions: Modify these however you please.

  DISTRIBUTED_ALLOCATE_NAME(...): Allocate the distributed buffers.
  DISTRIBUTE_DATA_NAME(...): takes the sequential data and distributes it across the system.
  COMPUTE_NAME(...): Performs the stencil computation.
  COLLECT_DATA_NAME(...): Collect the distributed output and combine it back to the sequential
  one for testing.
  DISTRIBUTED_FREE_NAME(...): Free the distributed buffers that were allocated


  - richard.m.veras@ou.edu

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME baseline_collect
#endif

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME baseline_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME baseline_free
#endif

void COMPUTE_NAME(int m0, int k0,
				  float *input_distributed,
				  float *weights_distributed,
				  float *output_distributed)

{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	int num_elements_per_node = m0 / num_ranks;

	for (int i0 = 0; i0 < num_elements_per_node; ++i0)
	{
		float res = 0.0f;
		for (int p0 = 0; p0 < k0; ++p0)
		{
			res += input_distributed[((i0 + p0) + (rid * num_elements_per_node)) % m0] * weights_distributed[p0];
		}
		output_distributed[i0] = res;
	}
}

// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME(int m0, int k0,
							   float **input_distributed,
							   float **weights_distributed,
							   float **output_distributed)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	int num_elements_per_node = m0 / num_ranks;

	if (rid == root_rid)
	{
		*input_distributed = (float *)malloc(sizeof(float) * m0);
		*output_distributed = (float *)malloc(sizeof(float) * m0);
		*weights_distributed = (float *)malloc(sizeof(float) * k0);
	}
	else
	{
		// Every node gets all the input
		*input_distributed = (float *)malloc(sizeof(float) * m0);
		// Each node only computes a certain number of answers though
		*output_distributed = (float *)malloc(sizeof(float) * num_elements_per_node);
		// Every node needs all the weights
		*weights_distributed = (float *)malloc(sizeof(float) * k0);
	}
}

void DISTRIBUTE_DATA_NAME(int m0, int k0,
						  float *input_sequential,
						  float *weights_sequential,
						  float *input_distributed,
						  float *weights_distributed)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	int num_elements_per_node = m0 / num_ranks;

	// Scatter the input array to each node
	memcpy(input_distributed, input_sequential, m0 * sizeof(float));
	MPI_Bcast(input_distributed, m0, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Scatter the weights to each node
	memcpy(weights_distributed, weights_sequential, k0 * sizeof(float));
	MPI_Bcast(weights_distributed, k0, MPI_FLOAT, 0, MPI_COMM_WORLD);

}

void COLLECT_DATA_NAME(int m0, int k0,
					   float *output_distributed,
					   float *output_sequential)
{

	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	int num_elements_per_node = m0 / num_ranks;

	// Gather all the elements from the nodes
	MPI_Gather(output_distributed, num_elements_per_node, MPI_FLOAT,
			   output_sequential, num_elements_per_node, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void DISTRIBUTED_FREE_NAME(int m0, int k0,
						   float *input_distributed,
						   float *weights_distributed,
						   float *output_distributed)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{
		free(input_distributed);
		free(weights_distributed);
		free(output_distributed);
	}
	else
	{
		free(input_distributed);
		free(weights_distributed);
		free(output_distributed);
	}
}
