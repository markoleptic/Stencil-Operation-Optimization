/*
  This is the 1D stencil using MPI Send and Receive with
  loop index set splitting and unswitching.

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
	int node_offset = rid * num_elements_per_node;

	for (int i = 0; i < num_elements_per_node; i++)
    {
      float res = 0.0f;
      // If there is going to be overflow
      if (i + k0 + node_offset >= m0)
      {
        int end = 0;
        // Do until wrap
        for (int j = 0; (i + j) + node_offset < m0; ++j)
        {
          res += input_distributed[i + j + node_offset] * weights_distributed[j];
          end = j;
        }
        // Do wrapped elements
        for (int j = end + 1; j < k0; ++j)
        {
          res += input_distributed[i + j + node_offset - m0] * weights_distributed[j];
        }
      }
      else
      {
        for (int j = 0; j < k0; ++j)
        {
          res += input_distributed[i + j + node_offset] * weights_distributed[j];
        }
      }
      output_distributed[i] = res;
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

	// Broadcast the input array to each node
	if (rid == root_rid){
		memcpy(input_distributed, input_sequential, m0 * sizeof(float));
		for (int i = 0; i < num_ranks; i++)
		{
			if (i == root_rid) continue;
			MPI_Send(input_distributed, m0, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Recv(input_distributed, m0, MPI_FLOAT, root_rid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Broadcast the weights to each node
	if (rid == root_rid){
		memcpy(weights_distributed, weights_sequential, k0 * sizeof(float));
		for (int i = 0; i < num_ranks; i++)
		{
			if (i == root_rid) continue;
			MPI_Send(weights_distributed, k0, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		}
	}
	else{
		MPI_Recv(weights_distributed, k0, MPI_FLOAT, root_rid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

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
	if (rid == root_rid)
	{
		for (int i = 0; i < num_ranks; i++)
		{
			if (i == root_rid)
			{
				for (int j = 0; j < num_elements_per_node; j++)
				{
					output_sequential[j + num_elements_per_node * root_rid] = output_distributed[j];
				}
			}
			else
			{
				MPI_Recv(&output_sequential[num_elements_per_node * i], num_elements_per_node, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	}
	else
	{
		MPI_Send(output_distributed, num_elements_per_node, MPI_FLOAT, root_rid, 0, MPI_COMM_WORLD);
	}

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
