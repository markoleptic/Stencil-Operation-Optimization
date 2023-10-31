/*
  This is the baseline implementation of a 1D Stencil operation.

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

  Functions:

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

#ifndef COMPUTE_NAME
#define COMPUTE_NAME optimized
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME optimized_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME optimized_collect
#endif

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME optimized_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME optimized_free
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
if (rid == root_rid)
    {
        int block_size = 1; // Adjust the block size as needed

        for (int i0 = 0; i0 < m0; i0 += block_size)
        {
            for (int b = 0; b < block_size; b++)
            {
                int input_idx = i0 + b;
                float res = 0.0f;
                for (int p0 = 0; p0 < k0; p0++)
                {
                    res += input_distributed[input_idx] * weights_distributed[p0];

                    // Check if input_idx exceeds array bounds and adjust
                    if (input_idx >= m0 - 1)
                    {
                        input_idx = 0;
                    }
                    else
                    {
                        input_idx++;
                    }
                }
                output_distributed[i0 + b] = res;
            }
        }
    }
    else
    {
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

    if (rid == root_rid)
    {
        *input_distributed = (float *)malloc(sizeof(float) * m0);
        *output_distributed = (float *)malloc(sizeof(float) * m0);
        *weights_distributed = (float *)malloc(sizeof(float) * k0);
    }
    else
    {
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

    if (rid == root_rid)
    {
        // Distribute the inputs
        for (int i0 = 0; i0 < m0; ++i0)
        {
            input_distributed[i0] = input_sequential[i0];
        }

        // Distribute the weights
        for (int p0 = 0; p0 < k0; ++p0)
        {
            weights_distributed[p0] = weights_sequential[p0];
        }
    }
    else
    {
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

    if (rid == root_rid)
    {
        // Collect the output
        for (int i0 = 0; i0 < m0; ++i0)
            output_sequential[i0] = output_distributed[i0];
    }
    else
    {
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
    }
}

