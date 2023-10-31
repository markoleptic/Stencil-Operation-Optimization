#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME optimized_unrolled
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

#define UNROLL_FACTOR 1 // Adjust the unroll factor as needed


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
        for (int i0 = 0; i0 < m0; ++i0)
        {
            float res = 0.0f;

            // Loop unrolling with a variable unroll factor
            for (int p0 = 0; p0 < k0; p0 += UNROLL_FACTOR)
            {
                // Compute multiple iterations in each loop iteration
                for (int u = 0; u < UNROLL_FACTOR; ++u)
                {
                    int index = p0 + u + i0;
                    if (index >= m0)
                    {
                        index -= m0;
                    }
                    res += input_distributed[index] * weights_distributed[p0 + u];
                }
            }

            output_distributed[i0] = res;
        }
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
