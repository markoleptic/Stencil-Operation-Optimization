/*
  This is the baseline implementation of a 1D Stencil operation.

  Parameters:

  m0 > 0: dimension of the original input and output vector(array) size
  k0 > 0: dimesnion of the original weights vector(array)

  float* input_sequential: pointer to original input data
  float* input_distributed: pointer to the input data that you have distributed
  across the system

  float* output_sequential:  pointer to original output data
  float* output_distributed: pointer to the output data that you have
  distributed across the system

  float* weights_sequential:  pointer to original weights data
  float* weights_distributed: pointer to the weights data that you have
  distributed across the system

  Functions:

  DISTRIBUTED_ALLOCATE_NAME(...): Allocate the distributed buffers.
  DISTRIBUTE_DATA_NAME(...): takes the sequential data and distributes it across
  the system. COMPUTE_NAME(...): Performs the stencil computation.
  COLLECT_DATA_NAME(...): Collect the distributed output and combine it back to
  the sequential one for testing. DISTRIBUTED_FREE_NAME(...): Free the
  distributed buffers that were allocated


  - richard.m.veras@ou.edu

  Original modulo removal:
    for (int i = 0; i < input_length; i++)
    {
      float res = 0.f;
      for (int j = 0; j < weights_length; ++j)
      {
        // if index exceeds length, overflow will occur. Wrap around by subtracting length
        int index = (i + j >= input_length) ? (i + j - input_length) : (i + j);
        res += input_distributed[index] * weights_distributed[j];
      }
      output_distributed[i] = res;
    }

*/
// Overflow condition, i + j will exceed input_length at some point in the j loop
// if (i + weights_length >= input_length)
// {
//   // The end index or non-wrapping elements
//   int end = 0;
//   // No wrapping yet
//   for (int j = 0; i + j < input_length; ++j)
//   {
//     res += input_distributed[j + i] * weights_distributed[j];
//     end = j;
//   }
//   // Start wrapping elements by subtracting input length
//   for (int j = end + 1; j < weights_length; ++j)
//   {
//     res += input_distributed[j + i - input_length] * weights_distributed[j];
//   }
// }
// else
// {
//   // Default behavior (no overflow from i to i + weights_length - 1)
//   res += input_distributed[j + i] * weights_distributed[j];
// }


    // for (int i = threshold; i < input_length; ++i)
    // {
    //   float res = 0.f;
    //   // Overflow condition, i + j will exceed input_length at some point in the j loop
    //   if (i + weights_length >= input_length)
    //   {
    //     // The end index or non-wrapping elements
    //     int end = 0;
    //     // No wrapping yet
    //     for (int j = 0; i + j < input_length; ++j)
    //     {
    //       res += input_distributed[j + i] * weights_distributed[j];
    //       end = j;
    //     }
    //     // Start wrapping elements by subtracting input length
    //     for (int j = end + 1; j < weights_length; ++j)
    //     {
    //       res += input_distributed[j + i - input_length] * weights_distributed[j];
    //     }
    //   }
    //   else
    //   {
    //     printf("else\n");
    //     // Default behavior (no overflow from i to i + weights_length - 1)
    //     for (int j = 0; j < weights_length; ++j)
    //     {
    //       res += input_distributed[j + i] * weights_distributed[j];
    //     }
    //   }
    //   output_distributed[i] = res;
    // }

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

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

void COMPUTE_NAME(int m0, int k0, float *input_distributed, float *weights_distributed, float *output_distributed)
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status status;
  int root_rid = 0;
  const int threshold = m0 - k0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if (rid == root_rid)
  {
    // No overflow
    for (int i = 0; i < threshold; ++i)
    {
      float res = 0.f;
      for (int j = 0; j < k0; ++j)
      {
        res += input_distributed[j + i] * weights_distributed[j];
      }
      output_distributed[i] = res;
    }

    // Overflow possible
    for (int i = threshold; i < m0; ++i)
    {
      float res = 0.f;
      int end = 0;
      for (int j = 0; i + j < m0; ++j)
      {
        res += input_distributed[j + i] * weights_distributed[j];
        end = j;
      }
      for (int j = end + 1; j < k0; ++j)
      {
        res += input_distributed[j + i - m0] * weights_distributed[j];
      }
      output_distributed[i] = res;
    }
  }
  else
  {
  }
}

// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME(int m0, int k0, float **input_distributed, float **weights_distributed,
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

void DISTRIBUTE_DATA_NAME(int m0, int k0, float *input_sequential, float *weights_sequential, float *input_distributed,
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

void COLLECT_DATA_NAME(int m0, int k0, float *output_distributed, float *output_sequential)
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

void DISTRIBUTED_FREE_NAME(int m0, int k0, float *input_distributed, float *weights_distributed,
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
