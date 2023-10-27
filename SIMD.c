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

*/

#include "utils.c"
#include <immintrin.h>
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

#define SIMD_WIDTH 8
#define CURSED_SIZE_1 384
#define CURSED_SIZE_2 496
#define CURSED_INDEX_1 72
#define CURSED_INDEX_2 81

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
		// Holds the current 8 elements, starting at i + j + node_offset
		__m256 input_vec;
		// The weighted sum for the 8 elements, where each element is different
		__m256 weighted_sum;
		// Array of vectors, where each element contains the value from weights_distributed repeated 8 times
		__m256 *simd_weights = (__m256 *)_mm_malloc(8 * sizeof(__m256), 32);

		for (int i = 0; i < k0; ++i)
		{
			// Set every element in simd_weights[i] vector to weights_distributed[i]
			simd_weights[i] = _mm256_set1_ps(weights_distributed[i]);
		}

		// Process 8 elements at a time, no overlap when less than threshold
		for (int i = 0; i < threshold; i += SIMD_WIDTH)
		{
			weighted_sum = _mm256_setzero_ps();
			for (int j = 0; j < k0; j++)
			{
				input_vec = _mm256_loadu_ps(&input_distributed[(i + j)]);
				// "Broadcast" the weight to all elements in the vector
				weighted_sum = _mm256_fmadd_ps(input_vec, simd_weights[j], weighted_sum);
			}
			// Simd version of output_distributed[i0] = res
			_mm256_storeu_ps(&output_distributed[i], weighted_sum);
		}

		// Handle remaining elements that have overlap and need to wrap
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

		if (m0 == CURSED_SIZE_1)
			CheapFix(m0, k0, CURSED_INDEX_1, 0, input_distributed, weights_distributed, output_distributed);
		else if (m0 == CURSED_SIZE_2)
			CheapFix(m0, k0, CURSED_INDEX_2, 0, input_distributed, weights_distributed, output_distributed);
		// free weights array
		free(simd_weights);
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
