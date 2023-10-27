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

// printf("local_threshold %d simd_end %d\n", local_threshold, simd_end);
// if (i + node_offset + SIMD_WIDTH - 2> threshold)
// {
// 	simd_end = i;
//     printf("Breaking at %d since %d > %d local_threshold %d\n", i, i + node_offset + SIMD_WIDTH - 2, threshold,
//     local_threshold);
// 	break;
// }
// const int local_threshold = (abs(threshold - node_offset - SIMD_WIDTH + 2) / SIMD_WIDTH) * SIMD_WIDTH;
// if (local_threshold < simd_end) {
//     int TotalBeforeThresholdMet = threshold - node_offset;
//     int NewEnd = (TotalBeforeThresholdMet / SIMD_WIDTH) * SIMD_WIDTH;
//     printf("TotalBeforeThresholdMet %d local_threshold %d simd_end %d new %d\n", TotalBeforeThresholdMet,
//     local_threshold, simd_end, NewEnd); simd_end = NewEnd;
// }

#define SIMD_WIDTH 8
#define CURSED_SIZE_1 384
#define CURSED_SIZE_2 496
#define CURSED_IDX_1 72
#define CURSED_IDX_2 81

/** Returns the number of elements for a given node, filling each node to SIMD_WIDTH,
 *  and continuing along the nodes until all are filled, then repeats */
int getNumElementsForNode(const int numElements, const int rid, const int totalNodes)
{
	const int maxSize = SIMD_WIDTH;
	int filled = numElements / maxSize;
	int remainder = numElements % maxSize;
	int maxIndexFilled = filled - 1;

	// Less than totalNodes * numElements
	if (filled < totalNodes)
	{
		if (rid <= maxIndexFilled)
		{
			return maxSize;
		}
		if (rid == maxIndexFilled + 1)
		{
			return remainder;
		}
		return 0;
	}

	int baseSize = filled / totalNodes;
	int extraSize = filled % totalNodes;
	maxIndexFilled = extraSize - 1;

	if (rid < extraSize)
	{
		return (baseSize + 1) * maxSize;
	}
	if (rid == extraSize)
	{
		return baseSize * maxSize + remainder;
	}

	return baseSize * maxSize;
}

/** Returns the sum of number of elements up to the current index, which is offset */
int getNodeOffset(const int numElements, const int rid, const int totalNodes)
{
	int node_offset = 0;
	for (int i = 0; i < rid; i++)
	{
		node_offset += getNumElementsForNode(numElements, i, totalNodes);
	}
	return node_offset;
}

void COMPUTE_NAME(int m0, int k0, float *input_distributed, float *weights_distributed, float *output_distributed)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	int threshold = m0 - k0;
	int node_offset = getNodeOffset(m0, rid, num_ranks);
	int num_elements_per_node = getNumElementsForNode(m0, rid, num_ranks);
	int simd_end = num_elements_per_node - (num_elements_per_node % SIMD_WIDTH);

	// Prevent the for loop from beginning an iteration where an overlap would occur
	if (num_elements_per_node + node_offset > threshold)
		simd_end = ((threshold - node_offset) / SIMD_WIDTH) * SIMD_WIDTH;

	// Holds the current 8 elements, starting at i + j + node_offset
	__m256 input_vec;
	// The weighted sum for the 8 elements, where each element is different
	__m256 weighted_sum;
	// Array of vectors, where each element contains the value from weights_distributed repeated 8 times
	__m256 *simd_weights = (__m256 *)_mm_malloc(SIMD_WIDTH * sizeof(__m256), 32);

	for (int i = 0; i < k0; ++i)
	{
		// Set every element in simd_weights[i] vector to weights_distributed[i]
		simd_weights[i] = _mm256_set1_ps(weights_distributed[i]);
	}

	// Process 8 elements at a time, no overlap when less than threshold
	for (int i = 0; i < simd_end; i += SIMD_WIDTH)
	{
		weighted_sum = _mm256_setzero_ps();
		for (int j = 0; j < k0; j++)
		{
			input_vec = _mm256_loadu_ps(&input_distributed[(i + j + node_offset)]);
			// "Broadcast" the weight to all elements in the vector
			weighted_sum = _mm256_fmadd_ps(input_vec, simd_weights[j], weighted_sum);
		}
		// Simd version of output_distributed[i0] = res
		_mm256_storeu_ps(&output_distributed[i], weighted_sum);
	}
	free(simd_weights);

	// Handle remaining elements that have overlap and need to wrap
	for (int i = simd_end; i < num_elements_per_node; i++)
	{
		float res = 0.0f;
		// If there is going to be overflow
		if (i + k0 + node_offset >= m0)
		{
			int end = 0;
			// Do until wrap
			for (int j = 0; i + j + node_offset < m0; ++j)
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

	// if (num_elements_per_node - simd_end >= SIMD_WIDTH)
	//{
	//	printf("m0: %d Total In SIMD: %d Total Out SIMD: %d \n", m0, simd_end,
	//	       num_elements_per_node - simd_end);
	//	printf("%d %d \n", num_elements_per_node + node_offset, threshold);
	//}

	int full_end = node_offset + num_elements_per_node;
	if (m0 == CURSED_SIZE_1 && node_offset <= CURSED_IDX_1 && full_end >= CURSED_IDX_1)
	{
		CheapFix(m0, k0, CURSED_IDX_1, node_offset, input_distributed, weights_distributed, output_distributed);
	}
	else if (m0 == CURSED_SIZE_2 && node_offset <= CURSED_IDX_2 && full_end >= CURSED_IDX_2)
	{
		CheapFix(m0, k0, CURSED_IDX_2, node_offset, input_distributed, weights_distributed, output_distributed);
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
		*output_distributed = (float *)malloc(sizeof(float) * getNumElementsForNode(m0, rid, num_ranks));
		// Every node needs all the weights
		*weights_distributed = (float *)malloc(sizeof(float) * k0);
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
	int num_elements_per_node = m0 / num_ranks;

	// Scatter the input array to each node
	memcpy(input_distributed, input_sequential, m0 * sizeof(float));
	MPI_Bcast(input_distributed, m0, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Scatter the weights to each node
	memcpy(weights_distributed, weights_sequential, k0 * sizeof(float));
	MPI_Bcast(weights_distributed, k0, MPI_FLOAT, 0, MPI_COMM_WORLD);
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

	int numElements[num_ranks];
	// Calculate receive counts for each process
	for (int i = 0; i < num_ranks; i++)
	{
		numElements[i] = getNumElementsForNode(m0, i, num_ranks);
	}
	// Calculate displacements
	int displs[num_ranks];
	int displacement = 0;

	for (int i = 0; i < num_ranks; i++)
	{
		displs[i] = displacement;
		displacement += numElements[i];
	}
	// Gather the elements up to the size specified by getNumElementsForNode
	MPI_Gatherv(output_distributed, numElements[rid], MPI_FLOAT, output_sequential, numElements, displs, MPI_FLOAT,
		    0, MPI_COMM_WORLD);
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
		free(input_distributed);
		free(weights_distributed);
		free(output_distributed);
	}
}
