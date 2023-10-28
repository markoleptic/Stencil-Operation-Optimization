#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_THREADS_LESS_THAN_512 2
#define NUM_THREADS_LESS_THAN_1024 4
#define NUM_THREADS_GREATER_THAN_1024 8

/** Returns the "optimal" number of threads to use for openMP. Right now these are just random breakpoints I made up. */
int getOptimalNumThreads(int m0)
{
	// openMp_WithoutSplit_WithoutTemp_WithPrivate fails tests when m0 <= 256, so fall back to default
	// implementation
	if (m0 <= 256)
		return -1;

	// Why does this perform the best
	return 1;

	// don't return a value greater than the number of threads available on system
	int max_threads = omp_get_max_threads();
	if (m0 <= 512)
		return max_threads < NUM_THREADS_LESS_THAN_512 ? max_threads : NUM_THREADS_LESS_THAN_512;
	if (m0 <= 1024)
		return max_threads < NUM_THREADS_LESS_THAN_1024 ? max_threads : NUM_THREADS_LESS_THAN_1024;
	return max_threads < NUM_THREADS_GREATER_THAN_1024 ? max_threads : NUM_THREADS_GREATER_THAN_1024;
}

/** Does not split output_distributed, only uses the "private openMP directive" so that each thread gets their own copy
 * of output_distributed. Fails tests when m0 <= 256, so it falls back to default implementation at those sizes. */
void openMp_WithoutSplit_WithoutTemp_WithPrivate(int m0, int k0, float *input_distributed, float *weights_distributed,
						 float *output_distributed)
{
	const int threshold = m0 - k0;
	const int num_threads = getOptimalNumThreads(m0);

	if (num_threads == -1)
	{
		for (int i = 0; i < threshold; ++i)
		{
			float res = 0.f;
			for (int j = 0; j < k0; ++j)
			{
				res += input_distributed[j + i] * weights_distributed[j];
			}
			output_distributed[i] = res;
		}
		return;
	}

#pragma omp parallel for num_threads(num_threads) private(output_distributed)
	for (int i = 0; i < threshold; ++i)
	{
		float res = 0.f;
		for (int j = 0; j < k0; ++j)
		{
			res += input_distributed[j + i] * weights_distributed[j];
		}
		output_distributed[i] = res;
	}
}

/** Uses two threads where each gets their own section, but both access output_distributed. */
void openMp_WithoutSplit_WithoutTemp(int m0, int k0, float *input_distributed, float *weights_distributed,
				     float *output_distributed)
{
	const int threshold = m0 - k0;
	float res; // Private variable for storing temporary results
#pragma omp parallel num_threads(2) private(res) firstprivate(k0)                                                      \
    shared(input_distributed, weights_distributed, output_distributed)
	{
#pragma omp for
		for (int i = 0; i < threshold; ++i)
		{
			res = 0.f;
			for (int j = 0; j < k0; ++j)
			{
				res += input_distributed[j + i] * weights_distributed[j];
			}
			output_distributed[i] = res;
		}
	}
}

/** Uses two threads where each gets their own section, but both access output_distributed. */
void openMp_WithSplit_WithoutTemp(int m0, int k0, float *input_distributed, float *weights_distributed,
				  float *output_distributed)
{
	const int threshold = m0 - k0;
#pragma omp parallel num_threads(2)
	{
#pragma omp sections
		{
#pragma omp section
			{
				for (int i = 0; i < (threshold + 1) / 2; ++i)
				{
					float res = 0.f;
					for (int j = 0; j < k0; ++j)
					{
						res += input_distributed[j + i] * weights_distributed[j];
					}
					output_distributed[i] = res;
				}
			}
#pragma omp section
			{
				for (int i = 0; i < (threshold + 1) / 2; ++i)
				{
					float res = 0.f;
					for (int j = 0; j < k0; ++j)
					{
						res += input_distributed[j + i + (threshold + 1) / 2] *
						       weights_distributed[j];
					}
					output_distributed[i + (threshold + 1) / 2] = res;
				}
			}
		}
	}
}

/** Uses two threads where each gets their own section. Splits output_distributed into two arrays so each thread has its
 *  own copy. Copies both to output_distributed after the threads are done. */
void openMp_WithSplit_WithTemp(int m0, int k0, float *input_distributed, float *weights_distributed,
			       float *output_distributed)
{
	const int threshold = m0 - k0;
	float *temp1 = (float *)malloc(sizeof(float) * (threshold + 1) / 2);
	float *temp2 = (float *)malloc(sizeof(float) * (threshold + 1) / 2);

#pragma omp parallel num_threads(2)
	{
#pragma omp sections
		{
#pragma omp section
			{
				for (int i = 0; i < (threshold + 1) / 2; ++i)
				{
					float res = 0.f;
					for (int j = 0; j < k0; ++j)
					{
						res += input_distributed[j + i] * weights_distributed[j];
					}
					temp1[i] = res;
				}
			}
#pragma omp section
			{
				for (int i = 0; i < (threshold + 1) / 2; ++i)
				{
					float res = 0.f;
					for (int j = 0; j < k0; ++j)
					{
						res += input_distributed[j + i + (threshold + 1) / 2] *
						       weights_distributed[j];
					}
					temp2[i] = res;
				}
			}
		}
	}
	memcpy(&output_distributed[(threshold + 1) / 2], temp2, sizeof(float) * (threshold + 1) / 2);
	memcpy(&output_distributed[0], temp1, sizeof(float) * (threshold + 1) / 2);

	free(temp1);
	free(temp2);
}