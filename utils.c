#include <immintrin.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void printOutputDistributed(const float *array, const int size, const char *fileName)
{
  FILE *file = fopen(fileName, "a");
  if (file == NULL)
  {
    perror("Failed to open file");
    return;
  }
  for (int i = 0; i < size; i++)
  {
    fprintf(file, "%.1f\n", array[i]);
  }
  fclose(file);
}

void printOutputDistributed2(const float *array, const float *array2, const int size, const char *fileName)
{
  FILE *file = fopen(fileName, "w");
  if (file == NULL)
  {
    perror("Failed to open file");
    return;
  }
  for (int i = 0; i < size; i++)
  {
    fprintf(file, "Ref: %.3f New: %.3f\n", array[i], array2[i]);
  }
  fclose(file);
}

void printDistributedDiff(const float *array, const float *array2, const int size, const char *fileName)
{
  FILE *file = fopen(fileName, "w");
  if (file == NULL)
  {
    perror("Failed to open file");
    return;
  }
  for (int i = 0; i < size; i++)
  {
    float max_diff = 0.0;
    float sum = fabs(array[i] + array2[i]);
    float diff = fabs(array[i] - array2[i]);
    float res = 0.0f;
    if (sum == 0.0f)
      res = diff;
    else
      res = 2 * diff / sum;
    if (res > max_diff)
      max_diff = res;

    fprintf(file, "Diff at index %d: %f \n", i, max_diff);
  }
  fclose(file);
}

void print_m256(__m256 vec)
{
  float values[8];
  _mm256_storeu_ps(values, vec);
  printf("__m256: ");
  for (int i = 0; i < 8; i++)
  {
    printf("%f ", values[i]);
  }
  printf("\n");
}

void print_weights(const float *array)
{
  printf("Normal: ");
  for (int i = 0; i < 8; i++)
  {
    printf("%f ", array[i]);
  }
  printf("\n");
}