/*
  Timer harness for running a "function under test" for num_runs number of
  runs.

  - richard.m.veras@ou.edu
*/

#include <limits.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#include "timer.h"

// Function under test
extern void COMPUTE_NAME_REF( int m0, int k0,
			      float *input_distributed,
			      float *weights_distributed,
			      float *output_distributed );

extern void COMPUTE_NAME_TST( int m0, int k0,
			      float *input_distributed,
			      float *weights_distributed,
			      float *output_distributed );


extern void DISTRIBUTED_ALLOCATE_NAME_REF( int m0, int k0,
					   float **input_distributed,
					   float **weights_distributed,
					   float **output_distributed );

extern void DISTRIBUTED_ALLOCATE_NAME_TST( int m0, int k0,
					   float **input_distributed,
					   float **weights_distributed,
					   float **output_distributed );



extern void DISTRIBUTE_DATA_NAME_REF( int m0, int k0,
				      float *input_sequential,
				      float *weights_sequential,
				      float *input_distributed,
				      float *weights_distributed );

extern void DISTRIBUTE_DATA_NAME_TST( int m0, int k0,
				      float *input_sequential,
				      float *weights_sequential,
				      float *input_distributed,
				      float *weights_distributed );




extern void COLLECT_DATA_NAME_REF( int m0, int k0,
				   float *output_distributed,
				   float *output_sequential );

extern void COLLECT_DATA_NAME_TST( int m0, int k0,
				   float *output_distributed,
				   float *output_sequential );




extern void DISTRIBUTED_FREE_NAME_REF( int m0, int k0,
				       float *input_distributed,
				       float *weights_distributed,
				       float *output_distributed );

extern void DISTRIBUTED_FREE_NAME_TST( int m0, int k0,
				       float *input_distributed,
				       float *weights_distributed,
				       float *output_distributed );

extern void FUN_NAME_TST( int m, int n,
			  float *src,
			  int rs_s, int cs_s,
			  float *dst,
			  int rs_d, int cs_d);




void fill_buffer_with_random( int num_elems, float *buff )
{
    for(int i = 0; i < num_elems; ++i)
	buff[i] = ((float)(rand()-((RAND_MAX)/2)))/((float)RAND_MAX);
}

void fill_buffer_with_value( int num_elems, float val, float *buff )
{
    for(int i = 0; i < num_elems; ++i)
	buff[i] = val;
}

long pick_min_in_list(int num_trials, long *results)
{
  long current_min = LONG_MAX;

  for( int i = 0; i < num_trials; ++i )
    if( results[i] < current_min )
      current_min = results[i];

  return current_min;
}

void flush_cache()
{
  
  int size = 1024*1024*8;

  int *buff = (int *)malloc(sizeof(int)*size);
  int i, result = 0;
  volatile int sink;
  for (i = 0; i < size; i ++)
    result += buff[i];
  sink = result; /* So the compiler doesn't optimize away the loop */

  free(buff);
}

void time_function_under_test(int num_trials,
			      int num_runs_per_trial,
			      long *results, // results from each trial
			      int m0, int k0,
			      float *input_distributed,
			      float *weights_distributed,
			      float *output_distributed
			      )
{
  // Initialize the start and stop variables.
  TIMER_INIT_COUNTERS(stop, start);

  // Click the timer a few times so the subsequent measurements are more accurate
  MPI_Barrier(MPI_COMM_WORLD);
  TIMER_WARMUP(stop,start);

  // flush the cache
  flush_cache();
  MPI_Barrier(MPI_COMM_WORLD);
  
  for(int trial = 0; trial < num_trials; ++trial )
    {

	/*
	  Time code.
	*/
        // start timer
      TIMER_GET_CLOCK(start);

	////////////////////////
        // Benchmark the code //
	////////////////////////

	for(int runs = 0; runs < num_runs_per_trial; ++runs )
	  {
	    COMPUTE_NAME_TST( m0, k0,
			      input_distributed,
			      weights_distributed,
			      output_distributed );

	  }

	////////////////////////
        // End Benchmark      //
	////////////////////////

        
        // stop timer
	TIMER_GET_CLOCK(stop);

	// subtract the start time from the stop time
	TIMER_GET_DIFF(start,stop,results[trial])

    }

}


int scale_p_on_pos_ret_v_on_neg(int p, int v)
{
  if (v < 1)
    return -1*v;
  else
    return v*p;
}



int main( int argc, char *argv[] )
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  // What we will output to
  FILE *result_file;
  
  int num_trials = 30;
  int num_runs_per_trial = 1;

  // Problem parameters
  int min_size;
  int max_size;
  int step_size;

  int in_m0;
  int in_k0;

  // Get command line arguments
  if(argc == 1 )
    {
      min_size  = 16;
      max_size  = 256;
      step_size = 16;

      // defaults
      in_m0=1;
      in_k0=-3;

      // default to printing to stdout
      result_file = stdout;
    }
  else if(argc == 5 + 1 || argc == 6 + 1 )
    {
      min_size  = atoi(argv[1]);
      max_size  = atoi(argv[2]);
      step_size = atoi(argv[3]);

      in_m0=atoi(argv[4]);
      in_k0=atoi(argv[5]);

      // default to printing to stdout
      result_file = stdout;

      if(argc == 6 + 1)
	{
	  // we don't want every node opening the same file
	  // to write to.
	  if(rid == 0 )
	    {
	      result_file = fopen(argv[6],"w");
	    }
	  else
	    {
	      result_file = NULL;
	    }
	}
    }
  else
    {
      printf("usage: %s min max step m0 k0 [filename]\n",
	     argv[0]);
      exit(1);
    }

  // Print out the first line of the output in csv format
  if( rid == 0 )
    {
      /*root node */ 
      fprintf(result_file, "num_ranks,m0,k0,result\n");
    }
  else
    {/* all other nodes*/ }


  for( int p = min_size;
       p < max_size;
       p += step_size )
    {

      // input sizes
      int m0=scale_p_on_pos_ret_v_on_neg(p,in_m0);
      int k0=scale_p_on_pos_ret_v_on_neg(p,in_k0);

      // How big of a buffer do we need
      int input_sequential_sz  =m0;
      int output_sequential_sz =m0;
      int weights_sequential_sz=k0;

      float *input_sequential_tst   = (float *)malloc(sizeof(float)*input_sequential_sz);
      float *output_sequential_tst  = (float *)malloc(sizeof(float)*output_sequential_sz);
      float *weights_sequential_tst = (float *)malloc(sizeof(float)*weights_sequential_sz);


      if( rid == 0)
	{ /* root node */
	  // fill src_ref with random values
	  fill_buffer_with_random( input_sequential_sz, input_sequential_tst );
	  fill_buffer_with_random( weights_sequential_sz, weights_sequential_tst );
	  fill_buffer_with_value( output_sequential_sz, -1, output_sequential_tst );
	}
      else
	{/* all other nodes. */}

     
      // run the test
      float *input_distributed_tst;
      float *weights_distributed_tst;
      float *output_distributed_tst;

      // Allocate distributed buffers for the reference
      DISTRIBUTED_ALLOCATE_NAME_TST( m0, k0,
				     &input_distributed_tst,
				     &weights_distributed_tst,
				     &output_distributed_tst );

      // Distribute the sequential buffers 
      DISTRIBUTE_DATA_NAME_TST( m0, k0,
				input_sequential_tst,
				weights_sequential_tst,
				input_distributed_tst,
				weights_distributed_tst );
     
      // Perform the computation
      long *results = (long *)malloc(sizeof(long)*num_trials);

      time_function_under_test(num_trials,
			       num_runs_per_trial,
			       results, // results from each trial
			       m0, k0,
			       input_distributed_tst,
			       weights_distributed_tst,
			       output_distributed_tst
			       );

      long min_res = pick_min_in_list(num_trials, results);

      float nanoseconds = ((float)min_res)/(num_runs_per_trial);

      // Number of floating point operations
      long num_flops = m0*k0*2;

      // This gives us throughput as GFLOP/s
      float throughput =  num_flops / nanoseconds;

      free(results);
     
      // Collect the distributed data and write it to a sequential buffer
      COLLECT_DATA_NAME_TST( m0, k0,
			     output_distributed_tst,
			     output_sequential_tst );     
     
      // Finally free the buffers
      DISTRIBUTED_FREE_NAME_TST( m0, k0,
				 input_distributed_tst,
				 weights_distributed_tst,
				 output_distributed_tst );


      if( rid == 0)
	{
	  /* root node */

	  fprintf(result_file, "%i,%i,%i,%2.2f\n",
		  num_ranks,
		  m0,k0, throughput);
	}
      else
	{/* all other nodes */}


      // Free the sequential buffers
      free(input_sequential_tst);
      free(output_sequential_tst);
      free(weights_sequential_tst);
    }

  if( rid == 0)
    {
      /* root node */

      fclose(result_file);
    }
  else
    {/* all other nodes */}

      
  MPI_Finalize();
}

