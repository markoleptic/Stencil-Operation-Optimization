#!/bin/bash
#SBATCH --partition=debug_5min
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=20
#SBATCH --output=./jobname_%J_stdout.txt
#SBATCH --error=./jobname_%J_stderr.txt
#SBATCH --time=5:00
#SBATCH --job-name=cs4473_cs5473_lab02
#SBATCH --mail-user=chasecummins1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=./


# Increments to use in the local tests
MIN=32
MAX=2048
STEP=32

# These are sufficiently small
KSMALL=3
KMEDIUM=7
KLARGE=11

module load OpenMPI

# Verify
mpiexec ./run_test_op_var01.x 32 2048 32 1 -7  result_verification_op_var01_k7.csv
mpiexec ./run_test_op_var02.x 32 2048 32 1 -7  result_verification_op_var02_k7.csv
mpiexec ./run_test_op_var03.x 32 2048 32 1 -7  result_verification_op_var03_k7.csv
mpiexec ./run_test_op_var04.x 32 2048 32 1 -7  result_verification_op_var04_k7.csv

echo "Number of FAILS: `grep "FAIL" result_verification_op_*.csv|wc -l`"

# Bench the results
mpiexec ./run_bench_op_var01.x 32 2048 32 1 -7  result_bench_op_var01_k7.csv
mpiexec ./run_bench_op_var02.x 32 2048 32 1 -7  result_bench_op_var02_k7.csv
mpiexec ./run_bench_op_var03.x 32 2048 32 1 -7  result_bench_op_var03_k7.csv
mpiexec ./run_bench_op_var04.x 32 2048 32 1 -7  result_bench_op_var04_k7.csv

# plot the results
module unload OpenMPI
module load scikit-learn/0.23.1-foss-2020a-Python-3.8.2
module load matplotlib/3.2.1-foss-2019b-Python-3.8.2
./plotter_multi.py "Results of Stencil Computation on Schooner" "PLOT_schooner.png" result_bench_op_var01_k7.csv result_bench_op_var02_k7.csv result_bench_op_var03_k7.csv result_bench_op_var04_k7.csv

