#!/usr/bin/env bash
#
# Modify this file to point to the variants you have created.
# Three variants are prepopulated as examples.
#
# -richard.m.veras@ou.edu


######################################
# DO NOT CHANGE THIS FOLLOWING LINE: #
OP_BASELINE_FILE="baseline_op.c"    #
######################################

############################################
# HOWEVER, CHANGE THESE LINES:             #
# Replace the filenames with your variants #
############################################

OP_SUBMISSION_VAR01_FILE="SplittingAndUnswitching.c"
OP_SUBMISSION_VAR02_FILE="MPI_Collective.c"
OP_SUBMISSION_VAR03_FILE="MPI_Collective_v2.c"
OP_SUBMISSION_VAR04_FILE="SIMD.c"
OP_SUBMISSION_VAR05_FILE="MPI_Send_Recv.c"
OP_SUBMISSION_VAR06_FILE="TwoParallel.c"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC=mpicc
# CFLAGS="-std=c99 -O2"
CFLAGS="-std=c99 -O2 -mavx2 -mfma -fopenmp"

