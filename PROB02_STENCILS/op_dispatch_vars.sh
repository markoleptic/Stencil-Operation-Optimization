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
OP_SUBMISSION_VAR01_FILE="tuned_variant01_op.c"
OP_SUBMISSION_VAR02_FILE="tuned_variant02_op.c"
OP_SUBMISSION_VAR03_FILE="tuned_variant03_op.c"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC=mpicc
# CFLAGS="-std=c99 -O2"
CFLAGS="-std=c99 -O2 -mavx2 -mfma"

