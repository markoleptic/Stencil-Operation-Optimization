# Stencil Operation Optimization

This repository implements various parallel optimization techniques for a 1D stencil operation. The starting point for all variants is `baseline_op.c`.

## Optimization Techniques

- **SIMD.c:** Utilizes AVX2 to parallelize the code in blocks of eight.
- **LoopBlock.c:** Applies loop blocking.
- **LoopUnrolled.c:** Applies loop unrolling.
- **OpenMP.c:** Applies OpenMP parallel regions and threads.
- **SplittingAndUnswitching.c:** Removes the modulo operator and separates the loop into two.
- **MPI_Collective.c, MPI_Collective_v2.c, and MPI_Send_Recv.c:** Use MPI for shared memory parallelism.
- **TwoParallel.c, ThreeParallel.c, and FourParallel.c:** Combine the previous forms of parallelism.

## File Descriptions

- **baseline_op.c:** The starting point for all variants.
- **writeup.pdf:** Contains a full description and results of the implemented optimization techniques.

