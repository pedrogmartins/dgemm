# Optimizing DGEMM Algotrith

Project for the course CS 267 (Applications of Parallel Computers) at UC Berkeley in Spring 2023. Optimized the DGEMM algorithm implementing loop reordering, multi-level blocking to fully utilize L1 and L2 cache, associated padding to tolerate any matrix size, repacking for improved memory access efficiency, and, finally,  microkernel SIMD computations with the Perlmutter NVIDIA hardware.

Authors: Pedro G. Martins with contributions from Danush Reddy.  
