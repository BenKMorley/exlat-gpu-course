/*
 * This is a simple CUDA code that negates an array of integers.
 * It introduces the concepts of device memory management, and
 * kernel invocation.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010 
 */

#include <stdio.h>
#include <stdlib.h>

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

/* The number of integer elements in the array */
#define ARRAY_SIZE 256

/*
 * The number of CUDA blocks and threads per block to use.
 * These should always multiply to give the array size.
 * For the single block kernel, NUM_BLOCKS should be 1 and
 * THREADS_PER_BLOCK should be the array size
 */

#define NUM_BLOCKS  1
#define THREADS_PER_BLOCK 256

/* The actual array negation kernel (basic single block version) */
__global__ void negate(int * d_a) {
  /* Part 2B: negate an element of d_a */
  int i = threadIdx.x;
  d_a[i] = -1.0 * d_a[i];
}

/* Multi-block version of kernel for part 2C */
#define NUM_BLOCKS  4
#define THREADS_PER_BLOCK 64

__global__ void negate_multiblock(int *d_a) {
  /* Part 2C: negate an element of d_a, using multiple blocks this time */
  int i = blockIdx.x;
  int j = threadIdx.x;
  int index = i * blockDim.x + j;

  d_a[index] = -d_a[index];
}

/* Main routine */

int main(int argc, char *argv[]) {
  int *h_a, *h_out;
  int *d_a;

  int i;
  size_t sz = ARRAY_SIZE * sizeof(int);

  /* Print device details */
  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceNum);
  printf("Device name: %s\n", prop.name);

  /* h_a holds the input array, h_out holds the result */
  h_a = (int *) malloc(sz);
  h_out = (int *) malloc(sz);

  /* Part 1A: allocate device memory */
  cudaMalloc(&d_a, sz);

  /* initialise host arrays */
  for (i = 0; i < ARRAY_SIZE; i++) {
    h_a[i] = i;
    h_out[i] = 0;
  }

  printf("Hello World 1");

  /* Part 1B: copy host array h_a to device array d_a */
  cudaMemcpy(h_a, d_a, sz * sizeof(int), cudaMemcpyDeviceToHost);

  /* Part 2A: configure and launch kernel (un-comment and complete) */
  dim3 blocksPerGrid(1, 1, 1);
  dim3 threadsPerBlock(256, 1, 1);
  negate <<<blocksPerGrid, threadsPerBlock>>>(d_a);

  /* wait for all threads to complete and check for errors */
  cudaDeviceSynchronize();
  checkCUDAError("kernel invocation");

  /* Part 1C: copy device array d_a to host array h_out */
  cudaMemcpy(h_out, d_a, sz * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("memcpy");

  /* print out the result */
  printf("Results: ");
  for (i = 0; i < ARRAY_SIZE; i++) {
    printf("%d, ", h_out[i]);
  }

  printf("\n\n");

  /* Part 1D: free d_a */
  cudaFree(d_a);

  /* free host buffers */
  free(h_a);
  free(h_out);

  return 0;
}

/* Utility function to check for and report CUDA errors */

void checkCUDAError(const char * msg) {

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
