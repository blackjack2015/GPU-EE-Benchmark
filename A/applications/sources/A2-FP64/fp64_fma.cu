#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define VECTOR_SIZE 655360
#define TILE_DIM 1024
#define COMP_ITERATIONS 1024000

__global__ void simpleKernel(double *A, double *C1, int size, int compute_iters, int tile_dim)
{
    int xIndex = blockIdx.x * tile_dim + threadIdx.x;
    double ra, rc;

    if (xIndex < size) {
        ra=A[xIndex];
        rc=A[size - xIndex - 1];
        // rb=A[xIndex];
        for (int i=0;i<compute_iters;i++) {
            //add_2regs
            rc += ra * ra;
        }
        C1[xIndex]=rc;
    }
}


int main(int argc, char **argv) {
    int compute_iters=COMP_ITERATIONS,
        vector_size=VECTOR_SIZE,
        tile_dim=TILE_DIM;

    // execution configuration parameters
    dim3 grid(vector_size/tile_dim, 1), threads(tile_dim, 1);

    // CUDA events
    cudaEvent_t start, stop;

    // allocate host memory
    size_t item_size = sizeof(double);
    size_t mem_size = item_size * vector_size;
    double *h_iA = (double *) malloc(mem_size);
    double *h_oC1 = (double *) malloc(mem_size);
    double elem = 2.7;
    // initalize host data
    for (int i = 0; i < vector_size; ++i)
    {
        // h_iA[i] = (float) i+3;
        h_iA[i] = elem;
    }
    // allocate device memory
    double *d_iA, *d_oC1;
    cudaMalloc((void **) &d_iA, mem_size);
    cudaMalloc((void **) &d_oC1, mem_size);

    // copy host data to device
    cudaMemcpy(d_iA, h_iA, mem_size, cudaMemcpyHostToDevice);

    // print out common data for all kernels
    printf("\nVector size: %d  TotalBlocks: %d blockSize: %d\n\n", vector_size, grid.x, threads.x);

    // initialize events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int secs = -1;
    int cIterations = 10;

    // Get environment variables
    if (getenv("secs") != NULL)
        secs = atoi(getenv("secs"));

    double total_time = 0;
    float kernelTime;
    for(int i = -10; i < cIterations; i++){
        cudaEventRecord(start, 0);
        simpleKernel<<<grid, threads>>>(d_iA, d_oC1, vector_size, compute_iters, tile_dim);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernelTime, start, stop);

        total_time += kernelTime / 1000.0;
        if (i == -1){
            if (secs > 0){
                double estimated_time = total_time / 10.0;
                cIterations = int((double)secs / estimated_time) + 1;
                printf("Estimated second is %f, adjust iteration to %d.\n", estimated_time, cIterations);
            }
            total_time = 0;
        }
    }

    kernelTime = total_time / cIterations;

    // take measurements for loop inside kernel
    cudaMemcpy(h_oC1, d_oC1, mem_size, cudaMemcpyDeviceToHost);

    printf("teste: %f\n", h_oC1[0]);

    //float peak_bw = 2 * compute_iters * mem_size * 1.0 / kernelTime / (1024.*1024.*1024.); 
    float peak_bw = 2 * compute_iters * mem_size * 1.0 / kernelTime / (1024.*1024.*1024.); 
    printf("Maximum bandwidth is %.3f GB/s.\n", peak_bw);
    printf("Maximum throughput is %.3f GOP/s.\n", peak_bw / item_size);

    free(h_iA);
    free(h_oC1);

    cudaFree(d_iA);
    cudaFree(d_oC1);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    printf("Test passed\n");

    exit(EXIT_SUCCESS);
}
