/**
 * cache_kernels.cu: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <lcutil.h>
#include "sys/time.h"
#define TOTAL_ITERATIONS  (16384000)
#define UNROLL_ITERATIONS (64)

#define UNROLL_ITERATIONS_MEM (UNROLL_ITERATIONS/2)

const int BLOCK_SIZE = 256;

texture< int, 1, cudaReadModeElementType> texdataI1;

template<class T>
class dev_fun{
public:
	// Pointer displacement operation
	__device__ unsigned int operator()(T v1, unsigned int v2);
	// Compute operation (#1)
	__device__ T operator()(const T &v1, const T &v2);
	// Compute operation (#2)
	__device__ T comp_per_element(const T &v1, const T &v2);
	// Value initialization
	__device__ T init(int v);
	// Element loading
	__device__ T load(volatile const T* p, unsigned int offset);
	// Element storing
	__device__ void store(volatile T* p, unsigned int offset, const T &value);
	// Get first element
	__device__ int first_element(const T &v);
	// Reduce elements (XOR operation)
	__device__ int reduce(const T &v);
};


template<>
__device__ unsigned int dev_fun<int>::operator()(int v1, unsigned int v2){
	return v2+(unsigned int)v1 ;
}
template<>
__device__ int dev_fun<int>::operator()(const int &v1, const int &v2){
  return v1 + v2;
}
template<>
__device__ int dev_fun<int>::comp_per_element(const int &v1, const int &v2){
  return v1 - v2;
}
template<>
__device__ int dev_fun<int>::init(int v){
	return v;
}
template<>
__device__ int dev_fun<int>::load(volatile const int* p, unsigned int offset){
	int retval;
	retval = tex1Dfetch(texdataI1, offset);
	return retval;
}
template<>
__device__ int dev_fun<int>::first_element(const int &v){
	return v;
}
template<>
__device__ int dev_fun<int>::reduce(const int &v){
	return v;
}



template <class T, bool readonly, int blockdim, int stepwidth, int index_clamping>
__global__ void benchmark_func(T * const g_data){
	dev_fun<T> func;
	const int grid_data_width = stepwidth*gridDim.x*blockdim;

	// Thread block-wise striding
	int index = stepwidth*blockIdx.x*blockdim + threadIdx.x;
	index = index_clamping==0 ? index : index % index_clamping;
	const int stride = blockdim;

	unsigned int offset = index;
	T temp = func.init(0);
	for(int j=0; j<TOTAL_ITERATIONS; j+=UNROLL_ITERATIONS){
		// Pretend updating of offset in order to force repetitive loads
		offset = func(temp, offset);
		/*volatile*/ T * const g_data_store_ptr = g_data+offset+grid_data_width;
#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			const unsigned int iteration_offset = (readonly ? i : i >> 1) % stepwidth;//readonly ? i % stepwidth : (i >> 1) % stepwidth;
			if( readonly || (i % 2 == 0) ){
				const T v = func.load(g_data, offset+iteration_offset*stride);
				if( readonly ){
					// Pretend update of offset in order to force reloads
					offset ^= func.reduce(v);
				}
				temp = v;
			} else
				func.store( g_data_store_ptr, iteration_offset*stride, temp );
		}
	}
	offset = func(temp, offset);
	if( offset != index ) // Does not occur
		*g_data = func.init(offset);
}

double max3(double v1, double v2, double v3){
	double t = v1>v2 ? v1 : v2;
	return t>v3 ? t : v3;
}

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

template<class datatype>
void runbench_warmup(datatype *cd, long size){
	const long reduced_grid_size = size/(UNROLL_ITERATIONS_MEM)/32;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	benchmark_func< datatype, true, BLOCK_SIZE, 1, 256 ><<< dimReducedGrid, dimBlock >>>(cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

template<class datatype, bool readonly, int stepwidth, int index_clamping>
double runbench(int total_blocks, datatype *cd, long size, bool spreadsheet){
	const long compute_grid_size = total_blocks*BLOCK_SIZE;
	const long data_size = ((index_clamping==0) ? compute_grid_size : min((int)compute_grid_size, (int)index_clamping))*stepwidth;//*(2-readonly);

	const long long total_iterations = (long long)(TOTAL_ITERATIONS)*compute_grid_size;
	const long long computations = total_iterations*(sizeof(datatype)/sizeof(int));//(long long)(TOTAL_ITERATIONS)*compute_grid_size;
	const long long memoryoperations = total_iterations;//(long long)(TOTAL_ITERATIONS)*compute_grid_size;

	// Set device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(datatype)) );  // initialize to zeros

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(total_blocks, 1, 1);

	cudaEvent_t start, stop;

	if(!spreadsheet){
		printf("\nElement size %d, Grid size: %ld threads, Data size: %ld elements (%ld bytes)\n", (int)sizeof(datatype), compute_grid_size, data_size, data_size*sizeof(datatype)/*size*//*, 32*(UNROLL_ITERATIONS-1)*sizeof(datatype)*/);
	}

	initializeEvents(&start, &stop);
	benchmark_func< datatype, readonly, BLOCK_SIZE, stepwidth, index_clamping ><<< dimGrid, dimBlock >>>(cd);
	float kernel_time = finalizeEvents(start, stop);
	double bandwidth = ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.);

	if(!spreadsheet){
		printf("\tKernel execution time :    %10.3f msecs\n", kernel_time);
		printf("\t   Compute throughput :    %10.3f GIOPS\n", ((double)computations)/kernel_time*1000./(double)(1000*1000*1000));
		printf("\tMemory bandwidth\n");
		printf("\t               Total  :    %10.2f GB/sec\n", ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.));
		printf("\t               Loads  :    %10.2f GB/sec\n", ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.) * (readonly ? 1.0 : 0.5));
		printf("\t               Stores :    %10.2f GB/sec\n", ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.) * (readonly ? 0.0 : 0.5));
	} else {
		int current_device;
		cudaDeviceProp deviceProp;
		CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
		printf("%12d;%9ld;%6d;%8d;%10ld;%8ld;%14.3f;%13.3f;%10.3f;%8.3f;%9.3f\n",
			(int)sizeof(datatype), compute_grid_size, stepwidth, index_clamping, data_size, data_size*sizeof(datatype),
			kernel_time, 
			((double)computations)/kernel_time*1000./(double)(1000*1000*1000),
			bandwidth,
			((double)memoryoperations)/kernel_time*1000./(1000.*1000.*1000.),
			((double)memoryoperations)/kernel_time*1000./(1000.*1000.*1000.) / (deviceProp.multiProcessorCount*deviceProp.clockRate/1000000.0));
	}
	return bandwidth;
}

template<class datatype, bool readonly>
double cachebenchGPU(double *c, long size, bool excel){
	// Construct grid size
	cudaDeviceProp deviceProp;
	int current_device;
	CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	const int SM_count = deviceProp.multiProcessorCount;
	const int Threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
	const int BLOCKS_PER_SM = Threads_per_SM/BLOCK_SIZE;
	const int TOTAL_BLOCKS = BLOCKS_PER_SM * SM_count;

	datatype *cd;

	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(datatype)) );

	// Set device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(datatype)) );  // initialize to zeros

	// Bind textures to buffer
	cudaBindTexture(0, texdataI1, cd, size*sizeof(datatype));

	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	runbench_warmup(cd, size);

        int secs = -1;
        int cIterations = 10;
        struct timeval start, end;

        // Get environment variables
        if (getenv("secs") != NULL)
            secs = atoi(getenv("secs"));

        double total_time = 0;
        double peak_bw = 0;
        for(int i = -3; i < cIterations; i++){
            gettimeofday(&start, NULL);
	    peak_bw += max( peak_bw, runbench<datatype, readonly,  1,    0>(TOTAL_BLOCKS, cd, size, excel) );
            gettimeofday(&end, NULL);
            total_time += end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec)/1000000.0;

            if (i == -1){
                if (secs > 0){
                    double estimated_time = total_time / 3.0;
                    cIterations = int((double)secs / estimated_time) + 1;
                    printf("Estimated second is %f, adjust iteration to %d.\n", estimated_time, cIterations);
                }
                peak_bw = 0;
                total_time = 0;
            }
        }

        peak_bw = peak_bw * 1.0 / cIterations;
	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(datatype), cudaMemcpyDeviceToHost) );

	// Unbind textures
	cudaUnbindTexture(texdataI1);

	CUDA_SAFE_CALL( cudaFree(cd) );
	return peak_bw;
}

extern "C" void cachebenchGPU(double *c, long size, bool excel){

	printf("Texture cache benchmark\n");

	printf("\nRead only benchmark\n");
	if( excel ){
		printf("EXCEL header:\n");
		printf("Element size;Grid size; Parameters;   ; Data size;        ;Execution time;Instr.thr/put;Memory b/w; Ops/sec;Ops/cycle\n");
		printf("     (bytes);(threads);(step);(idx/cl);(elements); (bytes);       (msecs);      (GIOPS);  (GB/sec);  (10^9);   per SM\n");
	}
	double peak_bw_ro_int1 = cachebenchGPU<int,  true>(c, size, excel);

	printf("\tRead only accesses:\n");
	printf("\t\tint1: %10.2f GB/sec\n", peak_bw_ro_int1);

        printf("Maximum bandwidth is %.3f GB/s.\n", peak_bw_ro_int1);
        printf("Maximum throughput is %.3f GOP/s.\n", peak_bw_ro_int1 / 4.0);
	CUDA_SAFE_CALL( cudaDeviceReset() );
}
