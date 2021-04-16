#include "nvrtcRoutines.h"

#include <cuda.h>

#include <iostream>

// --- Loading an ASCII file and assigning it to a string
#include <string>
#include <fstream>
#include <streambuf>

#define BLOCKSIZE 8

/**********/
/* iDivUp */
/**********/
int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/******************/
/* CUDA SAFE CALL */
/******************/
#define gpuErrchk(x) \
do { \
	CUresult result = x; \
	if (result != CUDA_SUCCESS) { \
		const char *msg; \
		cuGetErrorName(result, &msg); \
		std::cerr << "\nerror: " #x " failed with error " << msg << '\n'; \
		exit(1); } \
 } while(0)

/**************************************/
/* GLOBAL FUNCTIONS DEFINED BY STRING */
/**************************************/
const char* kernels = " \n\
extern \"C\" __global__ \n\
void kernel1(float *d_x, float *d_y, float *d_c, int N) \n\
{ \n\
 const int tid = blockIdx.x * blockDim.x + threadIdx.x; \n\
 if (tid < N) { \n\
 d_c[tid] = d_x[tid] + d_y[tid]; \n\
 } \n\
} \n\
extern \"C\" __global__ void kernel2(float *d_x, float *d_y, float *d_c, int N) \n\
{ \n\
 const int tid = blockIdx.x * blockDim.x + threadIdx.x; \n\
 if (tid < N) { \n\
 d_c[tid] = 2.f * d_x[tid] + 2.f * d_y[tid]; \n\
 } \n\
} \n";

	

/********/
/* MAIN */
/********/
int main() {

	int N = 21;
	
	// --- Loading an ASCII file containing the relevant kernels
	std::ifstream t("example.txt");
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	std::cout << str << std::endl;

	// --- Compiling to ptx string
	//char* ptx = compile2PTX((char *)str.c_str());
	char* ptx = compile2PTX(kernels);
	//std::cout << ptx << std::endl;

	// --- Host array allocation and initialization 
	float* h_x = (float*)malloc(N * sizeof(float));
	float* h_y = (float*)malloc(N * sizeof(float));
	float* h_z = (float*)malloc(N * sizeof(float));
	float* h_w = (float*)malloc(N * sizeof(float));
	for (int i = 0; i < N; i++) {
		h_x[i] = (float)i;
		h_y[i] = (float)(i * 2);
	}

	// --- Load the generated PTX and get handle to the SAXPY kernel.
	gpuErrchk(cuInit(0));
	CUdevice   cuDevice; gpuErrchk(cuDeviceGet(&cuDevice, 0));
	CUcontext  context;  gpuErrchk(cuCtxCreate(&context, 0, cuDevice));
	CUmodule   module;   gpuErrchk(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
	CUfunction kernel1;  gpuErrchk(cuModuleGetFunction(&kernel1, module, "kernel1"));
	CUfunction kernel2;  gpuErrchk(cuModuleGetFunction(&kernel2, module, "kernel2"));

	// --- Device array allocation.
	CUdeviceptr d_x; gpuErrchk(cuMemAlloc(&d_x, N * sizeof(float)));
	CUdeviceptr d_y; gpuErrchk(cuMemAlloc(&d_y, N * sizeof(float)));
	CUdeviceptr d_z; gpuErrchk(cuMemAlloc(&d_z, N * sizeof(float)));
	CUdeviceptr d_w; gpuErrchk(cuMemAlloc(&d_w, N * sizeof(float)));

	// --- Host-device mem copies.
	gpuErrchk(cuMemcpyHtoD(d_x, h_x, N * sizeof(float)));
	gpuErrchk(cuMemcpyHtoD(d_y, h_y, N * sizeof(float)));
	
	// --- Execute kernels.
	void* args1[] = { &d_x, &d_y, &d_z, &N };
	void* args2[] = { &d_x, &d_y, &d_w, &N };
	gpuErrchk(cuLaunchKernel(kernel1,
			iDivUp(N, BLOCKSIZE), 1, 1,	// --- grid dim
			BLOCKSIZE,  1, 1,			// --- block dim
			0, NULL,					// --- shared mem and stream
			args1, 0));					// --- arguments
	gpuErrchk(cuLaunchKernel(kernel2,
		iDivUp(N, BLOCKSIZE), 1, 1,	// --- grid dim
		BLOCKSIZE, 1, 1,			// --- block dim
		0, NULL,					// --- shared mem and stream
		args2, 0));					// --- arguments
	gpuErrchk(cuCtxSynchronize());
	
	// --- Device-host mem copies.
	gpuErrchk(cuMemcpyDtoH(h_z, d_z, N * sizeof(float)));
	gpuErrchk(cuMemcpyDtoH(h_w, d_w, N * sizeof(float)));

	for (int i = 0; i < N; i++) {
		std::cout << i << "\t" << h_x[i] << "\t" << h_y[i] << "\t" << h_z[i] << "\t" << h_w[i] << '\n';
	}
	
	// --- Release resources.
	gpuErrchk(cuMemFree(d_x));
	gpuErrchk(cuMemFree(d_y));
	gpuErrchk(cuMemFree(d_z));
	gpuErrchk(cuMemFree(d_w));
	gpuErrchk(cuModuleUnload(module));
	gpuErrchk(cuCtxDestroy(context));
	free(h_x);
	free(h_y);
	free(h_z);
	free(h_w);

	return 0;
}