// --- Link against nvrtc.lib
// --- Link against cuda.lib

#include <iostream>

#include <nvrtc.h>

/*********************/
/* NVRTC ERROR CHECK */
/*********************/
#define NVRTC_ERROR_CHECK(x) \
do { \
	nvrtcResult result = x; \
	if (result != NVRTC_SUCCESS) { \
	std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
	exit(1); \
 } \
} while(0)

/***************************/
/* COMPILE TO PTX FUNCTION */
/***************************/
char* compile2PTX(const char* code) {

	// --- Create an instance of nvrtcProgram.
	nvrtcProgram prog;
	NVRTC_ERROR_CHECK(nvrtcCreateProgram(&prog, code, NULL, 0, NULL, NULL));

	// --- Compile the program for compute_75 with fmad disabled.
	const char* opts[] = { "--gpu-architecture=compute_75",	"--fmad=false" };
	NVRTC_ERROR_CHECK(nvrtcCompileProgram(prog, 2, opts));

	// --- Compilation log.
	size_t logSize;
	NVRTC_ERROR_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
	char* log = new char[logSize];
	NVRTC_ERROR_CHECK(nvrtcGetProgramLog(prog, log));
	std::cout << log << '\n';
	delete[] log;

	// --- PTX.
	size_t ptxSize;
	NVRTC_ERROR_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
	char* ptx = new char[ptxSize];
	NVRTC_ERROR_CHECK(nvrtcGetPTX(prog, ptx));
	std::cout << ptx << std::endl;

	// --- Program destruction.
	NVRTC_ERROR_CHECK(nvrtcDestroyProgram(&prog));

	return ptx;
}
