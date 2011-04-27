
// common SDK header for standard utilities and system libs 
#include <oclUtils.h>

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "k_means_kernel.cc";

// Host buffers for demo
// *********************************************************************
void *srcA, *srcB, *dst;        // Host buffers for OpenCL test
void* Golden;                   // Host buffer for host golden processing cross check

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevSrcB;               // OpenCL device source buffer B 
cl_mem cmDevDst;                // OpenCL device destination buffer 
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;		    // 1D var for # of work items in the work group	
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation 

//////////////////////////////////////////////////////////////////////////
#include <time.h>
#include <stdlib.h>
cl_mem cmDevSrc_scalar_value;               // OpenCL device source buffer A
cl_mem cmDevSrc_gradient_magnitude;               // OpenCL device source buffer B 
cl_mem cmDevSrc_second_derivative_magnitude;               // OpenCL device source buffer B 
cl_mem cmDevDst_label_ptr;                // OpenCL device destination buffer 
//////////////////////////////////////////////////////////////////////////

// demo config vars
int iNumElements = 64;	// Length of float arrays to process (odd # for illustration)
shrBOOL bNoPrompt = shrFALSE;  

// Forward Declarations
// *********************************************************************
void VectorAddHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);
void Cleanup (int iExitCode);

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
	//////////////////////////////////////////////////////////////////////////
	unsigned int count = iNumElements;
	int k = 8;
	unsigned int random_seed, random_seed2;
	srand( (unsigned)time( NULL ) );
	random_seed = rand();
	random_seed2 = rand();
	//////////////////////////////////////////////////////////////////////////

	// get command line arg for quick test, if provided
	bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");

	// start logs 
	shrSetLogFileName ("oclVectorAdd.txt");
	shrLog("%s Starting...\n\n# of float elements per Array \t= %i\n", argv[0], iNumElements); 

	// set and log Global and Local work size dimensions
	szLocalWorkSize = 256;
	szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  // rounded up to the nearest multiple of the LocalWorkSize
	shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
		szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

	// Allocate and initialize host arrays 
	shrLog( "Allocate and Init Host Mem...\n"); 
	srcA = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
	srcB = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
	dst = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
	Golden = (void *)malloc(sizeof(cl_float) * iNumElements);
	shrFillArray((float*)srcA, iNumElements);
	shrFillArray((float*)srcB, iNumElements);
	//////////////////////////////////////////////////////////////////////////
	float *scalar_value = new float[count];
	float *gradient_magnitude = new float[count];
	float *second_derivative_magnitude = new float[count];
	unsigned char *label_ptr = new unsigned char[count];
	shrFillArray(scalar_value, count);
	shrFillArray(gradient_magnitude, count);
	shrFillArray(second_derivative_magnitude, count);
	//////////////////////////////////////////////////////////////////////////

	//Get an OpenCL platform
	ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

	shrLog("clGetPlatformID...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	//Get the devices
	ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	shrLog("clGetDeviceIDs...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	//Create the context
	cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
	shrLog("clCreateContext...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// Create a command-queue
	cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
	shrLog("clCreateCommandQueue...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// Allocate the OpenCL buffer memory objects for source and result on the device GMEM
	cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr1);
	cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	//////////////////////////////////////////////////////////////////////////
	cmDevSrc_scalar_value = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr1);
	cmDevSrc_gradient_magnitude = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevSrc_second_derivative_magnitude = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevDst_label_ptr = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	//////////////////////////////////////////////////////////////////////////
	shrLog("clCreateBuffer...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// Read the OpenCL kernel in from source file
	shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
	cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
	cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
	printf("%s\n%s\n", cSourceFile, cPathAndName);

	// Create the program
	cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
	shrLog("clCreateProgramWithSource...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// Build the program with 'mad' Optimization option
#ifdef MAC
	char* flags = "-cl-fast-relaxed-math -DMAC";
#else
	char* flags = "-cl-fast-relaxed-math";
#endif
	ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	shrLog("clBuildProgram...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// Create the kernel
	ckKernel = clCreateKernel(cpProgram, "k_means", &ciErr1);
	shrLog("clCreateKernel (VectorAdd)...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// Set the Argument values
	//ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
	//ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB);
	//ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevDst);
	//ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&iNumElements);
	//////////////////////////////////////////////////////////////////////////
	// __global const float *scalar_value, __global const float *gradient_magnitude, __global const float *second_derivative_magnitude, __global unsigned char *label_ptr, __global const unsigned int count, __global const int k, __global const unsigned int random_seed, __global const unsigned int random_seed2
	ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrc_scalar_value);
	ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrc_gradient_magnitude);
	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevSrc_second_derivative_magnitude);
	ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmDevDst_label_ptr);
	ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_uint), (void*)&count);
	ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_uint), (void*)&k);
	ciErr1 |= clSetKernelArg(ckKernel, 6, sizeof(cl_uint), (void*)&random_seed);
	ciErr1 |= clSetKernelArg(ckKernel, 7, sizeof(cl_uint), (void*)&random_seed2);
	//////////////////////////////////////////////////////////////////////////
	shrLog("clSetKernelArg 0 - 3...\n\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// --------------------------------------------------------
	// Start Core sequence... copy input data to GPU, compute, copy results back

	// Asynchronous write of data to GPU device
	//ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, srcA, 0, NULL, NULL);
	//ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, srcB, 0, NULL, NULL);
	//////////////////////////////////////////////////////////////////////////
	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrc_scalar_value, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, scalar_value, 0, NULL, NULL);
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrc_gradient_magnitude, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, gradient_magnitude, 0, NULL, NULL);
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrc_second_derivative_magnitude, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, second_derivative_magnitude, 0, NULL, NULL);
	//////////////////////////////////////////////////////////////////////////
	shrLog("clEnqueueWriteBuffer (SrcA and SrcB)...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// Launch kernel
	ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
	shrLog("clEnqueueNDRangeKernel (VectorAdd)...\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}

	// Synchronous/blocking read of results, and check accumulated errors
	//ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * szGlobalWorkSize, dst, 0, NULL, NULL);
	//////////////////////////////////////////////////////////////////////////
		ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevDst_label_ptr, CL_TRUE, 0, sizeof(cl_float) * szGlobalWorkSize, label_ptr, 0, NULL, NULL);
	//////////////////////////////////////////////////////////////////////////
	shrLog("clEnqueueReadBuffer (Dst)...\n\n"); 
	if (ciErr1 != CL_SUCCESS)
	{
		shrLog("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup(EXIT_FAILURE);
	}
	//--------------------------------------------------------

	// Compute and compare results for golden-host and report errors and pass/fail
	shrLog("Comparing against Host/C++ computation...\n\n"); 
	VectorAddHost ((const float*)srcA, (const float*)srcB, (float*)Golden, iNumElements);
	shrBOOL bMatch = shrComparefet((const float*)Golden, (const float*)dst, (unsigned int)iNumElements, 0.0f, 0);
	shrLog("%s\n\n", (bMatch == shrTRUE) ? "PASSED" : "FAILED");

	//////////////////////////////////////////////////////////////////////////
	//float *a = (float *)srcA;
	//float *b = (float *)srcB;
	//float *c = (float *)dst;
	//float *d = (float *)Golden;
	//for (int i=0; i<iNumElements; i++)
	//{
	//	printf("%f+%f=%f=%f\t", a[i], b[i], c[i], a[i]+b[i]);
	//	printf("%s\n", (a[i]+b[i]==c[i]?"equal":"not equal"));
	//}

	//for (int i=0; i<iNumElements; i++)
	//{
	//	printf("%f\n", ((float *)dst)[i]);
	//}
	//////////////////////////////////////////////////////////////////////////

	// Cleanup and leave
	Cleanup (EXIT_SUCCESS);

	//////////////////////////////////////////////////////////////////////////
	delete [] scalar_value;
	delete [] gradient_magnitude;
	delete [] second_derivative_magnitude;
	delete [] label_ptr;
	//////////////////////////////////////////////////////////////////////////
}

void Cleanup (int iExitCode)
{
	// Cleanup allocated objects
	shrLog("Starting Cleanup...\n\n");
	if(cPathAndName)free(cPathAndName);
	if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
	if(cpProgram)clReleaseProgram(cpProgram);
	if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
	if(cxGPUContext)clReleaseContext(cxGPUContext);
	if(cmDevSrcA)clReleaseMemObject(cmDevSrcA);
	if(cmDevSrcB)clReleaseMemObject(cmDevSrcB);
	if(cmDevDst)clReleaseMemObject(cmDevDst);

	//////////////////////////////////////////////////////////////////////////
	if(cmDevSrc_scalar_value)clReleaseMemObject(cmDevSrc_scalar_value);
	if(cmDevSrc_gradient_magnitude)clReleaseMemObject(cmDevSrc_gradient_magnitude);
	if(cmDevSrc_second_derivative_magnitude)clReleaseMemObject(cmDevSrc_second_derivative_magnitude);
	if(cmDevDst_label_ptr)clReleaseMemObject(cmDevDst_label_ptr);
	//////////////////////////////////////////////////////////////////////////

	// Free host memory
	free(srcA); 
	free(srcB);
	free (dst);
	free(Golden);

	// finalize logs and leave
	if (bNoPrompt)
	{
		shrLogEx(LOGBOTH | CLOSELOG, 0, "oclVectorAdd.exe Exiting...\n");
	}
	else 
	{
		shrLogEx(LOGBOTH | CLOSELOG, 0, "oclVectorAdd.exe Exiting...\nPress <Enter> to Quit\n");
		getchar();
	}
	exit (iExitCode);
}

// "Golden" Host processing vector addition function for comparison purposes
// *********************************************************************
void VectorAddHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements)
{
	int i;
	for (i = 0; i < iNumElements; i++) 
	{
		pfResult[i] = pfData1[i] + pfData2[i]; 
	}
}
