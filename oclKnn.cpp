/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* 
* INPUT:
* m: total num of points
* n: n dimensions
* k: num of nearest points
* V: point coordinates
* OUTPUT:
* out: k nearest neighbors
*/

// common SDK header for standard utilities and system libs 
#include <oclUtils.h>
#include <shrQATest.h>
#include <time.h>
#include "knn.h"

//#define DEBUG

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "knn.cl";

// *********************************************************************

int m,n,k;
cl_int *V, *out;				//host copies

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program

cl_kernel computeDistKernel;             // OpenCL kernel
cl_kernel knnKernel;             // OpenCL kernel

cl_mem d_V;						//device copies
cl_mem d_out;
cl_mem D;						
cl_mem SMem;

size_t GlobalWorkSize[2];
size_t LocalWorkSize[2];

size_t ComputeDistGlobalWorkSize[2];
size_t ComputeDistLocalWorkSize[2];

size_t KnnGlobalWorkSize;
size_t KnnLocalWorkSize;

size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code

cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation
const char* cExecutableName = NULL;

struct timespec start, end;
double comtime;

//shrBOOL bNoPrompt = shrFALSE;  

// Forward Declarations
// *********************************************************************
void computeDist(int m, int n, int *V, int *D);
void knn(int m, int k, int *V, int *D, int *out, int *SMem);
void Cleanup (int argc, char **argv, int iExitCode);

void showResult(int m, int k, int *out)
{
	int i,j;
	for(i=0; i<m; i++)
	{
		for(j=0; j<k; j++)
		{
//			printf("%d ", out[i*k+j]);
			shrLog("%d ", out[i*k+j]);
			if(j == k-1)
			{
//				printf("\n");
				shrLog("\n");
			}	
		}    	
	}        	
}            	

/*
double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}
*/

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
    shrQAStart(argc, argv);

    // get command line arg for quick test, if provided
//    bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    
    // start logs 
	cExecutableName = argv[0];
    shrSetLogFileName ("oclKnn.txt");

	FILE *fp;
	if(argc != 2)
	{
		printf("Usage: knn <inputfile>\n");
		exit(1);
	}
	if((fp = fopen(argv[1], "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}
	fscanf(fp, "%d %d %d", &m, &n, &k);
    // Allocate and initialize host arrays 
#ifdef DEBUG
    shrLog( "Allocate and Init Host Mem...\n"); 
#endif
	V = (cl_int *) malloc(m * n * sizeof(cl_int));
	out = (cl_int *) malloc(m * k * sizeof(cl_int));
	for(int i=0; i<m*n; i++)
	{
		fscanf(fp, "%d", &V[i]);
	}

	clock_gettime(CLOCK_REALTIME,&start);

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

#ifdef DEBUG
    shrLog("clGetPlatformID...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
#ifdef DEBUG
    shrLog("clGetDeviceIDs...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
#ifdef DEBUG
    shrLog("clCreateContext...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
#ifdef DEBUG
    shrLog("clCreateCommandQueue...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
	d_V = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_int) * m * n, NULL, &ciErr1);
	D = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * m * m, NULL, &ciErr2);
    ciErr1 |= ciErr2;
	d_out = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * m * k, NULL, &ciErr2);
    ciErr1 |= ciErr2;

#ifdef DEBUG
    shrLog("clCreateBuffer...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    
    // Read the OpenCL kernel in from source file
#ifdef DEBUG
    shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
#endif
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);

    // Create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
#ifdef DEBUG
    shrLog("clCreateProgramWithSource...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
    #endif
    ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
//    ciErr1 = clBuildProgram(cpProgram, 0, NULL, "-g -G", NULL, NULL);
//    ciErr1 = clBuildProgram(cpProgram, 1, &cdDevice, "-g -G", NULL, NULL);
#ifdef DEBUG
    shrLog("clBuildProgram...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
		printf("ciErr1:%d\n", ciErr1);
		if (ciErr1 == CL_BUILD_PROGRAM_FAILURE) { 
		    size_t log_size;
		    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		    char *log = (char *) malloc(log_size);
		    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		    printf("%s\n", log);
		}
#ifdef DEBUG
        shrLog("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // set and log Global and Local work size dimensions

	int gridDimX = (int)(ceil((float)m/TILE_WIDTH));
	int gridDimY = (int)(ceil((float)m/TILE_WIDTH));
	ComputeDistLocalWorkSize[0] = TILE_WIDTH;
	ComputeDistLocalWorkSize[1] = TILE_WIDTH;
	ComputeDistGlobalWorkSize[0] = ComputeDistLocalWorkSize[0] * gridDimX;
	ComputeDistGlobalWorkSize[1] = ComputeDistLocalWorkSize[1] * gridDimY;

#ifdef DEBUG
    shrLog("Global Work Size \t\t= %u %u\nLocal Work Size \t\t= %u %u\n# of Work Groups \t\t= %u %u\n\n", 
           ComputeDistGlobalWorkSize[0], ComputeDistGlobalWorkSize[1], ComputeDistLocalWorkSize[0], ComputeDistLocalWorkSize[1], (ComputeDistGlobalWorkSize[0] % ComputeDistLocalWorkSize[0] + ComputeDistGlobalWorkSize[0]/ComputeDistLocalWorkSize[1]), (ComputeDistGlobalWorkSize[1] % ComputeDistLocalWorkSize[1] + ComputeDistGlobalWorkSize[1]/ComputeDistLocalWorkSize[1])); 
#endif

//    shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
//           GlobalWorkSize, LocalWorkSize, (GlobalWorkSize % LocalWorkSize + GlobalWorkSize/LocalWorkSize)); 

    // Create the kernel
    computeDistKernel = clCreateKernel(cpProgram, "computeDist", &ciErr1);
#ifdef DEBUG
    shrLog("clCreateKernel (computeDist)...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Set the Argument values
    ciErr1 = clSetKernelArg(computeDistKernel, 0, sizeof(cl_int), (void*)&m);
    ciErr1 |= clSetKernelArg(computeDistKernel, 1, sizeof(cl_int), (void*)&n);
    ciErr1 |= clSetKernelArg(computeDistKernel, 2, sizeof(cl_mem), (void*)&d_V);
    ciErr1 |= clSetKernelArg(computeDistKernel, 3, sizeof(cl_mem), (void*)&D);
#ifdef DEBUG
    shrLog("clSetKernelArg 0 - 3...\n\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, d_V, CL_FALSE, 0, sizeof(cl_int) * m * n, V, 0, NULL, NULL);
#ifdef DEBUG
    shrLog("clEnqueueWriteBuffer (d_V)...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Launch kernel
    ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, computeDistKernel, 2, 0, ComputeDistGlobalWorkSize, ComputeDistLocalWorkSize, 0, NULL, NULL);
#ifdef DEBUG
    shrLog("clEnqueueNDRangeKernel (computeDist)...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }


	ciErr1 = clEnqueueBarrier(cqCommandQueue);
#ifdef DEBUG
    shrLog("clEnqueueBarrier...\n\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
			printf("%d\n", ciErr1);
    size_t log_size;
    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("%s\n", log);

#ifdef DEBUG
        shrLog("Error in clEnqueueBarrier, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }


    // set and log Global and Local work size dimensions
	int ptrNumInSMEM = (m<MAX_PTRNUM_IN_SMEM)? m: MAX_PTRNUM_IN_SMEM;
	KnnLocalWorkSize = (m<MAX_BLOCK_SIZE)? m: MAX_BLOCK_SIZE;
	KnnGlobalWorkSize = KnnLocalWorkSize * m;

#ifdef DEBUG
    shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
           KnnGlobalWorkSize, KnnLocalWorkSize, (KnnGlobalWorkSize % KnnLocalWorkSize + KnnGlobalWorkSize / KnnLocalWorkSize)); 
#endif

    // Create the kernel
    knnKernel = clCreateKernel(cpProgram, "knn", &ciErr1);
#ifdef DEBUG
    shrLog("clCreateKernel (knn)...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Set the Argument values
    ciErr1 = clSetKernelArg(knnKernel, 0, sizeof(cl_int), (void*)&m);
    ciErr1 |= clSetKernelArg(knnKernel, 1, sizeof(cl_int), (void*)&k);
    ciErr1 |= clSetKernelArg(knnKernel, 2, sizeof(cl_mem), (void*)&d_V);
    ciErr1 |= clSetKernelArg(knnKernel, 3, sizeof(cl_mem), (void*)&D);
    ciErr1 |= clSetKernelArg(knnKernel, 4, sizeof(cl_mem), (void*)&d_out);
//    ciErr1 |= clSetKernelArg(knnKernel, 5, sizeof(cl_int) * 2 * ptrNumInSMEM, (void*)&SMem);
    ciErr1 |= clSetKernelArg(knnKernel, 5, sizeof(int) * 2 * ptrNumInSMEM, 0);
#ifdef DEBUG
    shrLog("clSetKernelArg 0 - 5...\n\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // --------------------------------------------------------

    // Launch kernel
    ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, knnKernel, 1, 0, &KnnGlobalWorkSize, &KnnLocalWorkSize, 0, NULL, NULL);
#ifdef DEBUG
    shrLog("clEnqueueNDRangeKernel (knn)...\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
#ifdef DEBUG
        shrLog("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }


	ciErr1 = clFinish(cqCommandQueue);
#ifdef DEBUG
    shrLog("clFinish...\n\n"); 
#endif
    if (ciErr1 != CL_SUCCESS)
    {
		printf("%d\n", ciErr1);
#ifdef DEBUG
        shrLog("Error in clFinish, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, d_out, CL_TRUE, 0, sizeof(cl_int) * m * k, (void*)out, 0, NULL, NULL);
#ifdef DEBUG
    shrLog("clEnqueueReadBuffer (out)...\n\n"); 
#endif

//	showResult(m, k, out);
//    shrLog("  Kernel execution time on GPU %d \t: %.5f s\n", i, executionTime(GPUExecution));

    if (ciErr1 != CL_SUCCESS)
    {
		printf("%d\n",ciErr1);
		if (ciErr1 == CL_OUT_OF_RESOURCES) { 
		    size_t log_size;
		    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		    char *log = (char *) malloc(log_size);
		    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			printf("CL_OUT_OF_RESOURCES\n");
		    printf("%s\n", log);
		}

#ifdef DEBUG
        shrLog("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
#endif
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //--------------------------------------------------------

    // Compute and compare results for golden-host and report errors and pass/fail
//    shrLog("Comparing against Host/C++ computation...\n\n"); 
//    VectorAddHost ((const float*)srcA, (const float*)srcB, (float*)Golden, iNumElements);
//    shrBOOL bMatch = shrComparefet((const float*)Golden, (const float*)dst, (unsigned int)iNumElements, 0.0f, 0);

    // Cleanup and leave
//    Cleanup (argc, argv, (bMatch == shrTRUE) ? EXIT_SUCCESS : EXIT_FAILURE);
    Cleanup (argc, argv, EXIT_SUCCESS);
}

void Cleanup (int argc, char **argv, int iExitCode)
{
    // Cleanup allocated objects
#ifdef DEBUG
    shrLog("Starting Cleanup...\n\n");
#endif
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(computeDistKernel)clReleaseKernel(computeDistKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(d_V)clReleaseMemObject(d_V);
    if(d_out)clReleaseMemObject(d_out);
    if(D)clReleaseMemObject(D);

	clock_gettime(CLOCK_REALTIME,&end);
//	comtime = (double)(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)/(double)1000000000L;
	comtime = (double)(end.tv_sec-start.tv_sec)*1000 + (double)(end.tv_nsec-start.tv_nsec)/(double)1000000L;
	showResult(m, k, out);
	shrLog("%f\n", comtime);

    // Free host memory
	free(V);
	free(out);

    // finalize logs and leave
#ifdef DEBUG
    shrQAFinishExit(argc, (const char **)argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
#endif
}
