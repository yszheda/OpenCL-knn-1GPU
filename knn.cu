/* 
* INPUT:
* m: total num of points
* m is in [10, 1000]
* n: n dimensions
* n is in [1,1000]
* k: num of nearest points
* k is in [1,10]
* V: point coordinates
* the integer elements are in [-5,5]
* OUTPUT:
* out: k nearest neighbors
*/

#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>

#define INIT_MAX 100000
void showResult(int m, int k, int *out);

extern __shared__ int SMem[];

// compute the square of distance per dimension
// the kth dimension of the ith point and jth point
__device__ void computeDimDist(int i, int j, int n, int *V)
{
	int k = threadIdx.x;
	SMem[k] = (V[i*n+k]-V[j*n+k])*(V[i*n+k]-V[j*n+k]);
}

// compute the square of distance of the ith point and jth point
__global__ void computeDist(int m, int n, int *V, int *D)
{
	int i = blockIdx.x;
   	int j = blockIdx.y;
	int k = threadIdx.x;
	int s;
	int currentScale;
	// calculate the square of distance per dimensions
	// reduce duplicated calculations since d(i, j) = d(j, i)
	// also, we do not consider the trivial case of d(i, i) = 0
	// so we only compute the square distance when i < j 
	if(i < j)
	{
		// use paralel reduction
//		for(s=blockDim.x/2, currentScale=blockDim.x; s>0; s>>=1, currentScale>>=1)
		for(s=blockDim.x/2, currentScale=blockDim.x; s>0; s>>=1)
		computeDimDist(i, j, n, V);
		__syncthreads();
		// use parallel reduction
		for(s=n/2; s>0; s>>=1)
		{
			if(k < s)
			{
				SMem[k] += SMem[k+s];
			}
			__syncthreads();
			// when s is odd, the last element of SMem needs to be added
			if( currentScale>s*2 )
			{
				SMem[0] += SMem[currentScale-1];
			}
			currentScale>>=1;
			__syncthreads();
		}
	}
	D[i*m+j] = SMem[0];
}

__device__ void initSMem(int m, int *D)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	if(i < j)
	{
		SMem[j] = D[i*m+j];
	}
	else
	{
		SMem[j] = D[j*m+i];	
	}
}

__device__ int findMin(int m, int n, int k, int count, int *D, int *out)
{
	__shared__ int R[1000];
	int i = blockIdx.x;
  	int j = threadIdx.x;
	int s = blockDim.x/2;
	int currentScale;
//	int last_s = s;
	int num;
	initSMem(m, D);
	__syncthreads();
	R[j] = j;
	__syncthreads();
	if(j == i)
	{
		SMem[i] = INIT_MAX;
		for(num=0; num<count; num++)
		{
			SMem[ out[i*k+num] ] = INIT_MAX;
		}
	}
	/*
	for(num=0; num<count; num++)
	{
		SMem[ out[i*k+num] ] = INIT_MAX;
	}
	__syncthreads();
	if(j < count)
	{
		SMem[ out[i*k+j] ] = INIT_MAX;
	}
	*/
	__syncthreads();
//	for(s=blockDim.x/2; s>0; s>>=1, last_s=s) 
//	for(s=blockDim.x/2, currentScale=blockDim.x; s>0; s>>=1, currentScale>>=1) 
	for(s=blockDim.x/2, currentScale=blockDim.x; s>0; s>>=1) 
	{
		if(j < s) 
		{
			if(SMem[j] == SMem[j+s])
			{
				if(R[j] > R[j+s])
				{
					R[j] = R[j+s];
				}
			}
			else if(SMem[j] > SMem[j+s])
			{
				SMem[j] = SMem[j+s];
				R[j] = R[j+s];
			}
		}
		__syncthreads();
		if( currentScale>s*2 )
		{
			if(SMem[0] == SMem[currentScale-1])
			{
				if(R[0] > R[currentScale-1])
				{
					R[0] = R[currentScale-1];
				}
			}
			else if(SMem[0] > SMem[currentScale-1])
			{
				SMem[0] = SMem[currentScale-1];
				R[0] = R[currentScale-1];
			}
		}
		currentScale>>=1;
		__syncthreads();
	}
	return R[0];
}

// compute the k nearest neighbors
__global__ void knn(int m, int n, int k, int *V, int *D, int *out)
{
	int i;
	int count;

	i = blockIdx.x;
	__syncthreads();
	for(count=0; count<k; count++)
	{
		out[i*k+count] = findMin(m, n, k, count, D, out);
		__syncthreads();
	}
}

void showResult(int m, int k, int *out)
{
	int i,j;
	for(i=0; i<m; i++)
	{
		for(j=0; j<k; j++)
		{
			printf("%d ", out[i*k+j]);
			if(j == k-1)
			{
				printf("\n");
			}
		}
	} 
} 
int main(int argc, char *argv[]) 
{ 
	int m,n,k;
	int i;
	int *V, *out;				//host copies
	int *d_V, *d_out;			//device copies
	int *D;						
	FILE *fp;
//	FILE *fp_in;
//	FILE *fp_out;
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
	/*
	if((fp_in = fopen(argv[1], "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}
	if((fp_out = fopen("time.txt", "w")) == NULL)
	{
		printf("Error open output file!\n");
		exit(1);
	}
	*/
	while(fscanf(fp, "%d %d %d", &m, &n, &k) != EOF)
	{
		V = (int *) malloc(m*n*sizeof(int));
		out = (int *) malloc(m*k*sizeof(int));

		// allocate space for devices copies
		cudaMalloc((void **)&d_V, m*n*sizeof(int));
		cudaMalloc((void **)&d_out, m*k*sizeof(int));
		cudaMalloc((void **)&D, m*m*sizeof(int));

		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}
		// copy host values to devices copies
		cudaMemcpy(d_V, V, m*n*sizeof(int), cudaMemcpyHostToDevice);

		dim3 grid(m, m);
		// compute the execution time
		cudaEvent_t start, stop;
		// create event
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// record event
		cudaEventRecord(start);
		// launch knn() kernel on GPU
		computeDist<<<grid, n, n*sizeof(int)>>>(m, n, d_V, D);
		cudaDeviceSynchronize();
		knn<<<m, m, m*sizeof(int)>>>(m, n, k, d_V, D, d_out);
		// record event and synchronize
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float time;
		// get event elapsed time
		cudaEventElapsedTime(&time, start, stop);
//		fprintf(fp_out, "GPU calculation time:%f ms\n", time);
		// copy result back to host
		cudaMemcpy(out, d_out, m*k*sizeof(int), cudaMemcpyDeviceToHost);

		showResult(m, k, out);
		printf("%f\n", time);
		// cleanup
		cudaFree(d_V);
		cudaFree(d_out);
		cudaFree(D);

		free(V);
		free(out);
	}
//	fclose(fp_in);
//	fclose(fp_out);
	fclose(fp);
	return 0;
}

