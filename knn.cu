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

#define INIT_MAX 1000000
void showResult(int m, int k, int *out);

extern __shared__ int SM[];

// compute the square of distance per dimension
// the kth dimension of the ith point and jth point
__device__ void computeDimDist(int m, int n, int *V)
{
	int i = blockIdx.x;
   	int j = blockIdx.y;
	int k = threadIdx.x;
	SM[k] = (V[i*n+k]-V[j*n+k])*(V[i*n+k]-V[j*n+k]);
}

// compute the square of distance of the ith point and jth point
__global__ void computeDist(int m, int n, int *V, int *D)
{
	int i = blockIdx.x;
   	int j = blockIdx.y;
	int k = threadIdx.x;
	int s;
	// calculate the square of distance per dimensions
	computeDimDist(m, n, V);
	__syncthreads();
	// reduce duplicated calculations since d(i, j) = d(j, i)
	// also, we do not consider the trivial case of d(i, i) = 0
	// so we only compute the square distance when i < j 
	if(i < j)
	{
		// use paralel reduction
		for(s=n/2; s>0; s>>=1)
		{
			if(k < s)
			{
				SM[k] += SM[k+s];
			}
			__syncthreads();
		}
	}
	if(k == 0)
	{
		// when n is odd, the last element of SM needs to be added
		if(n > (n/2)*2)
		{
			D[i*m+j] = SM[0] + SM[n-1];
			D[j*m+i] = SM[0] + SM[n-1];
		}
		else
		{
			D[i*m+j] = SM[0];
			D[j*m+i] = SM[0];
		}
	}
}

__device__ void initSM(int m, int *D)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	if(i < j)
	{
		SM[j] = D[i*m+j];
	}
	else
	{
		SM[j] = D[j*m+i];	
	}
}

__device__ int findMin(int m, int n, int k, int count, int *D, int *out)
{
	__shared__ int R[1000];
	int i = blockIdx.x;
  	int j = threadIdx.x;
	int s = blockDim.x/2;
	int num;
	initSM(m, D);
	__syncthreads();
	R[j] = j;
	__syncthreads();
	SM[i] = INIT_MAX;
	__syncthreads();
	for(num=0; num<count; num++)
	{
		SM[ out[i*k+num] ] = INIT_MAX;
	}
	__syncthreads();
	for(s=m/2; s>0; s>>=1) 
	{
		// check whether the jth point is the same point as the ith one
		// or has already in the k-nn list
		if(j < s) 
		{
			SM[j] = SM[j]<SM[j+s]? SM[j]: SM[j+s];
			R[j] = SM[j]<SM[j+s]? R[j]: R[j+s];
			/*
			if(SM[j] == SM[j+s])
			{
				if(R[j] > R[j+s])
				{
					R[j] = R[j+s];
				}
			}
			else if(SM[j] > SM[j+s])
			{
				SM[j] = SM[j+s];
				R[j] = R[j+s];
			}
			*/
			__syncthreads();
		}
	}
	// when m is odd, the last element of SM needs to be compared
	if(m > (m/2)*2)
	{
		if(SM[0] > SM[m-1])
		{
			R[0] = R[m-1];
		}
	}
	__syncthreads();
	return R[0];
}

// compute the k nearest neighbors
__global__ void knn(int m, int n, int k, int *V, int *D, int *out)
{
	int i;
	int count;

	i = blockIdx.x;
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
			printf("%d", out[i*k+j]);
			if(j == k-1)
			{
				printf("\n");
			}
			else
			{
				printf(" ");
			}
		}
	} 
} 
void showD(int m, int *D)
{
	int i,j;
	printf("D:\n");
	for(i=0; i<m; i++)
	{
		for(j=0; j<m; j++)
		{
			printf("%d", D[i*m+j]);
			if(j == m-1)
			{
				printf("\n");
			}
			else
			{
				printf(" ");
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

	//test
	int *h_D;						
//	int *R;						
//	int *dim_D;					//be replaced with shared memory
	FILE *fp;
	if(argc != 2)
	{
		printf("Usage: knn [file path]\n");
		exit(1);
	}
	if((fp = fopen(argv[1], "r")) == NULL)
	{
		printf("Error open file!\n");
		exit(1);
	}
	while(fscanf(fp, "%d %d %d", &m, &n, &k) != EOF)
	{
		V = (int *) malloc(m*n*sizeof(int));
		out = (int *) malloc(m*k*sizeof(int));

		h_D = (int *)malloc(m*m*sizeof(int));
		// allocate space for devices copies
		cudaMalloc((void **)&d_V, m*n*sizeof(int));
		cudaMalloc((void **)&d_out, m*k*sizeof(int));
		cudaMalloc((void **)&D, m*m*sizeof(int));
//		cudaMalloc((void **)&R, m*m*sizeof(int));
//		cudaMalloc((void **)&dim_D, m*m*n*sizeof(int));

		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}
		// copy host values to devices copies
		cudaMemcpy(d_V, V, m*n*sizeof(int), cudaMemcpyHostToDevice);

		dim3 blk(m, n);
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
		//test
		cudaMemcpy(h_D, D, m*m*sizeof(int), cudaMemcpyDeviceToHost);
		showD(m, h_D);
		knn<<<m, m, m*sizeof(int)>>>(m, n, k, d_V, D, d_out);
		// record event and synchronize
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float time;
		// get event elapsed time
		cudaEventElapsedTime(&time, start, stop);
		printf("GPU calculation time:%f ms\n", time);
		// copy result back to host
		cudaMemcpy(out, d_out, m*k*sizeof(int), cudaMemcpyDeviceToHost);

		showResult(m, k, out);
		// cleanup
		cudaFree(d_V);
		cudaFree(d_out);
		cudaFree(D);
//		cudaFree(R);
//		cudaFree(dim_D);

		free(V);
		free(out);
	}
	fclose(fp);
	return 0;
}

