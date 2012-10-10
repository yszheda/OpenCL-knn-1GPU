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

#define INIT_MAX 100
void showResult(int m, int k, int *out);

extern __shared__ int block_dim_D[];

// compute the square of distance per dimension
// the kth dimension of the ith point and jth point
__device__ void computeDimDist(int m, int n, int *V)
{
	int i = blockIdx.x;
   	int j = threadIdx.x;
	int k = threadIdx.y;
	block_dim_D[j*n+k] = (V[i*n+k]-V[j*n+k])*(V[i*n+k]-V[j*n+k]);
}

// compute the square of distance of the ith point and jth point
__device__ void computeDist(int m, int n, int *V, int *D)
{
	int i = blockIdx.x;
   	int j = threadIdx.x;
	int k = threadIdx.y;
	int s;
	int base_idx;
	int dist = 0;
	// calculate the square of distance per dimensions
	computeDimDist(m, n, V);
	__syncthreads();
	base_idx = j*n;
	// reduce duplicated calculations since d(i, j) = d(j, i)
	// also, we do not consider the trivial case of d(i, i) = 0
	// so we only compute the square distance when i < j 
	if(i < j)
	{
		// use parallel reduction
		for(s=n/2; s>0; s>>=1)
		{
			if(k < s)
			{
				block_dim_D[base_idx+k] += block_dim_D[base_idx+k+s];
			}
		}
		__syncthreads();
		dist = block_dim_D[base_idx];
		// when n is odd, the last element of block_dim_D needs to be added
		if(n > (n/2)*2)
		{
			dist += block_dim_D[base_idx+n-1];
		}
	}
	D[i*m+j] = dist;
}

// compute the k nearest neighbors
__global__ void knn(int m, int n, int k, int *V, int *out, int *D)
{
	int i,j;
	int temp;
	int count;
	int num;
	int dist;
	int is_duplicate;

	computeDist(m, n, V, D);
	__syncthreads();
	// find the k nearest neighbors of the point with index = blockIdx.x
	i = blockIdx.x;
	// let the first thread select the k-min distance
	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		for(count=0; count<k; count++)
		{
			temp = INIT_MAX;
			// iterate the jth point
			for(j=0; j<m; j++)
			{
				// check whether the jth point is the same point as the ith one
				// or has already in the k-nn list
				is_duplicate = 0;
				if(j == i)
				{
					is_duplicate = 1;
				}
				for(num=0; num<count; num++)
				{
					if(out[i*k+num] == j)
					{
						is_duplicate = 1;
					}
				}
				// we have reduced duplicated calculation of the pair of d(i, j) and d(j, i) before,
			    // and only one of them (here is the one that satisfies i<j) is valid
				// now we need to load the correct one from the array D[]
				if(!is_duplicate)
				{
					if(i < j)
					{
						dist = D[i*m+j];
					}
					else
					{
						dist = D[j*m+i];
					}
					if(dist < temp)
					{
						temp = dist;
						out[i*k+count] = j;
					}
				}
			}
		}
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
int main(int argc, char *argv[]) 
{ 
	int m,n,k;
	int i;
	int *V, *out;				//host copies
	int *d_V, *d_out;			//device copies
	int *D;						
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
		// allocate space for devices copies
		cudaMalloc((void **)&d_V, m*n*sizeof(int));
		cudaMalloc((void **)&d_out, m*k*sizeof(int));
		cudaMalloc((void **)&D, m*m*sizeof(int));
//		cudaMalloc((void **)&dim_D, m*m*n*sizeof(int));

		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}
		// copy host values to devices copies
		cudaMemcpy(d_V, V, m*n*sizeof(int), cudaMemcpyHostToDevice);

		dim3 blk(m, n);
		// compute the execution time
		cudaEvent_t start, stop;
		// create event
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// record event
		cudaEventRecord(start);
		// launch knn() kernel on GPU
		knn<<<m, blk, m*n*sizeof(int)>>>(m, n, k, d_V, d_out, D);
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
//		cudaFree(dim_D);

		free(V);
		free(out);
	}
	fclose(fp);
	return 0;
}

