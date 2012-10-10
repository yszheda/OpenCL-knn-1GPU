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

__device__ void computeDimDist(int m, int n, int *V)
{
	int i = blockIdx.x;
   	int j = threadIdx.x;
	int k = threadIdx.y;
	block_dim_D[j*n+k] = (V[i*n+k]-V[j*n+k])*(V[i*n+k]-V[j*n+k]);
}

__device__ void computeDist(int m, int n, int *V, int *D)
{
	int i = blockIdx.x;
   	int j = threadIdx.x;
	int k = threadIdx.y;
	int s;
	int base_idx;
	int dist = 0;
	//dimention calculation
	computeDimDist(m, n, V);
	__syncthreads();
	//reduce duplications
	base_idx = j*n;
	if(i < j)
	{
		//paralle reduction
		for(s=n/2; s>0; s>>=1)
		{
			if(k < s)
			{
				block_dim_D[base_idx+k] += block_dim_D[base_idx+k+s];
			}
		}
		__syncthreads();
		dist = block_dim_D[base_idx];
		//when n is odd, the last element of dim_D needs to be added
		if(n > (n/2)*2)
		{
			dist += block_dim_D[base_idx+n-1];
		}
	}
	D[i*m+j] = dist;
}

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
	// let the first thread select the k-min dist
	i = blockIdx.x;
	if(threadIdx.x == 0)
	{
		for(count=0; count<k; count++)
		{
			temp = INIT_MAX;
			for(j=0; j<m; j++)
			{
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
		printf("m:%d, n:%d, k:%d\n", m, n, k);

		V = (int *) malloc(m*n*sizeof(int));
		out = (int *) malloc(m*k*sizeof(int));

		cudaMalloc((void **)&d_V, m*n*sizeof(int));
		cudaMalloc((void **)&d_out, m*k*sizeof(int));

		cudaMalloc((void **)&D, m*m*sizeof(int));
//		cudaMalloc((void **)&dim_D, m*m*n*sizeof(int));

		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}
		
		cudaMemcpy(d_V, V, m*n*sizeof(int), cudaMemcpyHostToDevice);

//		dim3 blk(m, m, n);
		dim3 blk(m, n);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		knn<<<m, blk, m*n*sizeof(int)>>>(m, n, k, d_V, d_out, D);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		printf("GPU calculation time:%f ms\n", time);

		cudaMemcpy(out, d_out, m*k*sizeof(int), cudaMemcpyDeviceToHost);

		showResult(m, k, out);

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

