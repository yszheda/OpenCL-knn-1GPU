/* 
* m: total num of points
* n: n dimensions
* k: num of nearest points
* V: point coordinates
* out: k nearest neighbors
*/

#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>

#define INIT_MAX 100
void showResult(int m, int k, int *out);

extern __shared__ int D[];

__device__ void computeDimDist(int m, int n, int *V, int *dim_D)
{
	int i = threadIdx.x;
   	int j = threadIdx.y;
	int k = threadIdx.z;
	dim_D[(i*m+j)*n+k] = (V[i*n+k]-V[j*n+k])*(V[i*n+k]-V[j*n+k]);
}

__device__ void computeDist(int m, int n, int *V, int *dim_D)
{
	int i = threadIdx.x;
   	int j = threadIdx.y;
	int k = threadIdx.z;
	int s = blockDim.y/2;
	int dist = 0;
	//dimention calculation
	computeDimDist(m, n, V, dim_D);
	__syncthreads();
	//reduce duplications
//	if(i < j)
//	{
//		for(k=0; k<n; k++)
//		{
//			dist += (V[i*n+k]-V[j*n+k])*(V[i*n+k]-V[j*n+k]);
//		}
//	}
	//paralle reduction
	if(i < j)
	{
		for(s=blockDim.y/2; s>0; s>>=1)
		{
			if(k < s)
			{
				dim_D[(i*m+j)*n+k] += dim_D[(i*m+j)*n+k+s];
			}
		}
		__syncthreads();
		dist += dim_D[(i*m+j)*n];
	}
	D[i*m+j] = dist;
}

//paralle reduction?
//cannot use D now!
__device__ int prmin(int m, int *D, int *R)
{
	int j = threadIdx.y;
	int s = blockDim.y/2;
	R[j] = j;
	__syncthreads();
	for(s=blockDim.y; s>0; s>>=1)
	{
		if(j < s)
		{
			D[j] = D[j]<D[j+s]? D[j]: D[j+s]; 
			R[j] = R[j]<R[j+s]? R[j]: R[j+s]; 
		}
		__syncthreads();
	}
	return R[0];
}

__global__ void knn(int m, int n, int k, int *V, int *out, int *dim_D)
{
	int i,j;
	int temp;
	int count;
	int num;
	int dist;
	int is_duplicate;
//	__shared__ int D[m*m];

	computeDist(m, n, V, dim_D);
	__syncthreads();
	// let the first thread select the k-min dist
	i = threadIdx.x;
	if(threadIdx.y == 0)
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
	
//	int *D;						//will be replaced with shared memory
	int *dim_D;

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

//		cudaMalloc((void **)&D, m*m*sizeof(int));
		cudaMalloc((void **)&dim_D, m*m*n*sizeof(int));

		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}
		
		cudaMemcpy(d_V, V, m*n*sizeof(int), cudaMemcpyHostToDevice);

		dim3 blk(m, m, n);
		knn<<<1, blk, m*m*sizeof(int)>>>(m, n, k, d_V, d_out, dim_D);

		cudaMemcpy(out, d_out, m*k*sizeof(int), cudaMemcpyDeviceToHost);

		showResult(m, k, out);

		cudaFree(d_V);
		cudaFree(d_out);

//		cudaFree(D);
		cudaFree(dim_D);

		free(V);
		free(out);
	}
	fclose(fp);
	return 0;
}

