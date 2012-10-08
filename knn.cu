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

// extern __shared__ int D[];

__device__ void computeDist(int m, int n, int *V, int *D)
// __device__ void computeDist(int m, int n, int *V)
{
	int i=blockIdx.x;
   	int j=threadIdx.x;
	int k;
	int dist = 0;
	for(k=0; k<n; k++)
	{
		dist += (V[i*n+k]-V[j*n+k])*(V[i*n+k]-V[j*n+k]);
	}
	D[i*m+j] = dist;
}

__global__ void knn(int m, int n, int k, int *V, int *out, int *D)
// __global__ void knn(int m, int n, int k, int *V, int *out)
{
	int i,j;
	int dim;
	int temp;
	int sum;
	int count;
	int num;
	int is_duplicate;
//	__shared__ int D[m*m];

	computeDist(m, n, V, D);
//	computeDist(m, n, V);
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
					if(D[i*m+j] < temp)
					{
						temp = D[i*m+j];
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
	
	int *D;						//will be replaced with shared memory

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

		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}
		
		cudaMemcpy(d_V, V, m*n*sizeof(int), cudaMemcpyHostToDevice);

		knn<<<m, m, m*m*sizeof(int)>>>(m, n, k, d_V, d_out, D);

		cudaMemcpy(out, d_out, m*k*sizeof(int), cudaMemcpyDeviceToHost);

		showResult(m, k, out);

		cudaFree(d_V);
		cudaFree(d_out);

		free(V);
		free(out);
	}
	fclose(fp);
	return 0;
}

