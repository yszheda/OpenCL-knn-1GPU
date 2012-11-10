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

#define INIT_MAX 10000000
#define TILE_WIDTH 32
#define TILE_DEPTH 128
#define MAX_BLOCK_SIZE 256

void showResult(int m, int k, int *out);


// compute the square of distance of the ith point and jth point
__global__ void computeDist(int m, int n, int *V, int *D)
{
	__shared__ int rowVector[TILE_WIDTH][TILE_DEPTH];
	__shared__ int colVector[TILE_DEPTH][TILE_WIDTH];
	__shared__ int dist[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
   	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row;
	int col;
//	int row = by*TILE_WIDTH+ty;
//	int col = bx*TILE_WIDTH+tx;
	int px;
	int py;	

	for(py=ty; py<TILE_WIDTH; py+=blockDim.y)
	{
		for(px=tx; px<TILE_WIDTH; px+=blockDim.x)
		{
		
			row = by*TILE_WIDTH+py;
			col = bx*TILE_WIDTH+px;
			dist[py][px] = 0;
			__syncthreads();
		
			for(int i=0; i<(int)(ceil((float)n/TILE_DEPTH)); i++)
			{
				for(int j=0; j<TILE_DEPTH; j++)
				{
					rowVector[py][j] = V[row*n+i*TILE_DEPTH+j];
				}
				for(int j=0; j<TILE_DEPTH; j++)
				{		
					colVector[j][px] = V[col*n+i*TILE_DEPTH+j];
				}
				__syncthreads();
		
				for(int j=0; j<TILE_DEPTH; j++)
				{
					dist[py][px] += (rowVector[py][j]-colVector[j][px])*(rowVector[py][j]-colVector[j][px]);
				}
				__syncthreads();
			}
			D[row*m+col] = dist[py][px];
		}
	}
}

extern __shared__ int SMem[];

__device__ void initSMem(int m, int k, int count, int *D, int *out)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	if(j == i)
	{
		SMem[i] = INIT_MAX;
	}
	else
	{
		SMem[j] = D[i*m+j];
	}
	if(j < count)
	{
		SMem[ out[i*k+j] ] = INIT_MAX;
	}
	//index
	SMem[j+m] = j;
}

__device__ int findMin(int m, int k, int count, int *D, int *out)
{
	int i = blockIdx.x;
  	int tid = threadIdx.x;
//  	int j = threadIdx.x;
	int s = blockDim.x/2;

	int indexBase = m;

//	initSMem(m, k, count, D, out);

//	j = tid;
//	while(j < m)
	for(int j=tid; j<m; j+=blockDim.x)
	{
		if(j == i)
		{
			SMem[i] = INIT_MAX;
		}
		else
		{
			SMem[j] = D[i*m+j];
		}
		//index
		SMem[indexBase+j] = j;
		__syncthreads();
/*
		if(j < count)
		{
			SMem[ out[i*k+j] ] = INIT_MAX;
		}
		__syncthreads();
*/
//		j+=blockDim.x;
	}
/*
	for(j=tid; j<m; j+=blockDim.x)
	{
		if(j < count)
		{
			SMem[ out[i*k+j] ] = INIT_MAX;
		}
		__syncthreads();
	}
*/
	for(int j=0; j<count; j++)
	{
		SMem[ out[i*k+j] ] = INIT_MAX;
	}
	__syncthreads();
/*
	if(tid < count)
	{
		SMem[ out[i*k+tid] ] = INIT_MAX;
	}
	__syncthreads();
*/
//	for(s=blockDim.x/2; s>0; s>>=1) 
	for(s=m/2; s>0; s>>=1) 
	{
		for(int j=tid; j<m; j+=blockDim.x)
		{
			if(j < s) 
			{
				if(SMem[j] == SMem[j+s])
				{
					if(SMem[indexBase+j] > SMem[indexBase+j+s])
					{
						SMem[indexBase+j] = SMem[indexBase+j+s];
					}
				}
				else if(SMem[j] > SMem[j+s])
				{
					SMem[j] = SMem[j+s];
					SMem[indexBase+j] = SMem[indexBase+j+s];
				}
			}
			__syncthreads();
		}
	}
	return SMem[indexBase];
}

// compute the k nearest neighbors
__global__ void knn(int m, int k, int *V, int *D, int *out)
{
	int i;
	int count;

	i = blockIdx.x;
	__syncthreads();
	for(count=0; count<k; count++)
	{
		out[i*k+count] = findMin(m, k, count, D, out);
		__syncthreads();
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
			printf("%d ", D[i*m+j]);
			if(j == m-1)
			{
				printf("\n");
			}	
		}    	
	}        	
printf("D:\n");
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

int *h_D;

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
	while(fscanf(fp, "%d %d %d", &m, &n, &k) != EOF)
	{
		V = (int *) malloc(m*n*sizeof(int));
		out = (int *) malloc(m*k*sizeof(int));

		h_D = (int *) malloc(m*m*sizeof(int));
//		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);	

		// compute the execution time
		cudaEvent_t start, stop;
		// create event
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// record event
		cudaEventRecord(start);

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

		int gridDimX = (int)(ceil((float)m/TILE_WIDTH));
		int gridDimY = (int)(ceil((float)m/TILE_WIDTH));

		dim3 grid(gridDimX, gridDimY);
//		dim3 grid(m, m);
		dim3 block(TILE_WIDTH/2, TILE_WIDTH/2);
//		dim3 block(TILE_WIDTH, TILE_WIDTH);
		// launch knn() kernel on GPU
		computeDist<<<grid, block>>>(m, n, d_V, D);
//		computeDist<<<grid, n, n*sizeof(int)>>>(m, n, d_V, D);
		cudaDeviceSynchronize();

cudaMemcpy(h_D, D, m*m*sizeof(int), cudaMemcpyDeviceToHost);
//		showD(m, h_D);

		int threadNum = (m<MAX_BLOCK_SIZE)? m: MAX_BLOCK_SIZE;
//		knn<<<m, m, 2*m*sizeof(int)>>>(m, k, d_V, D, d_out);
		knn<<<m, threadNum, 2*m*sizeof(int)>>>(m, k, d_V, D, d_out);

		// copy result back to host
		cudaMemcpy(out, d_out, m*k*sizeof(int), cudaMemcpyDeviceToHost);

		// record event and synchronize
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float time;
		// get event elapsed time
		cudaEventElapsedTime(&time, start, stop);

		showResult(m, k, out);
		if(m == 1024) {
			printf("SMALL:");
		} else if(m == 4096) {
			printf("MIDDLE:");
		} else if(m == 16384) {
			printf("LARGE:");
		}
		printf("%f\n", time);

		// cleanup
		cudaFree(d_V);
		cudaFree(d_out);
		cudaFree(D);
		free(V);
		free(out);
free(h_D);
	}
	fclose(fp);
	return 0;
}

