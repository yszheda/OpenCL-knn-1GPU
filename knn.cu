/* 
* INPUT:
* m: total num of points
* n: n dimensions
* k: num of nearest points
* V: point coordinates
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
#define MAX_PTRNUM_IN_SMEM 1024 

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
				for(int j=tx; j<TILE_DEPTH; j+=blockDim.x)
				{
					rowVector[py][j] = V[row*n+i*TILE_DEPTH+j];
				}
				for(int j=ty; j<TILE_DEPTH; j+=blockDim.y)
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

//find the min value and index in the count^th loop
__device__ int findMin(int m, int k, int count, int *D, int *out)
{
	int i = blockIdx.x;
  	int tid = threadIdx.x;

	int s = blockDim.x/2;
	int resultValue = INIT_MAX;
	int resultIndex = INIT_MAX;
	int indexBase = (m<MAX_PTRNUM_IN_SMEM)? m: MAX_PTRNUM_IN_SMEM;
	
	for(int num=0; num<m; num+=MAX_PTRNUM_IN_SMEM)
	{
		for(int j=tid; j<indexBase; j+=blockDim.x)
		{
			if(j+num == i)
			{
				SMem[j] = INIT_MAX;
			}
			else
			{
				SMem[j] = D[i*m+num+j];
			}
			//index
			SMem[indexBase+j] = j+num;
			__syncthreads();
		}
		if(tid < count)
		{
			if(out[i*k+tid]-num>=0 && out[i*k+tid]-num<indexBase)
			{
				SMem[ out[i*k+tid]-num ] = INIT_MAX;
			}
			__syncthreads();
		}
		__syncthreads();

/*
//		for(s=indexBase/2; s>0; s>>=1) 
		for(s=indexBase/2; s>32; s>>=1) 
		{
			for(int j=tid; j<indexBase; j+=blockDim.x)
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
*/
		if(indexBase >= 1024)
		{
			for(int j=tid; j<indexBase; j+=blockDim.x)
			{
				if(j < 512) 
				{
					if(SMem[j] == SMem[j+512])
					{
						if(SMem[indexBase+j] > SMem[indexBase+j+512])
						{
							SMem[indexBase+j] = SMem[indexBase+j+512];
						}
					}
					else if(SMem[j] > SMem[j+512])
					{
						SMem[j] = SMem[j+512];
						SMem[indexBase+j] = SMem[indexBase+j+512];
					}
				}
				__syncthreads();
			}
		}

		if(indexBase >= 512)
		{
			for(int j=tid; j<indexBase; j+=blockDim.x)
			{
				if(j < 256) 
				{
					if(SMem[j] == SMem[j+256])
					{
						if(SMem[indexBase+j] > SMem[indexBase+j+256])
						{
							SMem[indexBase+j] = SMem[indexBase+j+256];
						}
					}
					else if(SMem[j] > SMem[j+256])
					{
						SMem[j] = SMem[j+256];
						SMem[indexBase+j] = SMem[indexBase+j+256];
					}
				}
				__syncthreads();
			}
		}

		if(indexBase >= 256)
		{
			for(int j=tid; j<indexBase; j+=blockDim.x)
			{
				if(j < 128) 
				{
					if(SMem[j] == SMem[j+128])
					{
						if(SMem[indexBase+j] > SMem[indexBase+j+128])
						{
							SMem[indexBase+j] = SMem[indexBase+j+128];
						}
					}
					else if(SMem[j] > SMem[j+128])
					{
						SMem[j] = SMem[j+128];
						SMem[indexBase+j] = SMem[indexBase+j+128];
					}
				}
				__syncthreads();
			}
		}

		if(indexBase >= 128)
		{
			for(int j=tid; j<indexBase; j+=blockDim.x)
			{
				if(tid < 64) 
				{
					if(SMem[j] == SMem[j+64])
					{
						if(SMem[indexBase+j] > SMem[indexBase+j+64])
						{
							SMem[indexBase+j] = SMem[indexBase+j+64];
						}
					}
					else if(SMem[j] > SMem[j+64])
					{
						SMem[j] = SMem[j+64];
						SMem[indexBase+j] = SMem[indexBase+j+64];
					}
				}
				__syncthreads();
			}
		}

		__syncthreads();
		if(tid < 32)
		{
			/*
			#pragma unroll 5
			for(s=32; s>0; s>>=1)
			{ 
				if(SMem[tid] == SMem[tid+s])
				{
					if(SMem[indexBase+tid] > SMem[indexBase+tid+s])
					{
						SMem[indexBase+tid] = SMem[indexBase+tid+s];
					}
				}
				else if(SMem[tid] > SMem[tid+s])
				{
					SMem[tid] = SMem[tid+s];
					SMem[indexBase+tid] = SMem[indexBase+tid+s];
				}
			}
			*/
			if(SMem[tid] == SMem[tid+32])
			{
				if(SMem[indexBase+tid] > SMem[indexBase+tid+32])
				{
					SMem[indexBase+tid] = SMem[indexBase+tid+32];
				}
			}
			else if(SMem[tid] > SMem[tid+32])
			{
				SMem[tid] = SMem[tid+32];
				SMem[indexBase+tid] = SMem[indexBase+tid+32];
			}
			if(SMem[tid] == SMem[tid+16])
			{
				if(SMem[indexBase+tid] > SMem[indexBase+tid+16])
				{
					SMem[indexBase+tid] = SMem[indexBase+tid+16];
				}
			}
			else if(SMem[tid] > SMem[tid+16])
			{
				SMem[tid] = SMem[tid+16];
				SMem[indexBase+tid] = SMem[indexBase+tid+16];
			}
			if(SMem[tid] == SMem[tid+8])
			{
				if(SMem[indexBase+tid] > SMem[indexBase+tid+8])
				{
					SMem[indexBase+tid] = SMem[indexBase+tid+8];
				}
			}
			else if(SMem[tid] > SMem[tid+8])
			{
				SMem[tid] = SMem[tid+8];
				SMem[indexBase+tid] = SMem[indexBase+tid+8];
			}
			if(SMem[tid] == SMem[tid+4])
			{
				if(SMem[indexBase+tid] > SMem[indexBase+tid+4])
				{
					SMem[indexBase+tid] = SMem[indexBase+tid+4];
				}
			}
			else if(SMem[tid] > SMem[tid+4])
			{
				SMem[tid] = SMem[tid+4];
				SMem[indexBase+tid] = SMem[indexBase+tid+4];
			}
			if(SMem[tid] == SMem[tid+2])
			{
				if(SMem[indexBase+tid] > SMem[indexBase+tid+2])
				{
					SMem[indexBase+tid] = SMem[indexBase+tid+2];
				}
			}
			else if(SMem[tid] > SMem[tid+2])
			{
				SMem[tid] = SMem[tid+2];
				SMem[indexBase+tid] = SMem[indexBase+tid+2];
			}
			if(SMem[tid] == SMem[tid+1])
			{
				if(SMem[indexBase+tid] > SMem[indexBase+tid+1])
				{
					SMem[indexBase+tid] = SMem[indexBase+tid+1];
				}
			}
			else if(SMem[tid] > SMem[tid+1])
			{
				SMem[tid] = SMem[tid+1];
				SMem[indexBase+tid] = SMem[indexBase+tid+1];
			}
		}
	
		__syncthreads();
		if(resultValue == SMem[0])
		{
			if(resultIndex > SMem[indexBase])
			{
				resultIndex = SMem[indexBase];
			}
		} 
		else if (resultValue > SMem[0])
		{
			resultValue = SMem[0];
			resultIndex = SMem[indexBase];
		}
		__syncthreads();
	}
	return resultIndex;

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

		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}

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

		// copy host values to devices copies
		cudaMemcpy(d_V, V, m*n*sizeof(int), cudaMemcpyHostToDevice);

		int gridDimX = (int)(ceil((float)m/TILE_WIDTH));
		int gridDimY = (int)(ceil((float)m/TILE_WIDTH));

		dim3 grid(gridDimX, gridDimY);
		dim3 block(TILE_WIDTH, TILE_WIDTH);

		// launch knn() kernel on GPU
		computeDist<<<grid, block>>>(m, n, d_V, D);
		cudaDeviceSynchronize();

		int threadNum = (m<MAX_BLOCK_SIZE)? m: MAX_BLOCK_SIZE;
		int ptrNumInSMEM = (m<MAX_PTRNUM_IN_SMEM)? m: MAX_PTRNUM_IN_SMEM;
		knn<<<m, threadNum, 2*ptrNumInSMEM*sizeof(int)>>>(m, k, d_V, D, d_out);

		// copy result back to host
		cudaMemcpy(out, d_out, m*k*sizeof(int), cudaMemcpyDeviceToHost);

		// cleanup
		cudaFree(d_V);
		cudaFree(d_out);
		cudaFree(D);

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

		free(V);
		free(out);
	}
	fclose(fp);
	return 0;
}

