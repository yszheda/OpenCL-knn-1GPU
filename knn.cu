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
//#define TILE_DEPTH 32
#define TILE_DEPTH 128

void showResult(int m, int k, int *out);

extern __shared__ int SMem[];

// compute the square of distance of the ith point and jth point
__global__ void computeDist(int m, int n, int *V, int *D)
{
//	__shared__ int rowVector[TILE_DEPTH][TILE_WIDTH];
//	__shared__ int colVector[TILE_DEPTH][TILE_WIDTH];
	__shared__ int rowVector[TILE_WIDTH][TILE_DEPTH];
	__shared__ int colVector[TILE_DEPTH][TILE_WIDTH];
	__shared__ int dist[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
   	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by*TILE_WIDTH+ty;
	int col = bx*TILE_WIDTH+tx;
	int dim;

	dist[ty][tx] = 0;
	__syncthreads();

//	for(int i=0; i<n/TILE_WIDTH; i++)
	for(int i=0; i<(int)(ceil((float)n/TILE_DEPTH)); i++)
	{
		for(int j=0; j<TILE_DEPTH; j++)
		{
//			rowVector[ty][j] = V[row*m+i*TILE_DEPTH+j];
			rowVector[ty][j] = V[row*n+i*TILE_DEPTH+j];
		}
		for(int j=0; j<TILE_DEPTH; j++)
		{		
//			colVector[j][tx] = V[(i*TILE_DEPTH+j)*m+col];
			colVector[j][tx] = V[col*n+i*TILE_DEPTH+j];
		}
//		rowVector[ty][tx] = V[row*m+i*TILE_DEPTH+tx];
//		colVector[ty][tx] = V[(i*TILE_DEPTH+ty)*m+col];
/*
		for(int j=0; j<TILE_DEPTH/blockDim.x; j++)
		{
			dim = j*blockDim.x+tx;
			rowVector[ty][dim] = V[row*m+i*TILE_DEPTH+dim];
		}
		for(int j=0; j<TILE_DEPTH/blockDim.y; j++)
		{
			dim = j*blockDim.y+ty;
			colVector[dim][tx] = V[(i*TILE_DEPTH+dim)*m+col];
		}
*/
		__syncthreads();

		for(int j=0; j<TILE_DEPTH; j++)
		{
			dist[ty][tx] += (rowVector[ty][j]-colVector[j][tx])*(rowVector[ty][j]-colVector[j][tx]);
		}
		__syncthreads();
	}

//	for(int i=0; i<TILE_WIDTH; i++)
//	{
//		for(int j=0; j<TILE_WIDTH; j++)
//		{
			D[row*m+col] = dist[ty][tx];
//		}
//	}

}

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
//	__shared__ int R[1024];
	int i = blockIdx.x;
  	int j = threadIdx.x;
	int s = blockDim.x/2;
//	int currentScale;
//	int last_s = s;
	int num;

	int indexBase = m;

	initSMem(m, k, count, D, out);
	__syncthreads();

//	initSMem(m, D);
//	__syncthreads();
//	R[j] = j;
//	__syncthreads();
//	if(j == i)
//	{
//		SMem[i] = INIT_MAX;
//		/*
//		for(num=0; num<count; num++)
//		{
//			SMem[ out[i*k+num] ] = INIT_MAX;
//		}
//		*/
//	}
	/*
	for(num=0; num<count; num++)
	{
		SMem[ out[i*k+num] ] = INIT_MAX;
	}
	__syncthreads();
	*/
	if(j < count)
	{
		SMem[ out[i*k+j] ] = INIT_MAX;
	}
	__syncthreads();

//	currentScale=blockDim.x;
	for(s=blockDim.x/2; s>0; s>>=1) 
	{
		if(j < s) 
		{
			if(SMem[j] == SMem[j+s])
			{
				if(SMem[indexBase+j] > SMem[indexBase+j+s])
				{
					SMem[indexBase+j] = SMem[indexBase+j+s];
				}
				/*
				if(R[j] > R[j+s])
				{
					R[j] = R[j+s];
				}
				*/
			}
			else if(SMem[j] > SMem[j+s])
			{
				SMem[j] = SMem[j+s];
				SMem[indexBase+j] = SMem[indexBase+j+s];
//				R[j] = R[j+s];
			}
		}
		__syncthreads();
		/*
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
		*/
	}
	return SMem[indexBase];
//	return R[0];
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
		dim3 block(TILE_WIDTH, TILE_WIDTH);
		// launch knn() kernel on GPU
		computeDist<<<grid, block>>>(m, n, d_V, D);
//		computeDist<<<grid, n, n*sizeof(int)>>>(m, n, d_V, D);
		cudaDeviceSynchronize();


		cudaMemcpy(h_D, D, m*m*sizeof(int), cudaMemcpyDeviceToHost);
//		showD(m, h_D);


		knn<<<m, m, 2*m*sizeof(int)>>>(m, k, d_V, D, d_out);

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

