/* 
* INPUT:
* m: total num of points
* n: n dimensions
* k: num of nearest points
* V: point coordinates
* OUTPUT:
* out: k nearest neighbors
*/

#define INIT_MAX 10000000
#define TILE_WIDTH 32
#define TILE_DEPTH 128
#define MAX_BLOCK_SIZE 256
#define MAX_PTRNUM_IN_SMEM 1024 

// compute the square of distance of the ith point and jth point
__kernel void computeDist(int m, int n, __global int *V, __global int *D)
{
	__local int rowVector[TILE_WIDTH][TILE_DEPTH];
	__local int colVector[TILE_DEPTH][TILE_WIDTH];
	__local int dist[TILE_WIDTH][TILE_WIDTH];

    	// Block index
//	int bx = get_global_id(0);
//   	int by = get_global_id(1);
	int bx = get_group_id(0);
   	int by = get_group_id(1);
    	// Thread index
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	int row;
	int col;
	int px;
	int py;	

	for( py = ty; py < TILE_WIDTH; py += get_local_size(1) )
	{
		for( px = tx; px < TILE_WIDTH; px += get_local_size(0) )
		{
			row = by*TILE_WIDTH+py;
			col = bx*TILE_WIDTH+px;
			dist[py][px] = 0;
			barrier(CLK_LOCAL_MEM_FENCE);
		
			for( int i=0; i < (int)(ceil( (float)n/TILE_DEPTH) ); i++ )
			{
				for( int j = tx; j < TILE_DEPTH; j += get_local_size(0) )
				{
					rowVector[py][j] = V[row*n+i*TILE_DEPTH+j];
				}
				for( int j = ty; j < TILE_DEPTH; j += get_local_size(1) )
				{		
					colVector[j][px] = V[col*n+i*TILE_DEPTH+j];
				}
				barrier(CLK_LOCAL_MEM_FENCE);
		
				for( int j = 0; j < TILE_DEPTH; j++ )
				{
					dist[py][px] += (rowVector[py][j]-colVector[j][px])*(rowVector[py][j]-colVector[j][px]);
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			D[row*m+col] = dist[py][px];
		}
	}
}

//find the min value and index in the count^th loop
//__kernel void findMin(int m, int k, int count, __global int *resultIndex, __global int *D, __global int *out, __local int *SMem)
__kernel void findMin(int m, int k, int count, __global int *index, __global int *D, __global int *out, __local int *SMem)
{
	int i = get_group_id(0);
//	int i = get_global_id(0);
  	int tid = get_local_id(0);

	int s = get_local_size(0)/2;
	int resultValue = INIT_MAX;
	int resultIndex = INIT_MAX;
//	(*resultIndex) = INIT_MAX;
	int indexBase = ( m < MAX_PTRNUM_IN_SMEM )? m: MAX_PTRNUM_IN_SMEM;
	
	for( int num = 0; num < m; num += MAX_PTRNUM_IN_SMEM )
	{
		for( int j = tid; j < indexBase; j += get_local_size(0) )
		{
			if( j+num == i )
			{
				SMem[j] = INIT_MAX;
			}
			else
			{
				SMem[j] = D[i*m+num+j];
			}
			//index
			SMem[indexBase+j] = j+num;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
/*
		if( tid < count )
		{
			if( out[i*k+tid]-num >= 0 && out[i*k+tid]-num < indexBase )
			{
				SMem[ out[i*k+tid]-num ] = INIT_MAX;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
*/
		for( int j = 0; j < count; j++ )
		{
			if( out[i*k+j]-num >= 0 && out[i*k+j]-num < indexBase )
			{
				SMem[ out[i*k+j]-num ] = INIT_MAX;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for(s=indexBase/2; s>0; s>>=1) 
//		for( s = indexBase/2; s > 32; s >>= 1 ) 
		{
			for( int j = tid; j < indexBase; j += get_local_size(0) )
			{
				if( j < s ) 
				{
					if( SMem[j] == SMem[j+s] )
					{
						if( SMem[indexBase+j] > SMem[indexBase+j+s] )
						{
							SMem[indexBase+j] = SMem[indexBase+j+s];
						}
					}
					else if( SMem[j] > SMem[j+s] )
					{
						SMem[j] = SMem[j+s];
						SMem[indexBase+j] = SMem[indexBase+j+s];
					}
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
/*
		if(indexBase >= 1024)
		{
			for(int j=tid; j<512; j+=blockDim.x)
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
				__syncthreads();
			}
		}

		if(indexBase >= 512)
		{
			for(int j=tid; j<256; j+=blockDim.x)
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
				__syncthreads();
			}
		}

		if(indexBase >= 256)
		{
			for(int j=tid; j<128; j+=blockDim.x)
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
				__syncthreads();
			}
		}

		if(indexBase >= 128)
		{
			for(int j=tid; j<64; j+=blockDim.x)
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
				__syncthreads();
			}
		}
*/
/*
		if(tid < 32)
		{
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
			/*
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
			*/
//		}
	
		barrier(CLK_LOCAL_MEM_FENCE);
		if( resultValue == SMem[0] )
		{
			if( resultIndex > SMem[indexBase] )
			{
				resultIndex = SMem[indexBase];
			}
		} 
		else if ( resultValue > SMem[0] )
		{
			resultValue = SMem[0];
			resultIndex = SMem[indexBase];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	(*index) = resultIndex;
//	return resultIndex;

}

// compute the k nearest neighbors
__kernel void knn(int m, int k, __global int *V, __global int *D, __global int *out, __local int *SMem)
{
	int i;
	int count;

	i = get_group_id(0);
//	i = get_global_id(0);
        barrier(CLK_LOCAL_MEM_FENCE);
	for( count = 0; count < k; count++ )
	{
		findMin(m, k, count, (out+i*k+count), D, out, SMem);
//		findMin(m, k, count, &out[i*k+count], D, out, SMem);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
