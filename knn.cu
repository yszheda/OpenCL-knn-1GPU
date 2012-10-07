#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>

#define INIT_MAX 100

/* 
* m: total num of points
* n: n dimensions
* k: num of nearest points
* V: point coordinates
* out: k nearest neighbors
*/

void knn(int m, int n, int k, int *V, int *out)
{
	int i,j;
	int dim;
	int temp;
	int sum;
	int count;
	int last_idx;
	for(i=0; i<m; i++)
	{
		temp = INIT_MAX;
		last_idx = i;
		for(count=0; count<k; count++)
		{
			for(j=0; j<m; j++)
			{
				sum = 0;
				if(j != last_idx && j != i)
				{
					for(dim=0; dim<n; dim++)
					{
						sum+=(V[i*n+dim]-V[j*n+dim])*(V[i*n+dim]-V[j*n+dim]);
					}
					if(sum < temp)
					{
						temp = sum;
						out[i*k+count] = j;
						last_idx = j;
					}
				}
			}
		}
	}
}

int main(int argc, char *argv[])
{
	int m,n,k;
	int i,j;
	int *V;
	int *out;
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
		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}
		knn(m, n, k, V, out);
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
		free(V);
		free(out);
	}
	fclose(fp);
	return 0;
}

