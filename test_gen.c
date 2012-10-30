#include<stdio.h>
#include<stdlib.h>

int main(int argc, char* argv[]){
	srand(time(NULL));
	if(argc != 5) return 0;
	printf("generating test case %s...\n",argv[4]);
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);

	FILE* out = fopen(argv[4], "w");

	int i,j;
	fprintf(out, "%d %d %d\n",m,n,k);

	for(i = 0; i < m; i++){
		for(j = 0; j< n; j++){
			fprintf(out, "%d ", (rand()%10)-5);
		}
		fprintf(out, "\n");
	}

	fclose(out);
	return 0;
}
