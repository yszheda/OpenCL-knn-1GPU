NVCC = nvcc
CC = gcc
knn:
	$(NVCC) -o knn.o knn.cu
clean:
	rm knn.o
test:
	$(CC) -o test_gen test_gen.c
