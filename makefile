CC = nvcc

CUDA_INC = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64 -lcufft -lcudart
C_LIB = -lm


default: run

run: main.o
	$(CC) Solver.o -o run $(CUDA_LIB) $(C_LIB)

main.o: Solver.cu
	$(CC) $(CUDA_INC) -c Solver.cu -o Solver.o 

test: test.o
	$(CC) test.o -o runtest $(CUDA_LIB) $(C_LIB)

test.o: test.cu
	$(CC) $(CUDA_INC) -c test.cu -o test.o -std=c++11

clean:
	rm -f *.o run runtest