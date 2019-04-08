NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -Xcompiler -Wall 

GCC = g++
CPP_FLAGS = -g -Wall
LD_FLAGS = -L /usr/local/cuda/lib64 -lcudart 

main:main.o kernel.o
	${GCC} ${GCC_FLAGS} $^ -o $@ ${LD_FLAGS}

kernel.o:kernel.cu
	${NVCC} ${NVCC_FLAGS} $^ -c $@

clean:
	rm ./*.o
	rm ./main

