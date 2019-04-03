NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -Xcompiler -Wall 

hello-cuda:hello-cuda.cu
	${NVCC} ${NVCC_FLAGS} $^ -o $@
