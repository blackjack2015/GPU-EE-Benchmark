
CUDA_INSTALL_PATH=/usr/local/cuda-10.0
CC = g++
OPTFLAG = -O2
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g -ftz=true ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -lrt
NVCODE = -gencode=arch=compute_60,code=\"compute_60\" -gencode=arch=compute_61,code=\"compute_61\" -gencode=arch=compute_70,code=\"compute_70\" -gencode=arch=compute_75,code=\"compute_75\"
#NVCODE = -gencode=arch=compute_37,code=\"compute_37\" -gencode=arch=compute_60,code=\"compute_60\" -gencode=arch=compute_61,code=\"compute_61\"

.PHONY: all

all: A7-texture

A7-texture: main.o cache_kernels.o
	${CC} -o $@ $^ ${LFLAGS}

main.o: main.cpp cache_kernels.h
	${CC} -c ${FLAGS} $<

cache_kernels.o: cache_kernels.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -c $< -o $@

clean:
	\rm -f A7-texture *.o

rebuild: clean all

