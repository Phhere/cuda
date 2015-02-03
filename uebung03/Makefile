NAME=base
# Compilers
NVCC = nvcc
CC = gcc
LINK = nvcc
CUDA_INCLUDE=/opt/cuda
CUDA_LIBS= -lcuda -lcudart
GL_INCLUDE=/usr/include/GL
GL_LIBS=-lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW
SDK_INCLUDE=/opt/cuda/include
# Flags
COMMONFLAGS =-O2 -m64 -march=native
NVCCFLAGS =-gencode arch=compute_30,code=sm_30 -m64 -O2
CXXFLAGS =
CFLAGS =
INCLUDES = -I$(CUDA_INCLUDE) -I$(GL_INCLUDE)
LIBS =  $(GL_LIBS) $(CUDA_LIBS)
ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(COMMONFLAGS))
OBJS = cuda_kernel.o cuda_base.o
# Build rules
.DEFAULT: all

all: $(OBJS)
	$(LINK) -o $(NAME) $(LIBS) $(OBJS)
%.o: %.cu
	$(NVCC) -c $(ALL_CCFLAGS) $(INCLUDES) $<
%.o: %.c
	$(NVCC) -ccbin $(CC) -c $(ALL_CCFLAGS) $(INCLUDES) $<
%.o: %.cpp
	$(NVCC) -ccbin $(CXX) -c $(ALL_CCFLAGS) $(INCLUDES) $<
clean:
	rm $(OBJS) $(NAME)

cuda_kernel.o: cuda_threadindex.cuh
cuda_base.o: cuda_kernel.cuh
