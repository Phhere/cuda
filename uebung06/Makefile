NAME=mandelbrot

# Compilers
NVCC   = nvcc
CC     = gcc
CXX    = g++
LINK   = nvcc

CUDA_INCLUDE=/opt/cuda
CUDA_LIBS= -lcuda -lcudart

GL_INCLUDE=/usr/include/GL
GL_LIBS=-lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW

SDK_INCLUDE=/opt/cuda/include

# Flags
COMMONFLAGS =-O2 -m64
NVCCFLAGS   =-gencode arch=compute_20,code=sm_20 -m64 -O2
CXXFLAGS    =
CFLAGS      =


INCLUDES = -I$(CUDA_INCLUDE) -I$(GL_INCLUDE) -I/opt/cuda/samples/common/inc -I/usr/local/cuda-6.5/samples/common/inc
LIBS = -L/usr/X11R6/lib64 $(GL_LIBS) $(CUDA_LIBS)

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(COMMONFLAGS))

OBJS = mandel_GPU.o \
	main.o

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

mandel_GPU.o: mandel_GPU.cuh complexRange.h
main.o: complexRange.h mandel_GPU.cuh
