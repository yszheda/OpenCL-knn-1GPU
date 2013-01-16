NVCC = nvcc
CC = g++
# Add source files here
EXECUTABLE	:= oclKnn
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= oclKnn.cpp

# detect if 32 bit or 64 bit system
HP_64 =	$(shell uname -m | grep 64)
OSARCH= $(shell uname -m)
# architecture flag for nvcc and gcc compilers build
LIB_ARCH        := $(OSARCH)
# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(x86_64),1)
    LIB_ARCH         = x86_64

else
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(i386),1)
        LIB_ARCH         = i386
    else
        ifeq "$(strip $(HP_64))" ""
            LIB_ARCH        = i386
        else
            LIB_ARCH        = x86_64
        endif
    endif
endif

SDKDIR = ../../NVIDIA_GPU_Computing_SDK
SHAREDDIR  := $(SDKDIR)/shared
OCLROOTDIR := $(SDKDIR)/OpenCL
OCLCOMMONDIR ?= $(OCLROOTDIR)/common
OCLLIBDIR    := $(OCLCOMMONDIR)/lib

INCLUDES = -I$(OCLCOMMONDIR)/inc -I$(SHAREDDIR)/inc

LIB := -L${OCLLIBDIR} -L$(SHAREDDIR)/lib
LIB += -lOpenCL -loclUtil_$(LIB_ARCH)$(LIBSUFFIX) -lshrutil_$(LIB_ARCH) ${LIB} -lrt

CFLAGS += $(INCLUDES) $(LIB)

$(EXECUTABLE):
	$(CC) -o $@ $(CCFILES) $(CFLAGS)
clean:
	rm $(EXECUTABLE)
