# VROOM - GPU RM-Synthesis + RM-CLEAN (NUFFT-accelerated)
# Makefile

NVCC = nvcc
CXX = g++

# CUDA architecture (adjust for your GPU)
# sm_70 = Volta (V100), sm_80 = Ampere (A100), sm_86 = RTX 30xx
CUDA_ARCH ?= sm_89

NVCC_FLAGS = -arch=$(CUDA_ARCH) -std=c++17 -O3 -Xcompiler -Wall
NVCC_FLAGS += -Iinclude

LDFLAGS = -lcudart -lpthread

# ============================================================================
# AUTO-DETECT CFITSIO
# ============================================================================

CFITSIO_CFLAGS := $(shell pkg-config --cflags cfitsio 2>/dev/null)
CFITSIO_LIBS := $(shell pkg-config --libs cfitsio 2>/dev/null)

ifeq ($(CFITSIO_CFLAGS),)
  ifeq ($(wildcard /usr/include/fitsio.h),/usr/include/fitsio.h)
    CFITSIO_CFLAGS = -I/usr/include
    CFITSIO_LIBS = -L/usr/lib64 -lcfitsio
  else ifeq ($(wildcard /usr/local/include/fitsio.h),/usr/local/include/fitsio.h)
    CFITSIO_CFLAGS = -I/usr/local/include
    CFITSIO_LIBS = -L/usr/local/lib -lcfitsio
  else
    $(error fitsio.h not found! Install: apt install libcfitsio-dev)
  endif
endif

NVCC_FLAGS += $(CFITSIO_CFLAGS)
LDFLAGS += $(CFITSIO_LIBS)

# ============================================================================
# cuFINUFFT (required)
# ============================================================================
# Build: https://github.com/flatironinstitute/finufft
#   mkdir build && cd build
#   cmake -D FINUFFT_USE_CUDA=ON -D FINUFFT_USE_CPU=OFF ..
#   cmake --build . --parallel
#
# Library will be in build/ directory

FINUFFT_DIR ?= $(HOME)/finufft
FINUFFT_BUILD ?= $(FINUFFT_DIR)/build

# Include path
CUFINUFFT_CFLAGS := -I$(FINUFFT_DIR)/include

# Library path - cufinufft needs some CPU finufft helpers (gaussquad, next235beven)
ifneq ($(wildcard $(FINUFFT_BUILD)/libcufinufft.a),)
  CUFINUFFT_LIBS := -L$(FINUFFT_BUILD) -lcufinufft
  # Link finufft_common (or full finufft) for shared utility symbols
  ifneq ($(wildcard $(FINUFFT_BUILD)/src/common/libfinufft_common.a),)
    CUFINUFFT_LIBS += -L$(FINUFFT_BUILD)/src/common -lfinufft_common
  else ifneq ($(wildcard $(FINUFFT_BUILD)/src/libfinufft.a),)
    CUFINUFFT_LIBS += -L$(FINUFFT_BUILD)/src -lfinufft
  endif
  CUFINUFFT_LIBS += -lcufft
else ifneq ($(wildcard $(FINUFFT_BUILD)/libcufinufft.so),)
  CUFINUFFT_LIBS := -L$(FINUFFT_BUILD) -Wl,-rpath,$(FINUFFT_BUILD) -lcufinufft -lcufft
else
  $(warning cufinufft library not found in $(FINUFFT_BUILD))
  $(warning Build with: cmake -D FINUFFT_USE_CUDA=ON -D FINUFFT_USE_CPU=OFF ..)
  CUFINUFFT_LIBS := -lcufinufft -lcufft
endif

NVCC_FLAGS += $(CUFINUFFT_CFLAGS)
LDFLAGS += $(CUFINUFFT_LIBS)

# ============================================================================
# FILES
# ============================================================================

TARGET = vroom

SRC_DIR = src
INC_DIR = include

SOURCES = $(SRC_DIR)/vroom.cpp \
          $(SRC_DIR)/io.cpp \
          $(SRC_DIR)/cuda_kernels.cu

OBJECTS = $(SRC_DIR)/vroom.o \
          $(SRC_DIR)/io.o \
          $(SRC_DIR)/cuda_kernels.o

HEADERS = $(INC_DIR)/vroom.h \
          $(INC_DIR)/pipeline.h \
          $(INC_DIR)/io.h \
          $(INC_DIR)/cuda_kernels.h

# ============================================================================
# BUILD RULES
# ============================================================================

.PHONY: all clean help info test test_pixel

all: info $(TARGET)
	@echo ""
	@echo "✓ Build successful: ./$(TARGET)"
	@echo ""
	@echo "Usage: ./$(TARGET) -q Q.fits -u U.fits -f freq.txt -o output"
	@echo "       ./$(TARGET) --help for more options"

info:
	@echo "=========================================="
	@echo "VROOM - GPU RM-Synthesis + RM-CLEAN"
	@echo "        (NUFFT-accelerated)"
	@echo "=========================================="
	@echo "NVCC:      $(NVCC)"
	@echo "ARCH:      $(CUDA_ARCH)"
	@echo "CFITSIO:   $(CFITSIO_LIBS)"
	@echo "cuFINUFFT: $(CUFINUFFT_LIBS)"
	@echo "=========================================="
	@echo ""

$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/vroom.o: $(SRC_DIR)/vroom.cpp $(HEADERS)
	@echo "Compiling vroom.cpp..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(SRC_DIR)/io.o: $(SRC_DIR)/io.cpp $(INC_DIR)/io.h $(INC_DIR)/vroom.h
	@echo "Compiling io.cpp..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(SRC_DIR)/cuda_kernels.o: $(SRC_DIR)/cuda_kernels.cu $(INC_DIR)/cuda_kernels.h $(INC_DIR)/vroom.h
	@echo "Compiling cuda_kernels.cu..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

test_pixel: $(SRC_DIR)/test_pixel.o $(SRC_DIR)/io.o $(SRC_DIR)/cuda_kernels.o
	@echo "Linking test_pixel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/test_pixel.o: $(SRC_DIR)/test_pixel.cpp $(HEADERS)
	@echo "Compiling test_pixel.cpp..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	@echo "Cleaning..."
	rm -f $(OBJECTS) $(TARGET) test_pixel $(SRC_DIR)/test_pixel.o
	rm -f test_*.fits test_*.txt
	@echo "Done"

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

test: $(TARGET)
	@echo "Generating test data..."
	python3 generate_test_data.py
	@echo ""
	@echo "Running VROOM on test data..."
	./$(TARGET) -q test_Q.fits -u test_U.fits -i test_I.fits \
	            -f test_frequencies.txt -o test_output \
	            --save-rmsf --phi-min -200 --phi-max 200 --dphi 2
	@echo ""
	@echo "Test complete! Check test_output_*.fits"

help:
	@echo "VROOM - GPU RM-Synthesis + RM-CLEAN (NUFFT-accelerated)"
	@echo ""
	@echo "Build targets:"
	@echo "  make           Build vroom"
	@echo "  make clean     Remove build files"
	@echo "  make test      Generate test data and run"
	@echo "  make help      Show this help"
	@echo ""
	@echo "Options:"
	@echo "  CUDA_ARCH=sm_XX       Set CUDA architecture (default: sm_70)"
	@echo "  CUFINUFFT_DIR=/path   Set cuFINUFFT install prefix"
	@echo ""
	@echo "Dependencies:"
	@echo "  - CUDA Toolkit"
	@echo "  - cuFINUFFT: https://github.com/flatironinstitute/cufinufft"
	@echo "  - CFITSIO: apt install libcfitsio-dev"
	@echo ""
	@echo "Examples:"
	@echo "  make CUDA_ARCH=sm_80                    # Build for A100"
	@echo "  make CUFINUFFT_DIR=/opt/cufinufft       # Custom cuFINUFFT path"
