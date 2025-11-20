#!/usr/bin/env make -f
# RM Estimation + RM-CLEAN Makefile
# Auto-detects CUDA and CFITSIO installation paths

CC = nvcc
CFLAGS = -arch=sm_70 -std=c++11 -O3
LDFLAGS = -lcuda -lcudart -lm

TARGET = vroom
SOURCES = main.cpp cuda_rm.cu
OBJECTS = main.o cuda_rm.o

# ============================================================================
# AUTO-DETECT CFITSIO using pkg-config
# ============================================================================

CFITSIO_CFLAGS := $(shell pkg-config --cflags cfitsio 2>/dev/null)
CFITSIO_LIBS := $(shell pkg-config --libs cfitsio 2>/dev/null)

# If pkg-config fails, try common paths
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

CFLAGS += $(CFITSIO_CFLAGS)
LDFLAGS += $(CFITSIO_LIBS)

# ============================================================================
# BUILD TARGETS
# ============================================================================

.PHONY: all clean help info

all: info $(TARGET)
	@echo ""
	@echo "✓ Build successful: ./$(TARGET)"
	@echo ""

info:
	@echo "=========================================="
	@echo "RM Estimation + RM-CLEAN"
	@echo "=========================================="
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "=========================================="
	@echo ""

$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.cpp cuda_rm.h
	@echo "Compiling main.cpp..."
	$(CC) $(CFLAGS) -c main.cpp -o main.o

cuda_rm.o: cuda_rm.cu cuda_rm.h
	@echo "Compiling cuda_rm.cu..."
	$(CC) $(CFLAGS) -c cuda_rm.cu -o cuda_rm.o

clean:
	@echo "Cleaning..."
	@rm -f $(OBJECTS) $(TARGET)
	@echo "Clean complete"

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build (default)"
	@echo "  clean      - Remove objects and binary"
	@echo "  info       - Show build config"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Install dependencies:"
	@echo "  Ubuntu/Debian: apt install libcfitsio-dev"
	@echo "  RedHat/CentOS: yum install cfitsio-devel"
	@echo "  Fedora: dnf install cfitsio-devel"
