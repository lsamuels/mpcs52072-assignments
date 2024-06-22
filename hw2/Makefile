NVCC = nvcc
CFLAGS = -Xcompiler -Ofast -I/usr/include/libpng16
LDFLAGS = -lpng

all: grayscale

grayscale: grayscale.o png_flatten.o png_flatten.h
	$(NVCC) -o grayscale grayscale.o png_flatten.o $(LDFLAGS)

grayscale.o: grayscale.cu png_flatten.h
	$(NVCC) $(CFLAGS) -c grayscale.cu

png_flatten.o: png_flatten.c png_flatten.h
	$(NVCC) $(CFLAGS) -c png_flatten.c

clean:
	rm -f *.o grayscale
