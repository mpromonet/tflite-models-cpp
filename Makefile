CFLAGS = -W -Wall -pthread -g $(CFLAGS_EXTRA) 
RM = rm -rf
CC = $(CROSS)gcc
CXX = $(CROSS)g++
PREFIX?=/usr
DESTDIR?=$(PREFIX)

CFLAGS += -Itensorflow -Itflite_build/flatbuffers/include/
CFLAGS += -I libyuv/include -DHAVE_JPEG
LDFLAGS += -ljpeg -ldl

main: main.cpp tflite_build/libtensorflow-lite.a tflite_build/_deps/flatbuffers-build/libflatbuffers.a tflite_build/_deps/fft2d-build/libfft2d_fftsg.a tflite_build/_deps/fft2d-build/libfft2d_fftsg2d.a tflite_build/_deps/ruy-build/libruy.a tflite_build/_deps/farmhash-build/libfarmhash.a tflite_build/_deps/xnnpack-build/libXNNPACK.a tflite_build/pthreadpool/libpthreadpool.a tflite_build/cpuinfo/libcpuinfo.a tflite_build/clog/libclog.a libyuv.a
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS)
		
tflite_build/Makefile:
	git submodule update --init
	mkdir tflite_build

tflite_build/libtensorflow-lite.a: tflite_build/Makefile
	cd tensorflow/tensorflow/lite && cmake . && make

libyuv.a:
	git submodule init libyuv
	git submodule update libyuv
	cd libyuv && cmake . && make 
	mv libyuv/libyuv.a .
	make -C libyuv clean