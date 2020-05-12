CFLAGS = -W -Wall -pthread -g $(CFLAGS_EXTRA) 
RM = rm -rf
CC = $(CROSS)gcc
CXX = $(CROSS)g++
PREFIX?=/usr
DESTDIR?=$(PREFIX)

CFLAGS += -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
CFLAGS += -I libyuv/include -DHAVE_JPEG
LDFLAGS += -ljpeg

main: main.cpp tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a libyuv.a
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS)
		
tensorflow/tensorflow/lite/tools/make/Makefile:
	git submodule update --init

tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a: tensorflow/tensorflow/lite/tools/make/Makefile
	tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
	tensorflow/tensorflow/lite/tools/make/build_lib.sh

libyuv.a:
	git submodule init libyuv
	git submodule update libyuv
	cd libyuv && cmake . && make 
	mv libyuv/libyuv.a .
	make -C libyuv clean