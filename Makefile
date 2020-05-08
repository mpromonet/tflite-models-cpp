main: main.cpp tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a
	$(CXX) -o $@ -g -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -pthread $?
		
tensorflow/tensorflow/lite/tools/make/Makefile:
	git submodule update --init

tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a: tensorflow/tensorflow/lite/tools/make/Makefile
	tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
	tensorflow/tensorflow/lite/tools/make/build_lib.sh

