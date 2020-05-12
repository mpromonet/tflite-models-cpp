#include <cstdio>
#include <iostream>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void printOutput(std::unique_ptr<tflite::Interpreter> & interpreter, int idx) {
  int output_idx = interpreter->outputs()[idx];
  printf("output_idx:%d ",output_idx);
  
  TfLiteIntArray* odims = interpreter->tensor(output_idx)->dims;
  int size = odims->size;
  int totalsize=1;	
  for (int i=0; i<size; i++) {
	totalsize *= odims->data[i];
	printf("%d",odims->data[i]);
	if ((i+1) != size) {
		printf("x");
	}
  }	  
  printf(" %s\n",TfLiteTypeGetName(interpreter->tensor(output_idx)->type));  
  
  float* output = interpreter->typed_output_tensor<float>(idx);
  for (int i = 0; i < totalsize; ++i) {
		std::cout << i << ": " << output[i] << std::endl; 
  }
}
  
int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "main <tflite model> <file>\n");
    return 1;
  } 
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  int input_idx = interpreter->inputs()[0];
  printf("input_idx:%d ",input_idx);
  
  TfLiteIntArray* dims = interpreter->tensor(input_idx)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  
  printf("%dx%dx%d ",wanted_height,wanted_width,wanted_channels);
  printf("%s\n",TfLiteTypeGetName(interpreter->tensor(input_idx)->type));
  switch (interpreter->tensor(input_idx)->type) {
	case kTfLiteUInt8:
	{
	  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(input_idx);
	  for (int i = 0; i < wanted_height; ++i) {
	    for (int j = 0; j < wanted_width; j++) {
			for (int k = 0; k< wanted_channels; k++) {
				*(input) = 0;
				input++;
			}
	    }
	  }
	  printf("nb elem:%ld\n",(input- interpreter->typed_input_tensor<uint8_t>(input_idx)));
	}
	break;
	case kTfLiteFloat32:
	{
		float* input = interpreter->typed_input_tensor<float>(input_idx);
		std::ifstream is (argv[2]);
		if (is)
		{  
			printf("%s openned\n", argv[2]);
			float c = 0.0;
			for (int i = 0; i < wanted_height; ++i) {
				for (int j = 0; j < wanted_width; j++) {
					for (int k = 0; k< wanted_channels; k++) {
						is.read(reinterpret_cast<char*>(&c), sizeof(float));
						if (is) {
							*(input) = c;
							input++;
						}
					}
				}
			}
		}
		printf("nb elem:%ld\n",(input- interpreter->typed_input_tensor<float>(input_idx)));
	}
	break;
  }

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Read output buffers
  for (int o=0; o < interpreter->outputs().size(); o++) {
	printOutput(interpreter, o);
  }

  return 0;
}
