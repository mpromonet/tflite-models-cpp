
#include <cstdio>
#include <iostream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

using namespace tflite; 

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void printOutput(std::unique_ptr<Interpreter> & interpreter, int idx) {
  int output_idx = interpreter->outputs()[idx];
  printf("output_idx;%d\n",output_idx);
  
  TfLiteIntArray* odims = interpreter->tensor(output_idx)->dims;
  int size = odims->size;
  int totalsize=1;	
  for (int i=0; i<size; i++) {
	  totalsize *= odims->data[i];
      printf("%dx",odims->data[i]);
  }	  
  printf("\n");
  
  float* output = interpreter->typed_output_tensor<float>(idx);
  for (int i = 0; i < totalsize; ++i) {
		std::cout << i << ": " << output[i] << std::endl; 
  }
}
  
int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  int input_idx = interpreter->inputs()[0];
  printf("input_idx;%d\n",input_idx);
  
  TfLiteIntArray* dims = interpreter->tensor(input_idx)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  
  printf("%dx%dx%d\n",wanted_height,wanted_width,wanted_channels);
  
  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
  for (int i = 0; i < wanted_width; ++i) {
    for (int j = 0; j < wanted_width; j++) {
		for (int k = 0; k< wanted_channels; k++) {
			*(input) = 0;
			input++;
		}
    }
  }


  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
//  printf("\n\n=== Post-invoke Interpreter State ===\n");
//  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  for (int o=0; o < interpreter->outputs().size(); o++) {
	printOutput(interpreter, o);
  }

  return 0;
}
