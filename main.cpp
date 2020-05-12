#include <cstdio>
#include <iostream>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "libyuv.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void printOutput(std::unique_ptr<tflite::Interpreter> &interpreter, int idx)
{
  int output_idx = interpreter->outputs()[idx];
  printf("output_idx:%d ", output_idx);

  TfLiteIntArray *odims = interpreter->tensor(output_idx)->dims;
  int size = odims->size;
  int totalsize = 1;
  for (int i = 0; i < size; i++)
  {
    totalsize *= odims->data[i];
    printf("%d", odims->data[i]);
    if ((i + 1) != size)
    {
      printf("x");
    }
  }
  printf(" %s\n", TfLiteTypeGetName(interpreter->tensor(output_idx)->type));

  switch (interpreter->tensor(output_idx)->type)
  {
    case kTfLiteUInt8:
    {
      uint8_t *output = interpreter->typed_output_tensor<uint8_t>(idx);
      for (int i = 0; i < totalsize; ++i)
      {
        std::cout << i << ": " << (int)output[i] << std::endl;
      }
    }
    break;
    case kTfLiteFloat32:
    {
      float *output = interpreter->typed_output_tensor<float>(idx);
      for (int i = 0; i < totalsize; ++i)
      {
        std::cout << i << ": " << output[i] << std::endl;
      }
    }
    break;  
  }
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "main <tflite model> <file>\n");
    return 1;
  }
  const char *filename = argv[1];

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
  int input_idx = interpreter->inputs()[0];
  printf("input_idx:%d ", input_idx);

  TfLiteIntArray *dims = interpreter->tensor(input_idx)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  printf("%dx%dx%d ", wanted_height, wanted_width, wanted_channels);
  printf("%s\n", TfLiteTypeGetName(interpreter->tensor(input_idx)->type));
  switch (interpreter->tensor(input_idx)->type)
  {
  case kTfLiteUInt8:
  {
    std::ifstream is (argv[2]);
    std::string buffer((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    int32_t width = 0;
    int32_t height = 0;
    if (libyuv::MJPGSize((const uint8_t*)buffer.c_str(), buffer.size(), &width, &height) == 0) {
      printf("size:%dx%d\n", width, height);

      int stride_y = width;
      int stride_uv = (width + 1) / 2;

      int imagesize = width*height*3/2;
      uint8_t image[imagesize];

      uint8_t * buffer_y = (uint8_t *)&image;
      uint8_t * buffer_u = buffer_y + width*height;
      uint8_t * buffer_v = buffer_u + width*height/4;

      libyuv::ConvertToI420((const uint8_t*)buffer.c_str(), buffer.size(),
                                                    buffer_y, stride_y,
                                                    buffer_u, stride_uv,
                                                    buffer_v, stride_uv,
                                                    0, 0,
                                                    width, height,
                                                    width, height,
                                                    libyuv::kRotate0, ::libyuv::FOURCC_MJPG);


      int dst_sample_size = width*height*4;                                             
      uint8_t  dst_frame[dst_sample_size];
      libyuv::ConvertFromI420(buffer_y, stride_y,
                                    buffer_u, stride_uv,
                                    buffer_v, stride_uv,
                                    (uint8_t *)&dst_frame, 0,
                                    width, height,
                                    libyuv::FOURCC_RGBA);

      int scale_sample_size = wanted_width*wanted_height*4; 
      uint8_t scaled_frame[scale_sample_size];
      libyuv::ScalePlane(dst_frame,
                width*4,
                width*4,
                height,
                (uint8_t *)&scaled_frame,
                wanted_width*4,
                wanted_width*4,
                wanted_height,
                libyuv::kFilterBox);                                    

      std::ofstream os2("scaled.rgba");
      os2.write((const char*)&scaled_frame, scale_sample_size);

      uint8_t *input = interpreter->typed_input_tensor<uint8_t>(0);
      for (int i = 0; i < wanted_height; ++i)
      {
        for (int j = 0; j < wanted_width; j++)
        {
          for (int k = 0; k < wanted_channels; k++)
          {
            *(input) = scaled_frame[4*(i*wanted_width+j)+k];
            input++;
          }
        }
      }
    }
  }
  break;
  case kTfLiteFloat32:
  {
    float *input = interpreter->typed_input_tensor<float>(0);
    std::ifstream is(argv[2]);
    if (is)
    {
      printf("%s openned\n", argv[2]);
      float c = 0.0;
      for (int i = 0; i < wanted_height; ++i)
      {
        for (int j = 0; j < wanted_width; j++)
        {
          for (int k = 0; k < wanted_channels; k++)
          {
            is.read(reinterpret_cast<char *>(&c), sizeof(float));
            if (is)
            {
              *(input) = c;
              input++;
            }
          }
        }
      }
    }
  }
  break;
  }

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Read output buffers
  for (int o = 0; o < interpreter->outputs().size(); o++)
  {
    printOutput(interpreter, o);
  }

  return 0;
}
