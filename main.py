import tflite_runtime.interpreter as tflite
import argparse
import numpy as np
from PIL import Image

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', default='cat.jpg', help='image to process')
    parser.add_argument('-m', '--model_file', default='detect.tflite', help='.tflite model to be executed')
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    channels = input_details[0]['shape'][3]

    img = Image.open(args.image)
    img = img.resize((width, height))
    if channels == 3:
        img = img.convert("RGB")
    if channels == 1:
        img = img.convert("L")
        img = np.expand_dims(img, axis=-1)
    input_data = np.expand_dims(img, axis=0)
    if input_details[0]['dtype'] == np.float32:
       input_data = np.float32(input_data)
    print(np.shape(input_data))

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_details = interpreter.get_output_details()  
    for out in output_details:
        output_data = interpreter.get_tensor(out['index'])
        print(output_data)
