from flask import Flask, json, request, redirect, url_for, send_from_directory
import tflite_runtime.interpreter as tflite
import argparse
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image

api = Flask(__name__,static_url_path='',static_folder='.')
interpreter = None

def runmodel(img):
    input_details = interpreter.get_input_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    channels = input_details[0]['shape'][3]

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
    answer = []
    for out in output_details:
        output_data = interpreter.get_tensor(out['index'])
        answer.append(output_data.tolist())

    return answer

@api.route('/')
def root():
    return redirect('/upload')

@api.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = secure_filename(file.filename)

        file.save(filename)
        img = Image.open(filename)

        return json.dumps(runmodel(img))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file onchange="form.submit(); form.reset()">
    </form>
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tpu', action='store_true', help='enable tpu')
    parser.add_argument('-m', '--model_file', default='detect.tflite', help='.tflite model to be executed')
    args = parser.parse_args()

    if args.tpu:
        interpreter = tflite.Interpreter(model_path=args.model_file, experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
    else:
        interpreter = tflite.Interpreter(model_path=args.model_file)

    interpreter.allocate_tensors()

    port = int(os.environ.get("PORT", 5000))
    api.run(host='0.0.0.0', port=port)
