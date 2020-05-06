import time
import tflite_runtime.interpreter as tflite
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='./Models/MobileNetV2')
parser.add_argument('--model_name', type=str, default='mobilenet_v2')
args = parser.parse_args()

def tflite_inference(args):
    tflite_path = args.model_path + '/' + args.model_name + '.tflite'

    #Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    #get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Prepare random input data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    #Inference
    invoke_start = time.time()

    interpreter.invoke()

    invoke_end = time.time()

    #copy the tensor data
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    #print("FP32 - Time taken %f sec" % (invoke_end - invoke_start))
    print("%f" % (invoke_end - invoke_start))
    return (invoke_end - invoke_start)

def tflite_quant_int8_inference(args):
    tflite_path = args.model_path + '/' + args.model_name + '_quant_int8.tflite'

    #Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    #get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Prepare random input data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    #Inference
    invoke_start = time.time()

    interpreter.invoke()

    invoke_end = time.time()

    #copy the tensor data
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    #print("Int8 - Time taken %f sec" % (invoke_end - invoke_start))
    print("%f" % (invoke_end - invoke_start))
    return (invoke_end - invoke_start)

def tflite_quant_fp16_inference(args):
    tflite_path = args.model_path + '/' + args.model_name + '_quant_fp16.tflite'

    #Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    #get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Prepare random input data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    #Inference
    invoke_start = time.time()

    interpreter.invoke()

    invoke_end = time.time()

    #copy the tensor data
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    #print("FP16 - Time taken %f sec" % (invoke_end - invoke_start))
    print("%f" % (invoke_end - invoke_start))
    return (invoke_end - invoke_start)

def tflite_quant_act_io_inference(args):
    tflite_path = args.model_path + '/' + args.model_name + '_quant_act_io.tflite'

    #Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    #get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Prepare random input data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    #Inference
    invoke_start = time.time()

    interpreter.invoke()

    invoke_end = time.time()

    #copy the tensor data
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    #print("Act_IO - Time taken %f sec" % (invoke_end - invoke_start))
    print("%f" % (invoke_end - invoke_start))
    return (invoke_end - invoke_start)

def main():
    fp32_not_quant = tflite_inference(args)
    fp16_quantization = tflite_quant_fp16_inference(args)
    int8_quantization = tflite_quant_int8_inference(args)
    if args.model_name == 'mobilenet_v2':
        act_io_quantization = tflite_quant_act_io_inference(args)

if __name__ == '__main__':
    main()