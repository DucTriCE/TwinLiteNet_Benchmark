import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import random
from calibrator import Int8Calibrator
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

TRT_LOGGER = trt.Logger()

# This function builds an engine from a onnx model.
def build_int8_engine(model_file, calib, batch_size=32):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:    
        config = builder.create_builder_config()
        config.max_workspace_size = 4 * 1 << 30
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib

        with open(model_file, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        engine = builder.build_engine(network, config)
        if engine:
            print("Successfully built the engine!")
        else:
            print("Failed to build the engine.")
        
        return engine

def main():
    calibration_cache = "calibration.cache"
    model_file = "/home/ceec/TwinLiteNet_done/pretrained_ema/nano.onnx"
    training_set = "./images/" 
    batch_size = 1
    calib = Int8Calibrator(training_set, cache_file=calibration_cache, batch_size=batch_size)

    try:
        engine = build_int8_engine(model_file, calib, batch_size)
    except Exception as e:
        print(f"Error during engine building: {e}")

if __name__ == '__main__':
    main()
