Run sample.py to get calibration.cache for the onnx model
python sample.py

Use trtexec to convert the onnx model to .trt using int8 precision
trtexec --onnx=ResNet50.onnx --explicitBatch --optShapes=000_net:4x3x224x224 --maxShapes=000_net:4x3x224x224 --minShapes=000_net:1x3x224x224 --shapes=000_net:4x3x224x224 --calib=calibration.cache --int8 --saveEngine=ResNet50_int8_batch4.trt
