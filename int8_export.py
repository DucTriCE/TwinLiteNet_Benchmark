import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
quant_modules.initialize()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # print(name, module._calibrator)
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break
    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    #             print(F"{name:40}: {module}")
    model.cuda()

# def test_model_accuracy(fp16_mode=False, int8_mode=False, engine_path=None):
#     if engine_path:
#         engine = load_engine(engine_path)
#     else:
#         if int8_mode:
#             test_data = MiniImageNet('test')
#             dataset = torch.utils.data.DataLoader(test_data, batch_size=_val_batch_size, shuffle=True,
#                                                   num_workers=_num_workers, pin_memory=True)
#             max_batch_for_calibartion = 32
#             transform = None
#             img_size = (3, 224, 224)
#             calibration_stream = ImageBatchStreamDemo(dataset, transform, max_batch_for_calibartion, img_size)
#             calib = EntropyCalibrator(dataset, max_batches=2)
#             engine = build_engine(onnx_file_path=onnx_file_path, engine_file_path=trt_engine_path, fp16_mode=fp16_mode, int8_mode=int8_mode, calib=calib, save_engine=True)
#         else:
#             engine = build_engine(onnx_file_path=onnx_file_path, engine_file_path=trt_engine_path, fp16_mode=fp16_mode, int8_mode=int8_mode, save_engine=True)
#     context = engine.create_execution_context()
#     inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings
#     acc_info = test_model_trt(context, bindings, inputs, outputs, stream)
#     print('trt accuracy', acc_info)

# def test_trt_model_trt(context, bindings, inputs, outputs, stream):
#     test_data = MiniImageNet('test')
#     test_loader = torch.utils.data.DataLoader(
#         test_data,
#         batch_size=_val_batch_size, shuffle=True,
#         num_workers=_num_workers, pin_memory=True)
#     device = 'cuda'
#     return validate_trt(test_loader, context, bindings, inputs, outputs, stream)

# def validate_trt(val_loader, context, bindings, inputs, outputs, stream):
#     top1_hits = []
#     top5_hits = []
#     for images, target in tqdm(val_loader, leave=True, desc='val progress'):
#         target = target.to(device)
#         inputs[0].host = images.numpy().reshape(-1)
#         trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
#         output = postprocess_the_outputs(trt_outputs[0][:images.shape[0] * 100], (images.shape[0], 100))
#         output = torch.from_numpy(output).to(device)
#         for ins_output, ins_target in zip(output, target):
#             top1hit, top5hit = _topk_hit(ins_output, ins_target, topk=(1, 5))
#             top1_hits.append(top1hit)
#             top5_hits.append(top5hit)
#     top1_prec = len([hit for hit in top1_hits if hit]) / len(top1_hits)
#     top5_prec = len([hit for hit in top5_hits if hit]) / len(top5_hits)
#     return {
#         'top1_prec': top1_prec,
#         'top5_prec': top5_prec
#     }

def build_engine(onnx_file_path="", engine_file_path="", fp16_mode=False, int8_mode=False,
                 save_engine=False, calib=None, TRT_LOGGER=trt.Logger()):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        config.max_workspace_size = 4 * 1 << 30
        builder.max_batch_size = 1
        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        elif int8_mode:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
        else:
            pass
            # config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        # Parse model file
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        network.get_input(0).shape = [1, 3, 384, 640]
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        if save_engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
        return engine

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataset, cache_file=" ", batch_size=32, max_batches=2):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.max_batches = max_batches
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_index = 0
        self.batch_count = 0
        # Allocate enough memory for a whole batch.
        self.data = np.zeros((max_batches, 32, 3 , 384, 640))
        for k, (images, targets) in enumerate(self.dataset):
            if k >= self.max_batches: break
            self.data[k] = images.numpy()
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.batch_count < self.max_batches:
            batch = self.data[self.batch_count].ravel()
            cuda.memcpy_htod(self.device_input, batch)
            self.batch_count += 1
            return [self.device_input]
        else:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        return None

if __name__ == '__main__':
    from model import TwinLite2 as net
    import argparse 
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='pretrained_ema/nano.pth', help='model.pt path(s)')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--type', default="nano", help='')
    parser.add_argument('--is320', action='store_true')
    parser.add_argument('--seda', action='store_true', help='sigle encoder for Drivable Segmentation')
    parser.add_argument('--sell', action='store_true', help='sigle encoder for Lane Segmentation')
    args = parser.parse_args()



    device = 'cuda'
    model = net.TwinLiteNet(args)
    model = model.cuda()
    model.load_state_dict(torch.load(args.weights))


    # with torch.no_grad():
    #     collect_stats(model, train_loader, num_batches=10)
    #     compute_amax(model, method="percentile", percentile=99.99)
    # densenet.finetune_model(model)
    # with torch.no_grad():
    #     print(test_model_gpu(model)) # the accuracy is 0.854

    fp16_mode = False
    int8_mode = True
    trt_engine_path = '/home/ceec/TwinLiteNet_done/int8/trt8.0_model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
    batch_size = 1
    onnx_file_path = '/home/ceec/TwinLiteNet_done/pretrained_ema/nano.onnx'
    d_input = torch.randn(batch_size, 3, 384, 640).cuda()
    torch.onnx.export(model, d_input, onnx_file_path, input_names=['input'], output_names=['output'], verbose=False, opset_version=13)
    test_trt_model_accuracy(fp16_mode=fp16_mode, int8_mode=int8_mode)