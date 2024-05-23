import argparse
import torch
import torch.nn as nn
import torchvision
from utils import train, val
import DataSet as myDataLoader
from torch.quantization.observer import MovingAverageMinMaxObserver
import torch
import torch
from model import TwinLite2q as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import val, val_cpu, netParams
import torch.optim.lr_scheduler
from const import *
from loss import TotalLoss

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib

import numpy as np
import time
import random
import yaml
from pathlib import Path
from tqdm import tqdm
import os
def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (_,image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model((image/ 255.0).cuda())
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

def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir):


    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

if __name__ == '__main__':

    batch_size = 16
    num_workers = 12
    quant_modules.initialize()
    calibrator="max"
    quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
    with open('/home/ceec/huycq/TwinVast_1/TwinLiteNet_v2/hyperparameters/twinlitev2_hyper.yaml', errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], hyp["hgain"], hyp["sgain"], hyp["vgain"], valid=True),
        batch_size=16, shuffle=False, num_workers=20, pin_memory=True)

    model = net.TwinLiteNet("large")
    model.load_state_dict(torch.load('pretrained/model_25.pth'))
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
        cudnn.benchmark = True
    
    model.eval()
    
    

    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name="large",
            data_loader=valLoader,
            num_calib_batch=128,
            calibrator="max",
            hist_percentile=[99.9, 99.99, 99.999, 99.9999],
            out_dir="./")
    print("start validation")
    da_segment_results,ll_segment_results = val(valLoader, model)

    msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    print(msg)


    # dummy_input = torch.randn(1, 3, 384, 640, device='cuda')

    # input_names = [ "actual_input_1" ]
    # output_names = [ "output1","output2" ]

    # with pytorch_quantization.enable_onnx_export():
    #     # enable_onnx_checker needs to be disabled. See notes below.
    #     torch.onnx.export(
    #         model, dummy_input, "quant_twinL.onnx", verbose=True, opset_version=10, enable_onnx_checker=False
    #         )