import torch
import numpy as np
import shutil
import os
import cv2
from TwinVast.TwinLiteNet_done.model import TwinLite2_old as net
from const import *
from loss import TotalLoss
import argparse
import onnxruntime as ort


W_=640
H_=384

def show_seg_result(img, result, palette=None):
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    
    color_area[result[0] > 100] = [0, 255, 0]
    color_area[result[1] > 100] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    #img = img.astype(np.uint8)
    #img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)
    
    return 

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def Run(model,img):
    img = letterbox(img, (H_, W_))    
    img_rs=img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    # with torch.no_grad():
    #     img_out = model(img)

    out_da, out_ll = ort_session.run(
        ['da', 'll'],
        {"images": img}
    )

    # print(out_da.shape)

    _,da_predict=torch.max(out_da, 1)
    _,ll_predict=torch.max(out_ll, 1)

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    LL = ll_predict.byte().cpu().data.numpy()[0]*255

    # img_rs = cv2.resize(img_rs, (640, 360))
    
    # img_rs[DA>100]= [0,255,0]
    # img_rs[LL>100]=[0,0,255]
    # img_rs = img_rs[:, 12:-12]
    show_seg_result(img_rs, (DA, LL))
    img_rs = img_rs[12:-12, :]

    return img_rs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="large.onnx")
    args = parser.parse_args()

    ort.set_default_logger_severity(4)
    onnx_path = f"./pretrained/{args.weight}"
    ort_session = ort.InferenceSession(onnx_path)
    print(f"Loading done!")

    outputs = ort_session.get_outputs()
    inputs = ort_session.get_inputs()

    image_list=os.listdir('images')

    if(os.path.isdir('large_onnx')):
        shutil.rmtree('large_onnx')
        os.mkdir('large_onnx')
    else:
        os.mkdir('large_onnx')

    for i, imgName in enumerate(image_list):
        img = cv2.imread(os.path.join('images',imgName))
        img=Run(ort_session,img)
        cv2.imwrite(os.path.join('large_onnx',imgName),img)

