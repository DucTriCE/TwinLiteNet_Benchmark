import numpy as np
import os
import torch
import tensorrt as trt
from collections import OrderedDict, namedtuple
import torch.nn as nn
import shutil
import cv2

# def Run(model,img):
#     img = cv2.resize(img, (640, 360))
#     img_rs=img.copy()

#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)
#     img=torch.from_numpy(img)
#     img = torch.unsqueeze(img, 0)  # add a batch dimension
#     img=img.cuda().float() / 255.0
#     img = img.cuda()

#     img_out = model(img)
#     x0=img_out[0]
#     x1=img_out[1]

#     _,da_predict=torch.max(x0, 1)
#     _,ll_predict=torch.max(x1, 1)

#     DA = da_predict.byte().cpu().data.numpy()[0]*255
#     LL = ll_predict.byte().cpu().data.numpy()[0]*255
#     img_rs[DA>100]=[255,0,0]
#     img_rs[LL>100]=[0,255,0]
    
#     return img_rs
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
class TRT(nn.Module):
    def __init__(self,weight='model_best.engine'):
        super().__init__()
        device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weight, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            print(name)
            dtype = trt.nptype(model.get_binding_dtype(i))
            print(dtype)
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
    def forward(self, im):
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]


trt=TRT('/home/ceec/TwinLiteNet_done/large.trt')
img=torch.rand(1,3,384,640).cuda()
#img=torch.rand(1,3,192,320).cuda()

import time

for i in range(10):
    trt(img)

data = []
for i in range(5):
    st=0
    with torch.no_grad():
        for i in range(100):
            t=time.time()
            trt(img)
            st+=(time.time()-t)
    data.append(st*10)

print(data)
mean = np.mean(data)
std_dev = np.std(data)

print("{:.3f} + {:.3f}".format(mean, std_dev))