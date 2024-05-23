import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

class preprocess_obj_loc(object):
    def __init__(self, resize_tuple=(384,640)):
        self.resize_resolution = resize_tuple

    def process(self, input_image_path):
        image_preprocessed = self._load_and_resize(input_image_path)
        return image_preprocessed

    def _load_and_resize(self, input_image_path):
        image = Image.open(input_image_path)
        h, w = self.resize_resolution
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS))
        image_arr = image_arr.reshape(3,h,w)
        # This particular model requires some preprocessing, specifically, mean normalization.
        input_img = (image_arr / 255.0 - 0.45) / 0.225
        return input_img


class Int8Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, training_data, cache_file, batch_size=1):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        oPreprocessObj = preprocess_obj_loc()
        self.data = []
        for oRootDir, oSubDirs, oFiles in os.walk(training_data):
            for oFile in oFiles:
                if "jpg" not in oFile:
                    continue
                oPreImgNp = oPreprocessObj.process(os.path.join(oRootDir, oFile))
                self.data.append(oPreImgNp)
        self.data = np.array(self.data)           
        self.batch_size = batch_size
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % self.batch_size == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
