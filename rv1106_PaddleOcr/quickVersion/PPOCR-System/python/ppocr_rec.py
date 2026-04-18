# ppocr_rec.py
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0# ppocr_rec.py
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import numpy as np
from utils.rec_postprocess import CTCLabelDecode

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 模型输入形状 - 根据模型信息是 [1, 48, 320, 3] (NHWC)
REC_INPUT_SHAPE = [48, 320]  # h, w
CHARACTER_DICT_PATH = '../model/ppocr_keys_v1.txt'

POSTPROCESS_CONFIG = {
    'CTCLabelDecode': {
        "character_dict_path": CHARACTER_DICT_PATH,
        "use_space_char": True
    }
}

class TextRecognizer:
    def __init__(self, args) -> None:
        self.model, self.framework = self.setup_model(args)
        self.ctc_postprocess = CTCLabelDecode(**POSTPROCESS_CONFIG['CTCLabelDecode'])

    def preprocess(self, img):
        # 调整图像大小到模型期望的输入尺寸
        img = cv2.resize(img, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))  # (w, h) - OpenCV的resize参数是(w,h)
        
        # 确保图像是RGB格式
        if len(img.shape) == 3 and img.shape[2] == 3:
            if not hasattr(self, '_converted_color'):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._converted_color = True
        
        # 确保数据类型为uint8 (这是int8模型的要求)
        if img.dtype != np.uint8:
            if img.dtype in [np.float32, np.float64]:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # 添加批次维度 - 模型期望 [1, 48, 320, 3] (NHWC)
        processed_img = np.expand_dims(img, axis=0)  # 添加batch维度
        
        return processed_img

    def run(self, img):
        if isinstance(img, list):
            # 如果传入的是列表，逐个处理
            outputs = []
            for single_img in img:
                model_input = self.preprocess(single_img)
                
                print(f"Input data shape: {model_input.shape}, dtype: {model_input.dtype}")
                print(f"Input data range: [{model_input.min()}, {model_input.max()}]")
                
                # 运行推理，输入已经是NHWC格式
                output = self.model.run([model_input])
                
                # 处理输出 - 如果是int8需要反量化
                if isinstance(output, list) and len(output) > 0:
                    raw_preds = output[0]
                else:
                    raw_preds = output
                    
                # 检查是否需要反量化
                if raw_preds.dtype == np.int8:
                    print("Info: Quantized INT8 output detected, dequantizing...")
                    # 一般为: float_val = (int8_val - zp) * scale
                    float_preds = (raw_preds.astype(np.float32) - (-128)) * 0.007843
                else:
                    float_preds = raw_preds.astype(np.float32)
                    
                result = self.ctc_postprocess(float_preds)
                outputs.append(result)
            return outputs
        else:
            # 单张图片处理
            model_input = self.preprocess(img)
            
            print(f"Input data shape: {model_input.shape}, dtype: {model_input.dtype}")
            print(f"Input data range: [{model_input.min()}, {model_input.max()}]")
            
            # 运行推理，输入已经是NHWC格式
            output = self.model.run([model_input])
            
            # 处理输出 - 如果是int8需要反量化
            if isinstance(output, list) and len(output) > 0:
                raw_preds = output[0]
            else:
                raw_preds = output
                
            # 检查是否需要反量化
            if raw_preds.dtype == np.int8:
                print("Info: Quantized INT8 output detected, dequantizing...")
                float_preds = (raw_preds.astype(np.float32) - (-128)) * 0.007843
            else:
                float_preds = raw_preds.astype(np.float32)
                
            output = self.ctc_postprocess(float_preds)
            return output

    def setup_model(self, args):
        # 支持两种参数名：model_path 和 rec_model_path
        if hasattr(args, 'model_path'):
            model_path = args.model_path
        elif hasattr(args, 'rec_model_path'):
            model_path = args.rec_model_path
        else:
            raise AttributeError("Args must have either 'model_path' or 'rec_model_path' attribute")
            
        if model_path.endswith('.rknn'):
            platform = 'rknn'
            from py_utils.rknn_executor import RKNN_model_container 
            # 使用与模型编译时相同的平台
            model = RKNN_model_container(model_path, 'rv1103', args.device_id)
        elif model_path.endswith('.onnx'):
            platform = 'onnx'
            from py_utils.onnx_executor import ONNX_model_container
            model = ONNX_model_container(model_path)
        else:
            assert False, "{} is not rknn/onnx model".format(model_path)
        print('Model-{} is {} model, starting val'.format(model_path, platform))
        return model, platform
