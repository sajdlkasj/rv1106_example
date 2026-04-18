# ppocr_det.py
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
import cv2
import numpy as np
import argparse
import utils.operators
from utils.db_postprocess import DBPostProcess, DetPostProcess

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

DET_INPUT_SHAPE = [640, 640] # h,w

# 针对RV1103/RV1106模型的检测预处理配置
RKNN_DET_PRE_PROCESS_CONFIG = [
    {
        'DetResizeForTest': {
            'image_shape': DET_INPUT_SHAPE
        }
    },
]

POSTPROCESS_CONFIG = {
    'DBPostProcess':{
        'thresh': 0.3,
        'box_thresh': 0.6,
        'max_candidates': 1000,
        'unclip_ratio': 1.5,
        'use_dilation': False,
        'score_mode': 'fast',
    }
}

class TextDetector:
    def __init__(self, args) -> None:
        self.model, self.framework = self.setup_model(args)
        self.preprocess_funct = []
        
        # 使用检测模型的预处理配置
        PRE_PROCESS_CONFIG = RKNN_DET_PRE_PROCESS_CONFIG
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        self.db_postprocess = DBPostProcess(**POSTPROCESS_CONFIG['DBPostProcess'])
        self.det_postprocess = DetPostProcess()

    def run(self, img):
        # 保存原始图像信息
        original_image_shape = img.shape # [H, W, C]
        orig_h, orig_w = original_image_shape[0], original_image_shape[1]

        # 执行预处理 (仅尺寸调整)
        input_data = {'image': img, 'shape': [orig_h, orig_w]} # 传递原始形状信息
        for p in self.preprocess_funct:
            input_data = p(input_data)

        # 获取预处理后的图像
        processed_img = input_data['image']
        
        # 确保数据类型为uint8
        if processed_img.dtype != np.uint8:
            print(f"Warning: Preprocessed image dtype is {processed_img.dtype}, converting to uint8.")
            if processed_img.dtype in [np.float32, np.float64]:
                # 如果值在[0,1]范围内，需要乘以255
                if processed_img.max() <= 1.0:
                    processed_img = np.clip(processed_img * 255.0, 0, 255).astype(np.uint8)
                else:
                    processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
            else:
                processed_img = processed_img.astype(np.uint8)
        
        # 准备模型输入 - RKNN模型期望NHWC格式
        model_input_img = processed_img
        if len(model_input_img.shape) == 3: # HWC
            model_input_img = np.expand_dims(model_input_img, axis=0) # 添加批次维度 -> [1, 640, 640, 3]

        print(f"[Debug] Model Input Shape: {model_input_img.shape}, Dtype: {model_input_img.dtype}")
        print(f"[Debug] Model Input Range: [{model_input_img.min()}, {model_input_img.max()}]")

        # 模型推理
        try:
            output = self.model.run([model_input_img])
        except Exception as e:
            print(f"Model inference failed: {e}")
            print("Check if the model file is correctly compiled for your target platform.")
            sys.exit(1)

        raw_output = output[0]
        print(f"[Debug] Raw Output Shape: {raw_output.shape}, Range: [{raw_output.min():.4f}, {raw_output.max():.4f}], Dtype: {raw_output.dtype}")

        # 处理模型输出
        # 检测模型输出可能是INT8或FLOAT32
        if raw_output.dtype == np.int8:
            print("Info: Quantized INT8 output detected, dequantizing...")
            # 一般情况: float_val = (int8_val - zp) * scale
            float_output = (raw_output.astype(np.float32) - (-128)) * 0.003922
        else:
            print("Info: Float output detected.")
            float_output = raw_output.astype(np.float32)

        preds = {'maps': float_output}

        # 计算比率并创建 shape_list
        proc_h, proc_w = processed_img.shape[0], processed_img.shape[1]
        ratio_h = proc_h / float(orig_h)
        ratio_w = proc_w / float(orig_w)
        shape_list = [[orig_h, orig_w, ratio_h, ratio_w]]

        # 后处理
        result = self.db_postprocess(preds, shape_list)

        # 过滤结果
        output = self.det_postprocess.filter_tag_det_res(result[0]['points'], original_image_shape)
        return output

    def setup_model(self, args):
        model_path = args.det_model_path
        if model_path.endswith('.rknn'):
            platform = 'rknn'
            from py_utils.rknn_executor import RKNN_model_container 
            # 使用与模型编译时相同的平台
            model = RKNN_model_container(model_path, args.target, args.device_id)
        elif model_path.endswith('onnx'):
            platform = 'onnx'
            from py_utils.onnx_executor import ONNX_model_container
            model = ONNX_model_container(model_path)
        else:
            assert False, "{} is not rknn/onnx model".format(model_path)
        print('Model-{} is {} model, starting val'.format(model_path, platform))
        return model, platform
