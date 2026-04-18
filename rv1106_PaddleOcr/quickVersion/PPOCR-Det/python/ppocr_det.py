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


DET_INPUT_SHAPE = [480, 480] # h,w

ONNX_PRE_PROCESS_CONFIG = [
        {
            'DetResizeForTest': 
            {
                'limit_side_len': 480,
                'limit_type': 'max',
            }
        }, 
        {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, 
        ]

# RKNN的预处理配置。由于RKNN模型内部会处理量化，
# 我们只需要做尺寸调整，不能做归一化。
RKNN_PRE_PROCESS_CONFIG = [
        {
            'DetResizeForTest': {
                    'image_shape': DET_INPUT_SHAPE
                }
         },
         # 移除 'NormalizeImage' 步骤，因为RKNN需要原始的uint8数据
         # {
         #     'NormalizeImage': 
         #     {
         #             'std': [1., 1., 1.],
         #             'mean': [0., 0., 0.],
         #             'scale': '1.',
         #             'order': 'hwc'
         #     }
         # }
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
        self.model, self.framework = setup_model(args)
        self.preprocess_funct = []
        PRE_PROCESS_CONFIG = ONNX_PRE_PROCESS_CONFIG if self.framework == 'onnx' else RKNN_PRE_PROCESS_CONFIG
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        self.db_postprocess = DBPostProcess(**POSTPROCESS_CONFIG['DBPostProcess'])
        self.det_postprocess = DetPostProcess()

    def run(self, img):
        # --- 保存原始图像信息 ---
        original_image_shape = img.shape # [H, W, C]
        orig_h, orig_w = original_image_shape[0], original_image_shape[1]

        # --- 执行预处理 (仅尺寸调整) ---
        input_data = {'image': img, 'shape': [orig_h, orig_w]} # 传递原始形状信息
        for p in self.preprocess_funct:
            input_data = p(input_data)

        # 获取预处理后的图像 (现在应该是uint8, [320, 320, 3])
        processed_img = input_data['image']
        
        # --- 确保数据类型为uint8 ---
        if processed_img.dtype != np.uint8:
            print(f"Warning: Preprocessed image dtype is {processed_img.dtype}, converting to uint8.")
            # 如果是float，需要先转换回0-255范围再转uint8
            if processed_img.dtype in [np.float32, np.float64]:
                 processed_img = np.clip(processed_img * 255.0, 0, 255).astype(np.uint8)
            else:
                 processed_img = processed_img.astype(np.uint8)
        
        # --- 准备模型输入 ---
        # RKNN模型期望的输入通常是 NHWC
        model_input_img = processed_img # [320, 320, 3, uint8]
        if len(model_input_img.shape) == 3: # HWC
            model_input_img = np.expand_dims(model_input_img, axis=0) # Add batch dim -> [1, 320, 320, 3]

        print(f"[Debug] Model Input Shape: {model_input_img.shape}, Dtype: {model_input_img.dtype}")

        # --- 模型推理 ---
        try:
            output = self.model.run([model_input_img])
        except Exception as e:
            print(f"Model inference failed: {e}")
            print("This is likely due to a platform mismatch. Ensure the .rknn model was compiled for your device's exact chip (RV1106).")
            sys.exit(1)

        raw_output = output[0]
        print(f"[Debug] Raw Output Shape: {raw_output.shape}, Range: [{raw_output.min():.4f}, {raw_output.max():.4f}], Dtype: {raw_output.dtype}")

        # --- 处理模型输出 (INT8 -> Float) ---
        # 日志显示输出是 INT8, scale=0.003922, zp=-128
        # 反量化: float_val = (int8_val - zp) * scale
        if raw_output.dtype == np.int8:
            print("Info: Quantized INT8 output detected, dequantizing...")
            float_output = (raw_output.astype(np.float32) - (-128)) * 0.003922
        else:
            print("Info: Float output detected.")
            float_output = raw_output.astype(np.float32)

        preds = {'maps': float_output}

        # --- 计算比率并创建 shape_list ---
        proc_h, proc_w = processed_img.shape[0], processed_img.shape[1]
        ratio_h = proc_h / float(orig_h)
        ratio_w = proc_w / float(orig_w)
        shape_list = [[orig_h, orig_w, ratio_h, ratio_w]]

        # --- 后处理 ---
        result = self.db_postprocess(preds, shape_list)

        # --- 过滤结果 ---
        output = self.det_postprocess.filter_tag_det_res(result[0]['points'], original_image_shape)
        return output

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        # 重要：对于RKNN模型，必须指定正确的target
        # 如果模型编译时指定了rv1103，这里就必须用rv1103
        # 如果模型是为rv1106编译的，这里就要用rv1106
        # 根据错误日志，模型是为rv1103编译的，但硬件是rv1106，这是冲突的根源。
        # 因此，这里传入args.target，让使用者在命令行指定正确的平台。
        model = RKNN_model_container(model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(model_path)
    else:
        assert False, "{} is not rknn/onnx model".format(model_path)
    print('Model-{} is {} model, starting val on target {}'.format(model_path, platform, args.target))
    return model, platform

def init_args():
    parser = argparse.ArgumentParser(description='PPOCR-Det Python Demo')
    # basic params
    parser.add_argument('--model_path', type=str, required=True, help='model path, could be .onnx or .rknn file')
    parser.add_argument('--target', type=str, default='rv1106', help='target RKNPU platform, e.g., rv1106, rv1103, rk3566, etc.')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    return parser

if __name__ == '__main__':
    # Init model
    parser = init_args()
    args =  parser.parse_args()
    print(f"Using target platform: {args.target}")
    det_model = TextDetector(args)
    
    # Set inputs
    img_path = '../model/test.jpg'
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image from {img_path}")
        sys.exit(1)

    # Inference
    print("Starting inference...")
    output = det_model.run(img)

    # Post Process & Visualization
    for box in output:
        box = np.array(box).astype(np.int32)
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Detection results:")
    print(output.tolist())
