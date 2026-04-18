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
REC_INPUT_SHAPE = [320, 48]  # h, w
CHARACTER_DICT_PATH = '../model/ppocr_keys_v1.txt'

POSTPROCESS_CONFIG = {
    'CTCLabelDecode': {
        "character_dict_path": CHARACTER_DICT_PATH,
        "use_space_char": True
    }
}

class TextRecognizer:
    def __init__(self, args) -> None:
        self.model, self.framework = setup_model(args)
        self.ctc_postprocess = CTCLabelDecode(**POSTPROCESS_CONFIG['CTCLabelDecode'])

    def preprocess(self, img):
        # 调整图像大小
        img = cv2.resize(img, (320, 48))  # (w, h) - OpenCV的resize参数是(w,h)
        
        # 转换为RGB（如果需要）
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为float32
        img = img.astype(np.float32)
        
        # 归一化 - 这里需要注意，从模型信息看到zp=-128, scale=0.835294
        # 这意味着 (input - (-128)) * 0.835294 是模型内部的处理方式
        # 所以我们需要做逆变换: input = original_input / scale + zp
        # 实际上模型期望的输入应该是直接转换为uint8格式
        img = img / 255.0  # 归一化到[0,1]
        
        # 根据模型信息，输入是NHWC格式，所以保持形状为 [H, W, C]
        # 最终转换为uint8并添加batch维度
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 确保输入形状为 [1, 48, 320, 3] (NHWC)
        processed_img = np.expand_dims(img_uint8, axis=0)  # 添加batch维度
        
        return processed_img

    def run(self, img):
        model_input = self.preprocess(img)
        
        print(f"Input data shape: {model_input.shape}, dtype: {model_input.dtype}")
        print(f"Input data range: [{model_input.min()}, {model_input.max()}]")
        
        # 直接运行推理，输入已经是NHWC格式
        output = self.model.run([model_input])
        
        # 根据实际输出调整处理方式
        if isinstance(output, list) and len(output) > 0:
            preds = output[0].astype(np.float32)
        else:
            preds = output.astype(np.float32)
            
        output = self.ctc_postprocess(preds)
        return output

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(model_path, 'rv1103', args.device_id)
    elif model_path.endswith('.onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(model_path)
    else:
        assert False, "{} is not rknn/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def init_args():
    parser = argparse.ArgumentParser(description='PPOCR-Rec Python Demo')
    # 基本参数
    parser.add_argument('--model_path', type=str, required=True, help='model path, could be .onnx or .rknn file')
    parser.add_argument('--target', type=str, default='rv1103', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    parser.add_argument('--image_path', type=str, default='../model/word_4.jpg', help='test image path')
    return parser

if __name__ == '__main__':
    # 初始化模型
    parser = init_args()
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist!")
        sys.exit(1)
    
    # 加载模型
    det_model = TextRecognizer(args)
    
    # 设置输入图片
    img_path = args.image_path
    if not os.path.exists(img_path):
        print(f"Error: Image file {img_path} does not exist!")
        sys.exit(1)
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320,48))
    if img is None:
        print(f"Error: Cannot read image {img_path}")
        sys.exit(1)
    
    print(f"Original image shape: {img.shape}")
    
    # 推理
    try:
        output = det_model.run(img)
        print("Recognition result:", output)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
