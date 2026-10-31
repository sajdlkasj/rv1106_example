#!/usr/bin/env python3
"""Convert PP-OCRv4 recognition ONNX to RV1106 INT8 RKNN."""
import argparse
from pathlib import Path
from rknn.api import RKNN


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(root / 'models/onnx/ppocrv4_rec.onnx'))
    parser.add_argument('--dataset', default=str(root / 'vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt'))
    parser.add_argument('--output', default=str(root / 'models/rknn/ppocrv4_rec_rv1106_i8.rknn'))
    args = parser.parse_args()
    rknn = RKNN(verbose=True)
    print('--> Configure RV1106 with PP-OCR rec normalization')
    ret = rknn.config(target_platform='rv1106', mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]])
    if ret != 0:
        raise SystemExit('RKNN config failed: %s' % ret)
    print('--> Load ONNX')
    ret = rknn.load_onnx(model=args.model)
    if ret != 0:
        raise SystemExit('ONNX load failed: %s' % ret)
    print('--> Build INT8 model using the Model Zoo 20-image calibration set')
    ret = rknn.build(do_quantization=True, dataset=args.dataset)
    if ret != 0:
        raise SystemExit('RKNN build failed: %s' % ret)
    print('--> Export %s' % args.output)
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        raise SystemExit('RKNN export failed: %s' % ret)
    rknn.release()
    print('Conversion complete')


if __name__ == '__main__':
    main()
