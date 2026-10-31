# RV1106 示例工程

本仓库收录 Rockchip RV1106 相关示例，包含 OCR 推理、RKNN 测试和图像匹配项目。本文重点说明 `rv1106_PaddleOcr/PP-OCRv4_RV1106` 的工程结构和使用方法。

## 仓库目录

| 目录 | 用途 |
| --- | --- |
| `RKiveTest/` | RKNN 测试示例，包含 CMake 工程、构建脚本和 C++ 示例入口。 |
| `rv1106_ocr/` | C++ OCR 示例，包含 DBNet、CRNN、方向分类等处理代码及 OpenCV Mobile 依赖。 |
| `rv1106_PaddleOcr/PP-OCRv4_RV1106/` | PaddleOCR PP-OCRv4 在 RV1106 上的 RKNN 检测、识别和部署工程，详见下文。 |
| `wk_Gray_scale_matching/` | 基于 OpenCV 的灰度图像/NCC 匹配示例。 |

## PP-OCRv4 工程结构

进入 `rv1106_PaddleOcr/PP-OCRv4_RV1106/` 后，各目录作用如下：

| 目录 | 用途 |
| --- | --- |
| `accuracy/` | PP-OCRv4 识别准确率对比结果和人工校验数据。 |
| `data/` | 字符字典和默认测试图片。 |
| `install/rv1106_armhf/ppocr_rv1106/` | 已编译的 RV1106 部署包，含 demo、RKNN 模型和运行库。可用于先行验证。 |
| `models/onnx/` | PP-OCRv4 检测和识别 ONNX 模型。 |
| `models/rknn/` | 面向 RV1106 转换的 INT8 RKNN 模型；`candidates/` 保存候选/详细日志版本。 |
| `scripts/` | RKNN 模型转换、交叉编译、板端部署和运行脚本。 |
| `src/ppocr_system/` | C++ OCR 应用源码、后处理以及 RKNN 推理实现。RV1106 构建使用 `rknpu2/` 实现。 |
| `tools/` | 识别准确率和性能数据分析工具。 |
| `vendor/rknn_model_zoo/` | 推理工程依赖、第三方头文件/库及量化校准样例。第三方许可文件随依赖保留。 |
| `build/`、`logs/`、`output/` | 构建目录、运行日志和识别结果；运行时生成，不提交到 Git。 |
| `toolchain/` | 可放置交叉编译器；该目录被忽略，需要自行安装或配置工具链。 |

## 快速运行预编译部署包

以下命令在 Ubuntu/Linux 主机上运行，需要已连接并授权的 RV1106 设备及 `adb`。仓库已经包含预编译部署包和 RKNN 模型，初次验证无需重新转换模型或编译。

```bash
cd rv1106_PaddleOcr/PP-OCRv4_RV1106
adb devices
bash scripts/deploy_and_run.sh
```

脚本会把部署包推送到设备 `/tmp/ppocr_rv1106` 并运行默认测试图片。也可以传入自己的图片路径：

```bash
bash scripts/deploy_and_run.sh /absolute/path/to/image.jpg
```

识别结果保存到 `output/rv1106_test_result.jpg`，运行日志保存到 `logs/rv1106_run.log`。脚本在非 root 用户下通过 `sudo adb` 调用 ADB；如果设备未显示在 `adb devices` 中，请先处理 USB/ADB 连接和授权。

## 从源码重新编译

需要 CMake、GNU Make 和 Luckfox/Rockchip RV1106 ARMHF uClibc 工具链。默认查找位置为工程下的 `toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/`。也可通过环境变量指定工具链前缀：

```bash
cd rv1106_PaddleOcr/PP-OCRv4_RV1106
export RV1106_TOOLCHAIN_PREFIX=/absolute/path/to/arm-rockchip830-linux-uclibcgnueabihf
bash scripts/build_rv1106.sh
```

构建完成后，安装包生成在 `install/rv1106_armhf/ppocr_rv1106/`。若希望重新部署新编译的程序，再运行 `bash scripts/deploy_and_run.sh`。

## 重新转换 RKNN 模型（可选）

仅在替换 ONNX 模型或重新生成量化模型时需要。转换脚本使用 RKNN Toolkit2 和名为 `toolkit2` 的 Conda 环境；脚本默认从 `$HOME/miniforge3/etc/profile.d/conda.sh` 加载 Conda。

```bash
cd rv1106_PaddleOcr/PP-OCRv4_RV1106
bash scripts/convert_models.sh
```

转换结果写入 `models/rknn/`。重新转换后应重新构建并部署，以确保板端安装包使用最新模型。

## 注意事项

- 预编译包面向 RV1106 ARMHF/uClibc 运行环境；不同固件、RKNN Runtime 或板卡镜像可能需要重新编译或替换运行库。
- `toolchain/`、`build/`、`logs/` 和 `output/` 被 `.gitignore` 忽略。交叉编译工具链不随仓库提交。
- 运行脚本会清理设备上的 `/tmp/ppocr_rv1106` 并重新推送当前安装包。
