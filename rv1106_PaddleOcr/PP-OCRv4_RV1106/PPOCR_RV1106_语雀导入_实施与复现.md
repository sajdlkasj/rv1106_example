# PP-OCRv4 检测与识别部署到 RV1106

## 实施说明与复现手册

> 这份文档记录了从 PaddleOCR 2.7 系列 PP-OCRv4 检测、识别模型，到 RV1106 RKNN 模型和板端 C++ 程序的完整工程过程。内容包含工程目录、关键设计原因、实际转换和板端运行结果，以及从虚拟机重新转换、编译、部署和更换测试图片的步骤。可将本 Markdown 文件直接导入语雀。
>
> 算子兼容、失败复盘与审计工具扩充日期：2026-10-19。
>
> 官方代码与本工程改动逐项对照扩充日期：2026-10-29，见第 23 章。
>
> ONNX→RKNN 逐条转换教程扩充日期：2026-10-31，见第 24 章。
>
> RV1106 实板速度、两图人工真值精度与计时代码扩充日期：2026-10-31，见第 25 章。

---

## 1. 完成目标和当前状态

本工程把 PP-OCRv4 的两个模型分别部署到 Luckfox Pico / RV1106 板端：

1. **Det（Detection，文字检测）**：找出图片中可能包含文字的区域，并生成文字框。
2. **Rec（Recognition，文字识别）**：把检测到的文字区域裁剪、矫正后，识别成文字序列。
3. **System（端到端流程）**：板端 C++ 程序串起图像读取、检测、文本框后处理、文字区域裁剪、识别、结果绘制和输出图片保存。

工程已完成的验证：

- Det 和 Rec 的 ONNX 模型都已转换为面向 RV1106 的 INT8 RKNN 模型。
- 使用 Luckfox RV1106 Buildroot / uClibc 对应的 ARMhf 交叉工具链编译 C++ 程序。
- 在连接的 RV1106 实物板上运行检测、裁剪、识别和结果绘制流程。
- 官方样例图片在板端识别出 16 行文本，单行置信度约为 0.659–0.710。
- 板端结果图已保存回虚拟机工程目录。
- 模型文件的 SHA256 校验已通过。

最初的运行结果只是**端到端功能冒烟验证**；2026-10-31 又在同一块 RV1106 实板上增加了可开关的分段计时，按“5 次预热 + 30 次正式运行”完成了速度统计，并对工程内两张图片共 19 个视觉文本行建立了人工真值。速度结果、CER、逐行错误、原始日志和统计口径见第 25 章。两张图仍然只是小样本回归集，不能冒充 PaddleOCR 官方大数据集精度；没有 polygon 标注时也不能把“检测框数量一致”写成正式 Det precision/recall。

> **算子分配补充说明：**PP-OCRv4 Rec 的 RKNN 编译表显示 125 个节点分配到 NPU，5 个内部 `exNorm` 节点分配到 CPU，另外输入、输出是 CPU 边界节点。因此 v4 识别器应准确描述为“由 RKNN Runtime 管理的 NPU 为主、少量 LayerNorm CPU fallback 的混合执行”，不能写成内部所有算子均在 NPU。PP-OCRv6 工程的识别内部节点没有该 fallback。详细证据、日志行和判断方法见第 16 章。

## 2. 工程位置与平台信息

### 2.1 工程位置

工程建立在 Ubuntu 20.04 虚拟机内：

~~~text
/home/lubancat/PPOCR_RV1106
~~~

这是一个独立工程目录。模型、工程自己的 C++ 源码、转换脚本、编译脚本、部署脚本、工具链、日志和结果图都集中放在这里。以后要查看或复现，先进入这个目录。

### 2.2 已用环境

| 项目 | 实际环境 / 说明 |
|---|---|
| 虚拟机 | Ubuntu 20.04 |
| RKNN Toolkit2 Python 环境 | conda 环境 toolkit2 |
| RKNN Toolkit2 | 2.3.2 |
| Python | Toolkit2 环境使用 Python 3.8 |
| 交叉编译工具链 | Luckfox 官方 RV1106 Buildroot/uClibc ARMhf 工具链，GCC 8.3.0 |
| 板端 SoC | RV1106G3 |
| 板端 Linux | Linux 5.10.160，armv7l，Buildroot 2023.02.6 |
| 板端 NPU Runtime | 设备上可见 /dev/rknpu，运行时库位于 /oem/usr/lib/librknnmrt.so |
| 工程基线 | Rockchip RKNN Model Zoo 中的 PP-OCR 示例，模型来源说明指向 PaddleOCR release/2.7 |

PaddleOCR release/2.7 是本次选用的官方代码版本基线。Rockchip Model Zoo 的 PP-OCR README 也把模型来源标注为 PaddleOCR release/2.7。RV1106 的本工程适配由工程内脚本和 C++ 改动完成；Rockchip 上游示例的支持平台列表没有列出 RV1106，所以不能把本结果描述成上游开箱即用的 RV1106 官方支持。

## 3. 总体流程

~~~mermaid
flowchart LR
  A[PaddleOCR release/2.7 系列模型] --> B[Det ONNX + Rec ONNX]
  B --> C[RKNN Toolkit2 转换与 INT8 量化]
  C --> D[RV1106 Det RKNN + Rec RKNN]
  D --> E[ARMhf/uClibc C++ 交叉编译]
  E --> F[安装包：程序、模型、运行库]
  F --> G[ADB 部署到 RV1106]
  G --> H[检测框、裁剪文字、识别、绘图]
  H --> I[文本日志 + out.jpg 结果图]
~~~

Det 和 Rec 是两个独立的模型，各自有自己的 ONNX、RKNN 转换产物和预处理方式。板端程序通过前一个模型的检测框，将图片中的文字裁剪出来，再逐个送给识别模型。

### 3.1 一张图片在程序内部经历什么

1. **读图**：系统读取输入图片，并保留原图尺寸，后续需要把检测坐标映射回原图。
2. **准备 Det 输入**：按检测模型要求缩放、填充和规范化图片，写入 Det 的 RKNN 输入内存。
3. **NPU 执行检测**：调用 RKNN Runtime 执行 Det，读取检测输出张量。
4. **DBNet 后处理**：把检测输出转换为文字区域候选框，执行过滤和坐标整理，得到图片中的文字框。
5. **裁剪文字区域**：根据四边形框从原图裁剪文字；遇到倾斜或旋转区域时做相应矫正，避免直接把大图整张交给识别模型。
6. **准备 Rec 输入**：逐个处理文字图块，按识别模型要求缩放宽高比、填充到输入大小，并写入 Rec 的 RKNN 输入内存。
7. **CTC 识别**：Rec 为每个时间步输出字符分数。后处理将分数映射为字符类别，执行 CTC 贪心解码、重复字符合并和字典映射，得到文字及置信度。
8. **保存可检查结果**：在原图上绘制检测框和识别文字，输出 out.jpg；同时把每个区域的坐标、文字和分数写入运行日志。

Det 负责“哪里有字”，Rec 负责“这些字是什么”。如果 Det 漏掉某一行，Rec 不会凭空补出那行；如果框切得太窄、文字太小或裁剪方向错，Rec 的结果也会变差。因此排查识别问题时应先看结果图上的框，再看日志里的识别文字。

## 4. 工程目录结构

以下为主要目录用途。仓库内还包含上游示例的其他文件，因此这里只展开和本次 PP-OCR/RV1106 流程相关的部分。

~~~text
PPOCR_RV1106/
├── README_中文.md                         # 工程快速说明
├── 语雀导入_实施与复现.md                 # 本详细手册
├── models/
│   ├── onnx/
│   │   ├── ppocrv4_det.onnx                # Det 原始 ONNX
│   │   └── ppocrv4_rec.onnx                # Rec 原始 ONNX
│   ├── rknn/
│   │   ├── ppocrv4_det_rv1106_i8.rknn      # RV1106 INT8 Det
│   │   └── ppocrv4_rec_rv1106_i8.rknn      # RV1106 INT8 Rec
│   └── SHA256SUMS                          # 模型文件校验值
├── scripts/
│   ├── convert_models.sh                   # 一次重新转换 Det 和 Rec
│   ├── convert_rec_rv1106.py               # Rec 的 RV1106 转换配置
│   ├── build_rv1106.sh                     # ARMhf/uClibc 交叉编译和安装
│   └── deploy_and_run.sh                   # ADB 部署、运行、拉回结果
├── src/
│   └── ppocr_system/                       # 本工程维护的板端 C++ 工程
├── toolchain/
│   └── arm-rockchip830-linux-uclibcgnueabihf/
│                                            # Luckfox RV1106 交叉工具链
├── vendor/
│   └── rknn_model_zoo/                      # 上游示例、Runtime、图像辅助代码和校准集
├── data/
│   └── test.jpg                             # 默认样例图
├── build/
│   └── rv1106_armhf/                        # CMake 中间构建目录
├── install/
│   └── rv1106_armhf/ppocr_rv1106/           # 板端可部署程序、模型和运行库
├── logs/
│   ├── convert_all.log                      # 模型转换记录
│   ├── build_final.log                      # 编译记录
│   └── rv1106_run.log                       # 最近一次板端运行记录
└── output/
    └── rv1106_test_result.jpg               # 板端处理后拉回的结果图
~~~

建议把 models/onnx 和 models/rknn 看作“模型输入和转换产物”，把 scripts/ 看作“复现入口”，把 src/ppocr_system 看作“工程自己的板端实现”，把 install/rv1106_armhf/ppocr_rv1106 看作“实际推送到板上的软件包”。

## 5. 模型转换做了什么，为什么这么做

本章先解释转换的结构和参数；需要从进入环境开始逐条执行 Det/Rec 转换、保存日志、检查设备表和恢复基线时，直接阅读第 24 章。

### 5.1 为什么经过 ONNX 和 RKNN 两步

PaddleOCR 训练和发布的模型属于 Paddle 生态。RV1106 NPU 使用 Rockchip RKNN Runtime 执行模型，因此部署链路需要先得到可被转换工具读取的 ONNX，再用 RKNN Toolkit2 转成适用于 RV1106 的 RKNN 文件。

本工程保留了 ONNX 原始文件，便于以后检查和再次转换；板端使用的是 RKNN 文件，不会在 RV1106 上运行 ONNX Runtime 或 Paddle Inference。

当前保存的 ONNX 文件大小分别为 Det 约 4.77 MB、Rec 约 10.86 MB；量化后的 RKNN 文件约为 Det 2.1 MB、Rec 4.1 MB。体积变化来自转换和量化表示，不代表识别准确率自动保持不变，因此最终仍要看板端图片结果和业务集评测。

### 5.2 为什么使用 INT8 量化

INT8 量化把模型权重和计算转换为低比特表示，目标是适配边缘 NPU 的部署要求，并降低模型文件和运行资源占用。量化会引入精度变化，因此转换时提供校准图片集，让工具估算各层量化范围。是否满足实际产品的精度要求，还要用业务图片和人工标注另行评估。

本工程采用 RKNN Toolkit2 的静态量化流程，目标芯片为 rv1106，量化类型为 i8。校准数据来自 Rockchip Model Zoo 自带的 PPOCR 20 张图片清单：

~~~text
vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt
~~~

校准集用于转换时的量化范围估算，不等于完整的精度测试集，也不代表对所有字体、背景、语言和拍摄条件都做过验证。

### 5.3 Det 转换配置

Det 使用 Model Zoo 中的官方转换脚本：

~~~text
vendor/rknn_model_zoo/examples/PPOCR/PPOCR-Det/python/convert.py
~~~

工程调用时设置：

| 参数 | 设置 |
|---|---|
| 输入模型 | models/onnx/ppocrv4_det.onnx |
| 芯片目标 | rv1106 |
| 量化类型 | i8 |
| 输出模型 | models/rknn/ppocrv4_det_rv1106_i8.rknn |
| 图像均值 | [123.675, 116.28, 103.53] |
| 图像标准差 | [58.395, 57.12, 57.375] |
| 校准图片 | Model Zoo 提供的 PPOCR 20 张图片清单 |

均值和标准差用于把输入像素规范化到模型训练/导出时对应的数值范围。板端的图像预处理必须保持同一约定，否则输入分布会和转换、训练时不一致。

### 5.4 Rec 转换配置

Rec 使用工程内单独维护的脚本：

~~~text
scripts/convert_rec_rv1106.py
~~~

| 参数 | 设置 |
|---|---|
| 输入模型 | models/onnx/ppocrv4_rec.onnx |
| 芯片目标 | rv1106 |
| 量化 | 开启，INT8 |
| 输出模型 | models/rknn/ppocrv4_rec_rv1106_i8.rknn |
| 图像均值 | [127.5, 127.5, 127.5] |
| 图像标准差 | [127.5, 127.5, 127.5] |
| 校准图片 | 与 Det 相同的 20 张图片清单 |

识别模型接收的是已经裁剪出的文字图块。板端 C++ 路径按识别模型约定传入 uint8 像素，由 RKNN 模型的 mean/std 配置完成相应归一化；宽度不足的图块用 128 灰度值填充。这样可避免在应用层先归一化一次、模型侧又归一化一次。

### 5.5 校验模型文件

工程保存了四个 ONNX/RKNN 模型文件的 SHA256 值。转换、拷贝或备份后，可在工程根目录执行：

~~~bash
cd /home/lubancat/PPOCR_RV1106
sha256sum -c models/SHA256SUMS
~~~

每个文件出现 OK 表示当前文件内容和清单记录一致。如果你重新转换过模型，RKNN 文件自然会有新哈希值；旧清单校验失败时，需要在确认新模型正确后重新生成清单。

## 6. 为什么板端 C++ 做了 RV1106 专用适配

### 6.1 普通输入输出接口在板端失败

Rockchip Model Zoo 中通用 C++ 示例常用 rknn_inputs_set 和 rknn_outputs_get 来提交输入并取回输出。直接在本工程的 RV1106 板端使用这套普通路径时，Runtime 返回过：

~~~text
context config invalid
~~~

这表示此模型和当前 RV1106 Runtime 的上下文配置不接受该普通 IO 绑定路径。继续沿用会在真正推理前失败，转换成功本身并不能证明板端接口已经兼容。

### 6.2 改用 RV1106 零拷贝内存绑定

本工程改成以下步骤：

1. 创建 RKNN 输入/输出内存对象：rknn_create_mem。
2. 按模型查询出的原生输入、输出属性准备匹配的 buffer。
3. 通过 rknn_set_io_mem 把 buffer 绑定到 RKNN Context。
4. 把预处理数据写入绑定的输入内存，触发推理后从绑定的输出内存读取结果。

RV1106 的这条 IO 路径能和当前板端 Runtime 配合。零拷贝接口对数据格式、布局和内存大小有更严格的要求，所以代码使用模型查询到的原生属性，而不是假设所有输出都是普通浮点张量。

这里的“零拷贝”指应用直接向 RKNN 绑定的输入内存写数据、从绑定的输出内存读结果，避免通过普通 RKNN Input/Output 包装接口再次交换张量；图像解码、缩放和预处理仍会产生常规 CPU 图像处理与写入输入 buffer 的数据拷贝，不代表从摄像头到 NPU 的整个链路完全没有内存拷贝。

### 6.3 Det 和 Rec 输出的处理

- Det 的量化 INT8 输出在 C++ 后处理前按量化属性还原为浮点数，再进入文字框生成和过滤逻辑。
- Rec 的板端输出按 Runtime 查询得到的输出数据类型读取；工程处理了该路径中的 FP16 数值，并转换成 CTC 后处理可用的数据。
- Rec 按时间步整理 CTC 输出并做行归一化，然后继续使用已有的贪心解码和字典映射流程。
- 识别置信度使用后处理得到的字符序列得分。原始直接读取的分数曾高于 1；加入与输出格式一致的处理后，板端结果分数回到约 0.66–0.71 的合理范围。

这些处理属于模型输出张量到已有 PP-OCR 后处理之间的格式适配。遇到新版本模型、不同量化配置或不同 Runtime 时，应重新核对模型 IO 查询结果和后处理假设。逐文件改动清单和每项动机见第 14 节。

### 6.4 为什么 C++ 源码单独放在 src/ppocr_system

上游示例放在 vendor/rknn_model_zoo 中，工程的 RV1106 改动放在 src/ppocr_system。这样保留了上游代码作为对照，也让后续查看“原始示例”和“本项目适配”时更清楚。模型路径、后处理和零拷贝改动以本工程源码为准。

## 7. 编译、安装和板端部署

### 7.1 为什么必须用 Luckfox 对应的工具链

板端是 ARMv7、Buildroot 和 uClibc 组合。Windows/Ubuntu 主机上的原生编译器生成的可执行文件不能直接在板端运行；为普通 glibc 系统编译的程序也不一定能适配此处的 uClibc 运行环境。

因此工程使用 Luckfox 提供的 RV1106 ARMhf/uClibc 交叉编译器。它生成的目标程序面向板端系统 ABI。对应说明见 Luckfox 官方交叉编译指南：

- https://wiki.luckfox.com/Luckfox-Pico-RV1106/Cross-Compile/

### 7.2 编译命令

在 Ubuntu 虚拟机中：

~~~bash
cd /home/lubancat/PPOCR_RV1106
scripts/build_rv1106.sh
~~~

脚本会使用工程内的 GCC/G++，通过 CMake 为 rv1106 配置 ARM Linux 编译目标，执行 Release 编译，并安装到：

~~~text
/home/lubancat/PPOCR_RV1106/install/rv1106_armhf/ppocr_rv1106
~~~

如果修改了 C++ 代码或模型，需要重新编译，然后重新部署。只替换模型时，也建议完整执行部署脚本，确保板端目录里的模型和程序来自同一版工程。

### 7.3 安装包里包含什么

install/rv1106_armhf/ppocr_rv1106 是推送到板端的完整目录，至少包含：

- rknn_ppocr_system_demo：板端 C++ 可执行文件。
- Det 和 Rec RKNN 模型。
- lib/ 下所需运行库。
- model/ 下的类别/字符字典和默认测试图片等资源。

工程构建后的整个安装目录约 13 MB。具体文件清单以该目录当前内容为准。

### 7.4 板端运行目录为何在 /tmp

本次板端检查发现 /userdata 可用空间约 1.4 MB，而完整程序、模型和动态库需要约 13 MB；将工程安装到 /userdata 会放不下。板端 /tmp 可用空间约 90 MB，因此本工程临时部署到：

~~~text
/tmp/ppocr_rv1106
~~~

模型和安装包长期保存在虚拟机工程中。板子重启后 /tmp 内容会清除，需要重新运行部署脚本。请不要把 /tmp 的临时性误认为工程没有保存在虚拟机。

### 7.5 一键部署并运行默认样例

先确认板子已连到虚拟机的 USB/ADB，并在虚拟机执行：

~~~bash
cd /home/lubancat/PPOCR_RV1106
scripts/build_rv1106.sh
scripts/deploy_and_run.sh
~~~

部署脚本会检查构建结果和 ADB 连接，清理板端本工程专用目录 /tmp/ppocr_rv1106，将 install 目录推送到板上，加载随包提供的运行库，运行默认测试图，然后把输出图拉回虚拟机。

部署脚本清理的范围是 /tmp/ppocr_rv1106 这个工程目录。正常情况下不会清理板上的其他目录。

默认图片流程结束后，查看：

~~~text
工程/output/rv1106_test_result.jpg
工程/logs/rv1106_run.log
~~~

运行记录是最近一次脚本运行的日志；新运行会覆盖该日志和结果图。需要保留历史记录时，请先在虚拟机复制或改名备份。

## 8. 换自己的图片试跑

把图片放在虚拟机可访问的路径，再将图片路径作为脚本参数：

~~~bash
cd /home/lubancat/PPOCR_RV1106
scripts/deploy_and_run.sh /home/lubancat/PPOCR_RV1106/data/your_image.jpg
~~~

实际使用时，将 your_image.jpg 替换为真实文件名。路径必须是 Ubuntu 虚拟机里的路径，不能直接传 Windows 的 D:\ 路径，除非该目录已共享并在 Ubuntu 中挂载。

该脚本每次都会把已编译的程序和模型部署到板端；如果上次已经编译且没有改代码或模型，可以直接运行部署脚本。自定义图片会被推送到安装目录下的 model/input.jpg，板端处理结果仍写回 out.jpg，最后由脚本拉回工程 output/rv1106_test_result.jpg。

第一次建议先用自带 data/test.jpg 确认链路，再换自己的图。OCR 结果受图片清晰度、文字大小、倾斜/透视、字体、语言和背景影响；样例图成功不代表任意输入都会准确。

## 9. 实测结果

### 9.1 测试链路

本次验证包含：

1. 虚拟机把程序、模型和运行库推送到 RV1106。
2. 板端 C++ 程序加载两个 RKNN 模型。
3. Det 输出文字区域。
4. 系统对文字区域裁剪、整理方向并送入 Rec。
5. 板端输出文字和置信度，并生成带框结果图。
6. 虚拟机通过 ADB 取回结果图。

### 9.2 分层验证清单

| 层级 | 验证内容 | 本次结果 | 可以查看的证据 |
|---|---|---|---|
| 模型文件 | Det/Rec ONNX 和 RKNN 文件存在且哈希匹配 | 通过 | models/、models/SHA256SUMS |
| 模型转换 | Toolkit2 能将 Det、Rec 转为 rv1106 i8 | 通过 | logs/convert_all.log、models/rknn/ |
| 交叉编译 | ARMhf/uClibc 工具链能编译并安装 C++ 程序 | 通过 | logs/build_final.log、install/rv1106_armhf/ |
| 设备连接 | 虚拟机经 ADB 操作已连接的 RV1106 | 通过 | deploy_and_run.sh 的执行过程 |
| 模型加载与 IO | 两个 RKNN 模型在板端创建 Context 并使用零拷贝 IO | 通过 | logs/rv1106_run.log 中的板端运行信息 |
| 端到端图片 | 完成检测、文本裁剪、识别及结果图写出 | 通过，样例输出 16 行 | logs/rv1106_run.log、output/rv1106_test_result.jpg |
| 性能评测 | 测量时延、FPS、功耗、精度统计 | 未测 | 需要后续单独设计测试集和计时 |

这张表把“模型可转换”“程序可编译”“板端可运行”和“精度/性能达标”区分开。前三类已经有本次验证结果；最后一类需要结合你的业务输入和验收指标继续测量。

### 9.3 样例识别摘录

默认样例共输出 16 行。下面是板端实际输出的主要内容和分数，用于确认模型链路在板上有结果：

| 序号 | 板端识别文字 | 置信度 |
|---:|---|---:|
| 1 | 纯营养护发素 | 0.710 |
| 2 | 产品信息/参数 | 0.707 |
| 3 | （45元/每公斤，100公斤起订） | 0.695 |
| 4 | 每瓶22元，1000瓶起订） | 0.702 |
| 5 | 【品牌】：代加工方式/OEMODM | 0.700 |
| 6 | 【品名】：纯营养护发素 | 0.702 |
| 7 | 【产品编号】：YM-X-3011 | 0.708 |
| 8 | ODMOEM | 0.708 |
| 9 | 【净含量】：220ml | 0.709 |
| 10 | 【适用人群】：适合所有肤质 | 0.709 |
| 11 | 【主要成分】：硬脂醇、麦β-葡 | 0.680 |
| 12 | 糖、椰油酰胺丙基甜菜、泛配 | 0.659 |
| 13 | （成品包材） | 0.708 |
| 14 | 【主要功能】：可紧致头发磷层，从而达到 | 0.697 |
| 15 | 即时持久改善头发光泽的效果，给干燥的头 | 0.705 |
| 16 | 发足够的滋养 | 0.708 |

这是一张样例图上的运行展示，不是人工逐字校正后的精度评测。可见一些字符可能被误识别、漏识别或在相邻行间断开。若后续用于具体业务，建议用代表性业务图片建立测试集，并分别统计 Det 的漏检/误检和 Rec 的字符/整行准确率。

### 9.4 性能数据边界

目前没有在板端记录单张图片总耗时、Det 耗时、Rec 耗时、帧率或功耗。日志中的识别分数是模型后处理置信度，不是速度指标，也不能直接等同于人工标注准确率。需要性能数据时，可在 C++ 中分别为 Det、裁剪和 Rec 加计时，并固定输入图片、板端频率和 Runtime 版本后重复多次测量。

## 10. 从头复现的建议顺序

按以下顺序操作最容易定位问题：

### 第一步：进入工程并检查模型

~~~bash
cd /home/lubancat/PPOCR_RV1106
sha256sum -c models/SHA256SUMS
~~~

如果只想先跑板端，不需要重新转换模型；跳到第三步。只有模型缺失、准备重新转换或修改转换参数时才做第二步。

### 第二步：重新转换模型（可选）

~~~bash
cd /home/lubancat/PPOCR_RV1106
conda activate toolkit2
scripts/convert_models.sh
~~~

如果新开终端后提示 conda 找不到，先加载安装目录中的 conda 初始化脚本，或者使用工程已验证过的方式激活 toolkit2。转换成功后确认 models/rknn/ 下生成 Det 和 Rec 两个 RKNN 文件，再重新编译、部署。

### 第三步：检查 ADB 连板

~~~bash
adb get-state
~~~

应能看到设备状态为 device。若虚拟机内 ADB 权限配置要求 sudo，工程部署脚本会在非 root 终端下使用 sudo adb。若没有设备，先检查 VM USB 设备是否已连接/切换到虚拟机，然后执行 adb devices 查看状态。

### 第四步：编译板端程序

~~~bash
cd /home/lubancat/PPOCR_RV1106
scripts/build_rv1106.sh
~~~

### 第五步：部署并运行

~~~bash
scripts/deploy_and_run.sh
~~~

### 第六步：看日志和图像

~~~text
logs/rv1106_run.log
output/rv1106_test_result.jpg
~~~

先确认日志中出现检测框和识别结果，再打开结果图确认文字区域框选和图片写出。若需要换图，按第 8 节传入 Ubuntu 中的图片路径。

## 11. 常见问题排查

| 现象 | 常见原因 | 建议检查 |
|---|---|---|
| adb get-state 没有 device | USB 没切换到虚拟机、ADB 未授权或设备未启动 | 查看 adb devices；在虚拟机管理器中把板子 USB 连接给 Ubuntu，重新插拔后再检查 |
| context config invalid | 在 RV1106 使用了普通 inputs_set/outputs_get 路径，或者 IO 属性与模型不匹配 | 使用本工程 src/ppocr_system 的 rknn_create_mem/rknn_set_io_mem 路径；确认部署的模型与程序来自同一次工程构建 |
| C++ 程序提示找不到 .so | 运行库未部署或 LD_LIBRARY_PATH 没包含安装包 lib 目录 | 通过 deploy_and_run.sh 部署；查看 install/rv1106_armhf/ppocr_rv1106/lib 和日志 |
| 板端提示模型文件打不开 | 推送目录不完整、路径不同或文件名不一致 | 检查 install 下模型文件；重新执行部署脚本；查看板端 /tmp/ppocr_rv1106/model |
| 板端没有结果图 | 程序提前退出、图片解码失败或输出路径不可写 | 检查 logs/rv1106_run.log；确认输入图可被 OpenCV 解码；脚本会拉取板端 out.jpg |
| 自定义图没有上传 | 给的是 Windows 路径、文件路径写错或虚拟机无法访问该共享目录 | 用 Ubuntu 绝对路径并先检查文件确实存在 |
| 识别分数异常大或文本乱码 | 模型/字典版本不配、输入归一化重复、输出类型解释不一致 | 使用工程配套 RKNN、字典和预处理；检查 Rec 原始输出类型、量化属性和 CTC 解码 |
| 重启后程序不见了 | 程序部署在 /tmp，重启时会清空 | 回到虚拟机工程目录，重新执行 scripts/deploy_and_run.sh |
| 模型转换中断或找不到 rknn 包 | 没有激活 Toolkit2 环境，或环境版本与工程不符 | 检查 Python 是否来自 toolkit2，确认 rknn-toolkit2 版本为 2.3.2，并查看 logs/convert_all.log |

排查时尽量保留最新日志。板端运行问题优先看 logs/rv1106_run.log；转换问题看 logs/convert_all.log；编译问题看 logs/build_final.log。

## 12. 更新模型或代码时的操作顺序

### 只改板端 C++ 代码

~~~bash
cd /home/lubancat/PPOCR_RV1106
scripts/build_rv1106.sh
scripts/deploy_and_run.sh
~~~

### 重新导出/替换 ONNX 并重新量化

1. 把新的 ONNX 放到 models/onnx/，保持脚本里引用的文件名，或同步修改脚本路径。
2. 检查输入尺寸、通道顺序、均值/标准差、模型输出、字典和后处理是否和已有 C++ 逻辑相符。
3. 激活 toolkit2 并执行 scripts/convert_models.sh。
4. 在受控样例上检查转换日志和结果。
5. 执行 scripts/build_rv1106.sh 和 scripts/deploy_and_run.sh。
6. 确认板端文本与结果图片，再更新 models/SHA256SUMS。

更换模型后，不能只凭“生成了 .rknn 文件”判断移植完成。还要验证板端模型加载、IO 内存绑定、输出解释、后处理和实际图片识别。

## 13. 文件与结果从哪里查看

| 想查看的内容 | 工程内位置 |
|---|---|
| 本次详细说明 | 语雀导入_实施与复现.md |
| 简短启动说明 | README_中文.md |
| 原始 Det/Rec ONNX | models/onnx/ |
| RV1106 INT8 RKNN | models/rknn/ |
| 模型哈希 | models/SHA256SUMS |
| 重新转换脚本 | scripts/convert_models.sh、scripts/convert_rec_rv1106.py |
| 板端 C++ 适配源码 | src/ppocr_system/ |
| 编译脚本 | scripts/build_rv1106.sh |
| ADB 部署和运行脚本 | scripts/deploy_and_run.sh |
| 板端运行日志 | logs/rv1106_run.log |
| 转换日志 | logs/convert_all.log |
| 编译日志 | logs/build_final.log |
| 识别后图片 | output/rv1106_test_result.jpg |
| 板端临时安装位置 | /tmp/ppocr_rv1106 |

## 14. 具体改动清单：相对上游示例改了什么、为什么

这节按文件说明本工程相对 Rockchip Model Zoo 示例做了哪些变化，以及这些变化解决什么问题。读代码时可以先从 CMakeLists.txt、convert_models.sh、convert_rec_rv1106.py 和 rknpu2/ppocr_system.cc 开始。

### 14.1 工程组织上的变化

| 文件或目录 | 上游原状 / 本工程变化 | 这样做的原因 |
|---|---|---|
| vendor/rknn_model_zoo/ | 保存 Rockchip Model Zoo 示例、通用工具、运行库和校准数据；不把 RV1106 改动直接写进 vendor | 上游可作为对照，升级或检查上游时更容易区分本工程修改 |
| src/ppocr_system/ | 放置本工程实际交叉编译的 PPOCR-System C++ 源码 | 提供独立可维护的板端实现，改动清晰集中 |
| models/onnx/、models/rknn/ | 将输入 ONNX 和 RV1106 RKNN 输出放在工程根目录统一管理 | 转换脚本、CMake 安装和部署脚本引用同一组模型，避免拿错模型副本 |
| scripts/ | 增加转换、构建、部署三个复现入口 | 把多条手工命令固定下来，降低后续复现时漏步骤和路径错误的可能 |

### 14.2 模型转换改动

**scripts/convert_models.sh** 负责转换顺序和工程内输入/输出路径：Det 调用 vendor 中原有的 PPOCR-Det 转换程序，显式指定 rv1106 与 i8；Rec 调用工程内的 RV1106 专用转换脚本。这样 Det 可以继续复用上游转换流程，而 Rec 的目标平台、归一化和量化集由工程明确控制。

**scripts/convert_rec_rv1106.py** 与 vendor 的 PPOCR-Rec convert.py 有以下差异：

| 变化 | 为什么要改 |
|---|---|
| 将默认路径改为工程内 models/onnx、models/rknn 和 vendor 校准集，并提供 --model、--dataset、--output 参数 | 脚本在工程根目录任意位置执行时都能定位文件，也可替换模型或校准集 |
| 目标芯片明确设为 rv1106 | vendor 的 Rec 转换脚本帮助文本列出的平台没有 rv1106；本工程需要将目标芯片明确传给 Toolkit2 |
| 配置 mean/std 为 [127.5, 127.5, 127.5] | 与识别模型的图像像素范围和板端 uint8 输入流程一致，让归一化由编译后的模型完成 |
| 将 vendor 默认的 do_quantization=False 改为 do_quantization=True，并传入 dataset_20.txt | 生成本工程需要的 RV1106 INT8 模型，并让 Toolkit2 基于校准图片估算量化参数 |
| 开启详细转换输出，对 config、load、build、export 各阶段检查返回值 | 转换失败时能知道停在哪个阶段，避免把不完整或旧模型误当成新结果 |

### 14.3 CMake 与安装目录改动

**src/ppocr_system/CMakeLists.txt** 指向本工程保存的 vendor/rknn_model_zoo 目录，而不是假设源码位于上游原有目录层级。这是因为本工程的源文件独立放在 src/ppocr_system，继续使用上游相对路径会找不到 utils、3rdparty 和 OpenCV 配置。

CMake 安装规则也改成从工程内的 models/rknn/ 安装 Det/Rec 模型，并在目标安装目录中将名称改为 demo 原有加载逻辑使用的 ppocrv4_det_i8.rknn 和 ppocrv4_rec_i8.rknn；默认测试图从工程的 data/test.jpg 安装到 model/。这样运行程序时，二进制文件预期的相对路径和实际安装文件能够对上。

**scripts/build_rv1106.sh** 增加固定的工具链、CMake 架构和安装路径设置：工具链为工程内 Luckfox RV1106 ARMhf/uClibc GCC，目标处理器为 ARM，目标平台为 rv1106，构建类型为 Release，最终安装到 install/rv1106_armhf/ppocr_rv1106。原因是板端 ABI 和虚拟机主机 ABI 不同，必须交叉编译并把部署文件整理在一个目录中。

### 14.4 输入/输出内存管理改动

**src/ppocr_system/ppocr_system.h** 为 RKNN Context 增加 input_mem 和 output_mems 成员，保存零拷贝输入/输出内存句柄。原来的结构体只保存模型 Context 和普通 IO 属性，不足以管理 RV1106 所需的绑定内存。

**src/ppocr_system/rknpu2/ppocr_system.cc** 中，普通输入输出调用被替换为以下流程：

| 改动 | 原因 / 作用 |
|---|---|
| 输入属性查询改为 RKNN_QUERY_NATIVE_INPUT_ATTR | 按模型实际原生输入布局和 stride 分配内存，供 RV1106 的零拷贝 API 使用 |
| 输出属性查询改为 RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR | 获取与板端输出内存相匹配的布局、类型、stride 和量化信息 |
| 调用 rknn_create_mem 并用 rknn_set_io_mem 绑定输入及所有输出 | 修复普通 inputs_set/outputs_get 在板端返回 context config invalid 的问题 |
| 新增 copy_to_input()，按 w_stride/h_stride 逐行拷贝，并先清零整个目标 buffer | 模型内存行跨度可能比有效图像行宽大；按真实 stride 复制可避免行错位或越界，也能清理 padding 区域 |
| Det/Rec 推理前都把预处理图像写入绑定 buffer，再调用推理 | 对齐 RV1106 的输入内存绑定机制，不再为普通 rknn_input 临时包装图像指针 |
| Context 销毁时逐一释放 output_mems 和 input_mem | 避免模型初始化失败或程序退出时遗留 RKNN 内存 |

### 14.5 输出张量和识别后处理改动

同一个程序要兼容 RKNN 输出的实际数据类型。工程增加 fp16_to_float() 和 get_output_float()：

- FLOAT32 输出直接复制为 float。
- FLOAT16 输出逐元素转换为 float，供原有后处理使用。
- INT8/UINT8 输出依据模型属性中的量化类型、scale、zero point 或 fractional length 还原数值。
- 不支持的数据类型给出错误并停止该次推理，而不是把未知字节按 float 解释。

Det 原来通过 rknn_outputs_get(... want_float=1) 取得浮点输出，改为读取绑定的 native 输出并用 get_output_float() 还原，之后仍进入原有 DBNet 文字区域后处理。这样保留检测后处理逻辑，同时适配零拷贝和量化输出。

Rec 的输入预处理也改了：原示例把图像转成 float，再在 C++ 中执行 (pixel - 127.5) / 127.5；本工程的 RKNN 转换已经把 mean/std 写入模型配置，因此板端改为保留 uint8 像素，避免应用层和模型侧重复归一化。Rec 的宽度 padding 从浮点 0 改为 uint8 128，因为 128 接近该 mean/std 约定下的零中心像素。图像最后通过 copy_to_input() 写入绑定内存。

Rec 原来从 want_float 输出直接交给 CTC 后处理；工程现在先转换 native 输出类型，再判断每个时间步的字符分数是否已是概率分布。若行最大值高于 1.0001，或行和不在 0.99–1.01 范围内，就做行归一化；如果发现负数，则先做减去最大值的 softmax，再归一化，如果没有负数，则除以该行分数总和。最后仍调用原有 rec_postprocess() 做贪心 CTC 解码和字典映射。

这段行归一化逻辑是为了解决 RV1106 Rec 输出进入后处理时置信度超过 1 的现象。它是按当前模型输出观察实现的兼容处理，业务图片仍要检查分数和文本质量；如果以后更换模型输出语义，必须重新验证这一判断条件，不能盲目沿用。

### 14.6 哪些上游逻辑保留了

检测框的 DBNet 后处理、文本区域裁剪/旋转、Rec 的 CTC 贪心解码和字典映射沿用上游 PP-OCR-System 代码。本次没有为了适配 RV1106 重写模型算法；主要调整集中在模型转换参数、构建/安装路径、NPU IO 内存接口、输入格式和输出张量解释。这使变化更集中，也方便将来逐项排查。

### 14.7 部署脚本的改动

**scripts/deploy_and_run.sh** 把部署过程固定为：检查可执行文件和 ADB 状态、清理本工程专用的板端临时目录、推送安装目录、可选推送自定义图片、设置运行库搜索路径、执行 demo、保存运行日志并拉回 out.jpg。部署目的路径固定为 /tmp/ppocr_rv1106，因为板端 /userdata 空间不够容纳整套约 13 MB 安装内容，而 /tmp 当前可用空间足够。

脚本在普通用户下使用 sudo adb，在 root 下使用 adb；这样可以适配 Ubuntu 主机的 ADB 权限配置。每次脚本只清理 /tmp/ppocr_rv1106 这个工程目录，避免遗留旧模型与新程序混用。

## 15. 参考来源

- PaddleOCR release/2.7：https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7
- Rockchip RKNN Model Zoo PP-OCR 示例：https://github.com/airockchip/rknn_model_zoo/tree/main/examples/PPOCR
- Luckfox RV1106 交叉编译指南：https://wiki.luckfox.com/Luckfox-Pico-RV1106/Cross-Compile/

模型、源码、工具链和运行库的许可条件应以各自仓库和随附 LICENSE 为准。将工程用于产品时，请单独核实对应模型和第三方组件的许可要求。

---

**复现入口：** 在 Ubuntu 虚拟机执行 `cd /home/lubancat/PPOCR_RV1106 && scripts/build_rv1106.sh && scripts/deploy_and_run.sh`。首次运行成功后，查看 `output/rv1106_test_result.jpg` 和 `logs/rv1106_run.log`。

## 16. PP-OCRv4 的算子兼容性实测

这一章不是根据模型名称猜兼容性，而是读取工程里的原始 ONNX 和 `logs/convert_rec_i8.log`。ONNX 原始节点会被 RKNN 做常量折叠、删除、改写和融合，最终 Network Layer Information Table 才是板端执行分配的直接编译证据。

### 16.1 PP-OCRv4 Det 原始 ONNX

文件：`models/onnx/ppocrv4_det.onnx`。实际结构：IR version 8，ONNX opset 12；输入 `x` 是 float32 `[1,3,480,480]`，输出 `sigmoid_0.tmp_0` 是 float32 `[1,1,480,480]`。共有 672 个原始节点、14 种算子：

| ONNX 算子 | 数量 | 说明 |
|---|---:|---|
| Constant | 342 | shape、权重和导出辅助常量；多数在编译时折叠 |
| Add | 89 | 偏置、残差或导出表达式 |
| Mul | 86 | 激活、缩放及导出表达式 |
| Conv | 62 | 检测主干、neck 和 head |
| Clip | 24 | 激活限制，常参与 HardSwish/HardSigmoid 展开 |
| Div | 24 | 激活/缩放表达式 |
| Relu | 12 | 激活 |
| GlobalAveragePool | 10 | 通道注意力 |
| HardSigmoid | 10 | 门控激活 |
| Resize | 6 | 多尺度上采样 |
| BatchNormalization | 3 | 归一化，可和 Conv 融合 |
| ConvTranspose | 2 | DB 输出上采样 |
| Concat | 1 | 特征拼接 |
| Sigmoid | 1 | DB 概率输出 |

最初的 `convert_det_i8.log` 没有保存完整 Network Layer Information Table。2026-10-19 补充了独立脚本 `scripts/convert_det_rv1106_verbose.py`，保持与 Model Zoo 转换器相同的均值、标准差、RV1106 平台和 INT8 校准集，以 `RKNN(verbose=True)` 重新构建候选模型，完整日志保存为 `logs/convert_det_verbose.log`。候选文件写入 `models/rknn/candidates/`，没有覆盖已经上板验证的基线。

审计结果是 170 个编译后节点：1 个 CPU `InputOperator` 边界、169 个 NPU 节点，**内部 CPU fallback 为 0**。NPU 节点包括 Add 37、BatchNormalizationRelu 1、Concat 1、Conv 41、ConvHardSigmoid 10、ConvHardSwish 24、ConvRelu 11、ConvTranspose 2、Mul 34、Resize 6、Sigmoid 1 和 OutputOperator 1。编译器报告内部内存 9000 KB、权重内存 1423.38 KB。

日志还提示 9 个卷积权重存在 outlier，并对 6 组 Resize 空 initializer/12 个 Resize 常量默认 FP16 给出 18 条 warning；这些 warning 后面有完整设备表、临时 RKNN 导出和正式 `SUCCESS`。候选大小 2,118,529 bytes、SHA256 为 `bf7610a7084f3d111bee745bfd53e48756d9ff83b9740856f39860354810774e`；原基线大小 2,118,509 bytes、SHA256 为 `9fda1629...`。重新构建文件并非逐字节相同，因此它继续作为候选，不能只因为配置相同就替换基线；应在板端完成数值和任务回归后再晋级。补测时 ADB 未连接，所以本轮只确认编译与设备分配，不声称候选已完成板端回归。

### 16.2 PP-OCRv4 Rec 原始 ONNX

文件：`models/onnx/ppocrv4_rec.onnx`。IR version 8、opset 12；输入 `x` 是 float32 `[1,3,48,320]`。原始 output value_info 的维度为空，只有名字 `softmax_11.tmp_0`；RKNN shape 推导后实际输出为 `[1,40,6625]` FP16。C++ 不能因为 ONNX 公开 output 没写维度就猜长度，而应查询 Runtime 输出属性；若重新导出，最好在 ONNX 验证阶段显式推导输出 shape 并做数值对照。

对这个文件执行 `onnx.checker.check_model()` 会报 `Field 'shape' of 'type' is required but missing`。这条错误指向公开输出 `ValueInfo` 缺少 shape 元信息；它与“不支持某个计算算子”不是同一类问题。实际 RKNN 解析器随后完成 shape 推导、图融合和导出，Runtime 查询得到 `[1,40,6625]`，板端结果也可读，所以工程保留了该输入文件并在应用侧以 Runtime attr 为准。若要修补元信息，必须先用 ONNX Runtime/RKNN 对照确认输出 shape，再写入 `[1,40,6625]`；不能为了让 checker 变绿随意填写。审计脚本会把它显示为 `shape=<missing> dynamic=True`，同时保留 checker 原始错误。

原始图有 860 个节点、25 种算子：

| ONNX 算子 | 数量 | 兼容性关注点 |
|---|---:|---|
| Constant | 420 | 大量 shape/索引常量，静态化有助图优化 |
| Add | 107 | 残差、偏置、LayerNorm 后仿射 |
| Mul | 107 | 激活、LayerNorm、attention 等 |
| Conv | 38 | CNN 主干和投影 |
| Div | 33 | 激活/归一化 |
| Clip | 28 | 激活展开 |
| Cast | 23 | dtype 边界，过多 Cast 可能影响融合 |
| MatMul | 13 | attention 和输出映射；需关注 rank/transpose/常量权重 |
| Slice | 10 | attention Q/K/V 分支切分；动态参数会增加转换难度 |
| ReduceMean | 10 | LayerNorm 子图关键部分 |
| Transpose | 9 | CNN/序列/attention 维度重排 |
| Concat | 7 | 特征/shape 拼接 |
| Sigmoid | 7 | 激活/门控 |
| Squeeze | 7 | 去掉单位维度 |
| BatchNormalization | 6 | CNN 归一化 |
| Pow / Sqrt / Sub | 各 5 | 5 组 LayerNorm 的方差、开方、去均值 |
| Reshape | 6 | CNN 到序列、attention 维度整理 |
| Shape | 4 | 动态 shape 子图；固定输入后仍参与推导 |
| Softmax | 3 | attention 两处及最终 CTC 类别概率 |
| AveragePool | 1 | 高度压缩 |
| GlobalAveragePool | 2 | 通道特征 |
| HardSigmoid / Relu | 各 2 | 激活 |

相比 v6 Rec，v4 识别图包含显式 attention、5 组 LayerNorm 数学子图、更多 MatMul/Transpose/Slice/Shape，图更大，算子兼容和 NPU/CPU 分区更复杂。

### 16.3 RKNN 编译后 v4 Rec 的真实设备分配

`logs/convert_rec_i8.log` 的 Network Layer Information Table 有 132 个编译后节点：

- NPU：125 个。
- CPU：7 个，其中 `InputOperator` 和 `OutputOperator` 是边界，5 个 `exNorm` 是内部 CPU fallback。
- 内部 CPU 节点 ID：101、108、111、118、121。
- 名称分别是 `exNorm:p2o.ReduceMean.0_2ln`、`.4_2ln`、`.8_2ln`、`.12_2ln`、`.16_2ln`。
- attention 被编译为 NPU 的 `Split`、`exSDPAttention`、Conv/MatMul 融合节点；最终 Softmax 是 NPU `exSoftmax13`，输出再经 NPU Transpose/Reshape，最后 CPU `OutputOperator` 交给应用。

因此准确描述是：**v4 Rec 主体由 NPU 执行，5 个 LayerNorm 融合节点由 ARM CPU 执行。**经过这些节点可能发生 NPU→CPU→NPU 同步/搬运；由于张量已经压到高度 1、序列长度 40，实际代价要靠计时确定，不能只按节点数推断。

### 16.4 `Unkown op target: 0` 为什么没有直接判定构建失败

转换日志连续出现 5 条：

~~~text
E RKNN: Unkown op target: 0
~~~

拼写 `Unkown` 来自 Toolkit2。它们位于 ProfileAnalysis/设备规划附近，数量与最终表中的 5 个 CPU `exNorm` 一致。随后日志继续完成 RKNNModelBuildPass、设备表、Flatc model build、临时 RKNN 导出、正式 `export_rknn`，板端也完成 `rknn_init/rknn_run` 和 16 行 OCR。

所以这里将它解释为编译器处理 fallback 节点时的 error-level 诊断，而非最终构建失败。依据不是“忽略 E”，而是后续 API 返回、设备表、导出文件和板端运行均成功。若相同信息后没有 build/export 成功，或 `rknn.build()` 返回非 0，就必须判失败。

### 16.5 qtype warning 怎样处理

日志对 `Slice.*_2sp_split`、Softmax 输出、Reshape/Transpose 中间张量提示没有显式 qtype，默认 FP16。这些位置靠近 attention 分支和最终 FP16 Softmax 边界。工程没有强制全图 Cast INT8，原因是：

- 编译表已明确展示主要 CNN/attention 节点的 dtype/target；
- 最终输出本来就是 FP16；
- 增加 Cast 可能破坏融合并放大量化误差；
- 板端 C++ 已按 Runtime `attr.type` 读取 FP16。

警告被接受的条件是构建成功、输出 shape 正确、板端结果可读。若未来输出异常，要将警告对应的张量加入分层数值对比，不能继续无条件接受。

## 17. PP-OCRv4 实际问题复盘：为什么这样修

### 17.1 Rec 非量化构建失败

**尝试**：执行 `do_quantization=False`。  
**日志**：先提示 mean/std 未配置时会用 0/1，随后明确报 `Current target_platform(rv1106) not support do_quantization = False!` 和 `Build model failed!`。  
**修复**：Rec 脚本显式设置 rv1106、mean/std 127.5、`do_quantization=True` 和校准清单。  
**为什么**：目标平台明确拒绝该 FP build，不能通过忽略返回值生成伪成功；INT8 是当前能生成 RV1106 模型的路径。

### 17.2 vendor Rec 转换器没有完整覆盖本工程配置

**现象**：上游默认行为没有直接覆盖 RV1106 Rec 的平台、归一化、量化和项目路径。  
**修复**：新增 `scripts/convert_rec_rv1106.py`，参数化 model/dataset/output，逐步检查 config/load/build/export 返回值。  
**为什么**：Det 可复用上游转换器；Rec 的关键假设要固化在工程里，避免换终端后凭记忆执行不同命令。

### 17.3 普通 IO 报 `context config invalid`

**现象**：模型已转换，板端沿用 `rknn_inputs_set/rknn_outputs_get` 时上下文配置失败。  
**定位**：模型能初始化并查询属性，问题在当前 RV1106 Runtime 的 native IO 契约，而不是 OCR 阈值或 ONNX 加载。  
**修复**：查询 `RKNN_QUERY_NATIVE_INPUT_ATTR` 和 `RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR`；用 `rknn_create_mem` 分配，用 `rknn_set_io_mem` 绑定输入和所有输出。  
**为什么**：这条路径遵循 Runtime 给出的 native layout、type 和 stride，并已在实板通过。

### 17.4 为什么不能直接连续 memcpy 输入

native buffer 可能有 `w_stride/h_stride` 对齐，行跨度大于有效宽度。工程的 `copy_to_input()` 先清零整个 input buffer，再以 `dst_stride=w_stride*channels` 逐行复制 `row_bytes=model_width*channels`。若忽略 stride，下一行可能错位，padding 也会残留上次输入。

### 17.5 Rec 输入重复归一化风险

原示例在 C++ 做 `(x-127.5)/127.5`；本工程 RKNN config 又写入相同 mean/std。两边都做会重复归一化。修复是 C++ 保留 uint8 resize、不足宽度填 128、归一化由编译模型完成。128 在该公式下接近 0；填 uint8 0 对应约 -1，不是中性 padding。

### 17.6 Rec FP16 输出不能按 float32/INT8 猜

Runtime 实测输出 `[1,40,6625]` FP16、530000 bytes。工程新增 `fp16_to_float()` 和 `get_output_float()`：FLOAT32 复制、FLOAT16 转换、INT8/UINT8 按 qnt_type/scale/zp/fl 反量化，未知类型报错。文件名 `_i8.rknn` 说明主要量化路径，不保证输出 tensor 也是 INT8。

### 17.7 识别置信度曾超过 1

**现象**：native Rec 行数据直接进入旧后处理时，分数不在概率范围。  
**定位**：检查每时间步 6625 类的最小/最大与行和。  
**修复**：若有负值，做数值稳定 softmax（先减行最大）；若无负值但行和偏离 1，则除以行和；已是概率的行不重复 softmax。  
**为什么**：最终模型本身有 Softmax，对概率再次 softmax 会压平分布。条件归一化解决实际输出契约差异，但换模型时必须重新验证。

### 17.8 v4 字典和 6625 类必须配套

v4 输出 6625 类；v6 输出 6906 类。输出最后一维、字典条数、CTC blank、末尾空格必须逐项对应。混用时 argmax 仍可能落在合法整数范围，程序不崩，但字符会系统性错位，因此它比直接报错更危险。

### 17.9 CMake 原相对路径失效

复制上游源码到独立 `src/ppocr_system` 后，原 `../../../../3rdparty`、`utils`、OpenCV 路径不成立。CMake 改为解析工程根并定义 `MZ_ROOT=vendor/rknn_model_zoo`，安装规则指向工程自己的 model/data。这样保留 vendor 原件，同时让本地源码独立维护。

### 17.10 必须使用 ARMHF/uClibc 工具链

板端是 armv7l、Buildroot/uClibc，主机是 x86_64/glibc。工程使用 `arm-rockchip830-linux-uclibcgnueabihf` GCC 8.3.0，并链接 armhf OpenCV/RKNN/RGA。主机 GCC 生成的指令集错误；另一套 ARM glibc 工具链也可能缺 loader/libc 符号。

### 17.11 `/userdata` 容量不足

板端 `/userdata` 当时约剩 1.4 MB，安装包约 13 MB；推送可能中断或截断。改部署到约 90 MB 可用的 `/tmp/ppocr_rv1106`。代价是重启后需重推，主副本始终保存在 VM。

### 17.12 为什么保留 vendor 原始源码

修改放在 `src/ppocr_system`，vendor 作为上游对照。可以用 `diff -u` 精确查看零拷贝、dtype、归一化和 CMake 变化；升级 Model Zoo 时也不会把本地修复和上游混在一起。

## 18. PP-OCRv4 与 PP-OCRv6 的关键差异

| 项目 | PP-OCRv4 工程 | PP-OCRv6 Tiny 工程 | 不能混用的原因 |
|---|---|---|---|
| 目录 | `/home/lubancat/PPOCR_RV1106` | `/home/lubancat/PPOCR_RV1106_latest` | 保留历史并防止串档 |
| 来源 | PaddleOCR release/2.7 系列 v4 ONNX | PaddleOCR 3.7.0 v6 Tiny bundle | 模型结构和配置不同 |
| Det 输入 | 480×480 | 当前 tuned 704×704 | 坐标、速度和召回不同 |
| Rec 输入 | 48×320 | 当前 tuned 48×320 | 尺寸相同不代表网络/字典相同 |
| Rec 类别 | 6625 | 6906 | 字典索引不兼容 |
| Rec 原始图 | 860 节点、25 op types、含 attention/5 组 LN | 459 节点、18 op types | v4 兼容压力更高 |
| Det 设备 | verbose 重建候选：169 NPU + 输入边界，内部 fallback 0 | tuned det704：154 NPU + 输入边界，内部 fallback 0 | 两者输入尺寸和融合图不同，节点数不能直接比较速度 |
| Rec 设备 | 125 NPU + 5 内部 CPU exNorm + IO 边界 | 62 内部 NPU + IO 边界，内部 fallback 0 | v4 Rec 是混合执行，v6 Rec 内部全 NPU |
| Rec 输出 | `[1,40,6625]` FP16 | `[1,40,6906]` FP16 | 类别维不同 |
| 校准 | 初始 det/rec 共用 20 张整页 | det 整页，rec 128 单行 crop | rec 校准分布不同 |
| 评测 | 功能冒烟，未做正式精度/性能集 | CTW100 调参、400 独立复核、500 全集 | 指标不能直接横比 |

## 19. v4 遇到新不兼容算子的处理流程

### 19.1 先保存证据

保存原 ONNX、SHA、Toolkit2 版本、转换命令、完整 verbose 日志和失败节点名。新候选输出到单独目录，禁止直接覆盖已经能上板的模型。

### 19.2 判断“失败”还是“fallback”

- `load_onnx/build` 返回非 0且没有导出：转换失败，必须改导出、图或版本。
- build 成功且中间节点 Target=CPU：可运行的 fallback；报告为混合执行并测代价。
- 只有 Input/OutputOperator=CPU：通常是 IO 边界，不算内部 fallback。
- 日志含 E 但后面有完整设备表、export 和板端运行：定位对应节点后判断；本工程 5 条 `Unkown op target` 与 5 个 CPU exNorm 是实例。

### 19.3 修复优先级

1. 从 Paddle 导出端选择适配的 opset和静态 shape，保持原语义。
2. 用 ONNX checker、shape inference、ONNX Runtime 先验证浮点图。
3. 动态 Shape/Slice/Resize 先固定 shape/参数。
4. LayerNorm 等复合模式比较“保留小 fallback”和“数学等价展开”；展开必须保持 axis、epsilon、broadcast，并重做量化精度。
5. NMS/CTC/DB 等后处理可放 C++，但必须完整复现，不能删除主干节点。
6. 自定义算子或换网络是最后方案，涉及 Runtime 插件或重新训练。

### 19.4 改图后的必测项

- 修改前后 ONNX 多输入逐元素误差、top1/CER/框结果。
- RKNN 表里的 CPU/NPU 节点是否真的改善；拆算子可能让更多节点 fallback。
- 量化后文本/框和分数；浮点等价不保证 INT8 等价。
- 板端 `rknn_run` 稳定性和 Runtime attr。
- 端到端耗时，而非只看 NPU 节点数。

## 20. ONNX 与 RKNN 日志检查工具

工程新增只读脚本 `scripts/inspect_onnx_rknn.py`：

~~~bash
cd /home/lubancat/PPOCR_RV1106
conda activate ppocr_latest
python scripts/inspect_onnx_rknn.py \
  --model models/onnx/ppocrv4_det.onnx \
  --model models/onnx/ppocrv4_rec.onnx \
  --log logs/convert_det_verbose.log \
  --log logs/convert_rec_i8.log \
  --show-nodes Shape Slice ReduceMean Pow Sqrt MatMul Softmax Resize
~~~

脚本输出 checker、opset、IO shape/dtype、动态/未知维度、原始 op 统计和指定节点；解析 RKNN 表的 CPU/NPU 数量、内部 CPU fallback、warning/error-looking 行及后续 export/success 证据。`convert_det_verbose.log` 已补齐 Det 设备表；保留短日志时，工具仍会明确提示“不能从 build success 推断全 NPU”。脚本需要 `onnx`，VM 的 `ppocr_latest` 可直接使用；不要为了只读审计升级稳定的 Toolkit2 环境。

## 21. v4 学习与复现实操检查单

1. `sha256sum -c models/SHA256SUMS`：确认没有拿错模型。
2. 运行第 20 章脚本：看原始图、未知 output shape、5 个 exNorm fallback。
3. 阅读 `logs/convert_rec_fp.log`：理解目标不支持非量化构建时为何改 INT8。
4. 阅读 `logs/convert_rec_i8.log` 的设备表：从 ONNX 860 节点看到 RKNN 132 节点和融合。
5. 对比 vendor 与 src 的 `rknpu2/ppocr_system.cc`：理解普通 IO 到 native 零拷贝、FP16 解码和条件归一化。
6. 运行 `scripts/build_rv1106.sh`：观察 ARMHF/uClibc 工具链、OpenCV 和安装目录。
7. 运行 `scripts/deploy_and_run.sh`：确认 ADB、`LD_LIBRARY_PATH`、`/tmp` 和结果回拉。
8. 在日志核对 det `[1,480,480,3] INT8`、rec `[1,48,320,3] INT8`、rec output `[1,40,6625] FP16`。
9. 先看检测框再判断识别错字；框切错时继续调 rec 不会解决根因。
10. 更换模型时核对类别数、字典、颜色、归一化、padding、输出 axis、Runtime dtype 和 CPU fallback，全部通过后再更新 SHA 清单。

## 22. 全量补充技术手册

同目录新增 `PPOCR_RV1106_全量补充技术手册.md`，集中补充两代模型共用的底层内容：环境冻结、INT8 量化数学、输入输出契约、DB/CTC、算子改图实例、RKNN 日志判读、native 内存与 stride、精度/性能测试、错误字典、模型升级、发布和回滚。该总册也记录了 2026-10-19 新增的 v4 Det verbose 转换证据：169 个 NPU 节点、仅输入边界在 CPU、内部 fallback 0，以及候选未完成板端回归所以没有覆盖基线的原因。

## 23. 官方代码与本工程改动的完整对应关系

> 本章补充日期：2026-10-29。目的不是重复罗列文件，而是回答学习时最容易混淆的问题：什么是 PaddleOCR 官方内容，什么是 Rockchip 官方示例，哪些文件保持原样，哪些文件复制后修改，哪些文件是本工程新增，以及每项变化为什么必要。

### 23.1 先区分三层来源

本工程同时使用了三个层次的内容。如果把它们都称作“官方代码”，就很难理解修改边界。

| 层次 | 来源 | 提供的主要内容 | 本工程怎样使用 |
|---|---|---|---|
| PaddleOCR 官方 | PaddlePaddle/PaddleOCR `release/2.7` | PP-OCRv4 模型结构、训练权重、推理配置、字典、DB/CTC 算法定义 | 作为 OCR 模型和算法来源；没有重新训练或任意删层 |
| Rockchip 官方 | airockchip/rknn_model_zoo PPOCR 示例、RKNN Toolkit2/Runtime | ONNX→RKNN 示例、PPOCR C++ Demo、OpenCV/RGA/RKNN 依赖、校准示例 | 原始副本保存在 `vendor/`；Det 转换流程继续复用，C++ 示例复制到独立目录后适配 RV1106 |
| 本工程适配层 | `/home/lubancat/PPOCR_RV1106` | RV1106 Rec 转换、native IO、stride/dtype、ARMHF 编译、ADB 部署、审计和文档 | 解决上游示例没有直接覆盖的 RV1106 转换与运行问题 |

这三层的关系是：PaddleOCR 定义“模型算什么”，RKNN 工具定义“怎样编译和执行”，本工程负责“让这套模型在当前 RV1106 板子上按正确输入输出契约运行”。

### 23.2 文件状态总表

下面用四种状态描述文件，避免把“调用参数变化”和“修改上游源码”混为一谈。

| 状态 | 含义 |
|---|---|
| 保留原件 | 文件留在 `vendor/` 中作为上游对照，没有把本工程改动直接写进去 |
| 直接复用 | 运行时直接调用上游文件，只改变命令行参数或输入路径 |
| 复制后修改 | 从上游示例复制到 `src/`，再做 RV1106 适配；vendor 原件仍可比较 |
| 本工程新增 | 上游没有这个工程入口，由本工程新建 |

| 文件/内容 | 状态 | 是否改变模型数学语义 | 说明 |
|---|---|---:|---|
| PaddleOCR PP-OCRv4 Det/Rec 权重 | 保留官方模型 | 否 | 没有重训、剪枝或手工改权重 |
| `models/onnx/ppocrv4_det.onnx` | 官方模型转换产物 | 否 | 用作 RKNN Det 输入，没有手工删除算子 |
| `models/onnx/ppocrv4_rec.onnx` | 官方模型转换产物 | 否 | 保留原图，包括 5 组 LayerNorm 数学子图和缺失的公开 output shape 元信息 |
| `vendor/rknn_model_zoo/.../PPOCR-Det/python/convert.py` | 保留原件、直接复用 | 否 | Det 基线调用该脚本，传入 `rv1106 i8` 和工程输出路径 |
| vendor 中 PPOCR-System C++ | 保留原件 | 否 | 作为 diff 对照，不直接承载本工程 RV1106 修改 |
| `scripts/convert_models.sh` | 本工程新增 | 否 | 固定 Det/Rec 转换顺序、环境、路径和失败行为 |
| `scripts/convert_rec_rv1106.py` | 本工程新增 | 否 | 明确 RV1106、Rec mean/std、INT8 校准和返回值检查 |
| `scripts/convert_det_rv1106_verbose.py` | 本工程新增 | 否 | 保持 Det 配置不变，只增加完整编译日志和候选隔离 |
| `src/ppocr_system/rknpu2/ppocr_system.cc` | 复制后修改 | 不改变网络；改变 Runtime 适配 | native 零拷贝、stride、dtype 解码、Rec 输入和概率处理 |
| `src/ppocr_system/ppocr_system.h` | 复制后修改 | 否 | Context 增加输入/输出内存句柄和 native 属性所需状态 |
| `src/ppocr_system/postprocess.cc` | 以上游算法为基础 | DB/CTC主体不变 | 保留 DB 和 CTC 主逻辑；输入数据改由 native 输出转换后提供 |
| `src/ppocr_system/CMakeLists.txt` | 复制后修改 | 否 | 修复独立目录后的依赖路径、模型安装和 RV1106 构建 |
| `scripts/build_rv1106.sh` | 本工程新增 | 否 | 固定 ARMHF/uClibc 工具链和安装目录 |
| `scripts/deploy_and_run.sh` | 本工程新增 | 否 | 固定 ADB 推送、库路径、执行、日志和结果回拉 |
| `scripts/inspect_onnx_rknn.py` | 本工程新增 | 只读 | 审计 ONNX 和 RKNN 日志，不修改模型 |
| `scripts/verify_delivery.py` | 本工程新增 | 只读 | 校验哈希、文件、ONNX、日志、文档和 ADB 状态 |

### 23.3 模型层到底有没有改

#### 23.3.1 没有做的事情

本工程没有执行以下模型结构改动：

- 没有删除 Conv、MatMul、Softmax、LayerNorm 或 attention。
- 没有为了让日志显示全 NPU 而跳过计算节点。
- 没有修改 Det/Rec 网络通道数。
- 没有更换 v4 Rec 的 6625 类输出定义。
- 没有把 v6 的 6906 字典套到 v4。
- 没有重新训练、微调、剪枝或蒸馏 v4 权重。
- 没有修改 DB/CTC 的模型训练目标。

#### 23.3.2 RKNN 编译器自动做的变化

ONNX 节点进入 Toolkit2 后会自动发生常量折叠、算子改写和融合。例如：

- Conv、Add、激活可能融合成 `ConvAdd`、`ConvRelu`、`ConvHardSwish`。
- GlobalAveragePool 可能被编译器改写成适合 NPU 的卷积模式。
- v4 Rec 的 LayerNorm 数学子图被识别为 `exNorm`。
- attention 片段可能融合为 `exSDPAttention`。
- Softmax 可能显示为 `exSoftmax13`。

这些是 RKNN 编译器生成执行图，不是手工删除 ONNX 节点。判断板端算子分配应看 Network Layer Information Table，不能只看原始 ONNX op 名单。

#### 23.3.3 v4 Rec 的 5 个 CPU `exNorm`

v4 Rec 原始 ONNX 中有 5 组：

~~~text
ReduceMean → Sub → Pow → ReduceMean → Add(epsilon)
→ Sqrt → Div → Mul(gamma) → Add(beta)
~~~

Toolkit2 把它们融合为 5 个 `exNorm`，最终分配到 CPU。工程保留这个结果并明确报告为混合执行，没有修改数学公式来伪造全 NPU。原因是随意展开、替换或删除 LayerNorm 可能改变 axis、epsilon、广播和量化误差。

### 23.4 Det 转换：官方脚本保留了什么，工程只改变了什么

#### 23.4.1 直接复用的官方文件

Det 基线调用：

~~~text
vendor/rknn_model_zoo/examples/PPOCR/PPOCR-Det/python/convert.py
~~~

该文件保留在 vendor 目录，没有为了 RV1106 直接编辑。它仍负责：

1. 创建 RKNN 对象。
2. 设置 Det mean/std。
3. 加载 Det ONNX。
4. 使用校准清单构建量化模型。
5. 导出 RKNN。
6. 释放 RKNN 对象。

#### 23.4.2 本工程传入的参数

实际调用逻辑为：

~~~bash
python convert.py \
  /home/lubancat/PPOCR_RV1106/models/onnx/ppocrv4_det.onnx \
  rv1106 \
  i8 \
  /home/lubancat/PPOCR_RV1106/models/rknn/ppocrv4_det_rv1106_i8.rknn
~~~

这里发生的是参数适配：

| 参数 | 含义 | 为什么明确指定 |
|---|---|---|
| Det ONNX 绝对路径 | 转换输入 | 避免当前目录变化后拿错模型 |
| `rv1106` | RKNN 目标平台 | 编译器需要按 RV1106 能力生成执行图 |
| `i8` | 开启 INT8 PTQ | 当前目标平台实际构建路径要求量化 |
| 工程输出路径 | RKNN 产物 | 统一模型管理并供 CMake/部署脚本引用 |

Det mean/std 继续使用上游转换器的：

~~~python
mean_values = [[123.675, 116.28, 103.53]]
std_values = [[58.395, 57.12, 57.375]]
~~~

这组值属于 Det 输入契约，不能因为 Rec 使用 127.5/127.5 就替换。

#### 23.4.3 为什么又新增 verbose Det 脚本

上游 Det 转换器使用的日志详细程度不足以保存完整设备分配证据。因此新增 `scripts/convert_det_rv1106_verbose.py`，但保持：

- 同一个 Det ONNX。
- 同一组 mean/std。
- 同一个 `rv1106` target。
- 同一类 INT8 校准清单。
- 同一量化目的。

它只增加：`RKNN(verbose=True)`、每阶段返回值检查、候选目录和成功文件信息。新候选不会自动覆盖板端基线。补充日志证明候选编译图为 169 个 NPU 节点、1 个 CPU 输入边界、内部 CPU fallback 0。

### 23.5 Rec 转换：为什么没有照搬 Det 转换器

Rec 与 Det 的输入分布和输出语义不同。不能把 Det 的 mean/std、尺寸和后处理直接用于 Rec。

#### 23.5.1 新增的专用脚本

文件：

~~~text
scripts/convert_rec_rv1106.py
~~~

核心调用顺序：

~~~python
rknn = RKNN(verbose=True)

ret = rknn.config(
    target_platform="rv1106",
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[127.5, 127.5, 127.5]],
)

ret = rknn.load_onnx(model=args.model)
ret = rknn.build(do_quantization=True, dataset=args.dataset)
ret = rknn.export_rknn(args.output)
rknn.release()
~~~

#### 23.5.2 每一项变化的原因

| 变化 | 技术原因 | 若不这样做可能出现什么 |
|---|---|---|
| `target_platform='rv1106'` | 让 Toolkit2 选择 RV1106 可用的量化、融合和设备实现 | 生成错误平台模型或在板端 `rknn_init` 失败 |
| Rec mean/std 127.5/127.5 | 对应识别输入的 `[-1,1]` 归一化约定 | 字符特征整体偏移，置信度和识别率下降 |
| `do_quantization=True` | RV1106 实测拒绝本工程 Rec 非量化构建 | build 返回非 0，没有有效 RKNN |
| 传入 calibration dataset | 估计 Rec 激活量化范围 | 无法完成 PTQ，或用错分布造成量化精度下降 |
| `verbose=True` | 保存融合、warning、CPU/NPU 分配 | 只知道有文件，不知道内部是否 fallback |
| 检查每个返回值 | 防止失败后继续执行 | 旧 RKNN 可能被误认为新产物 |
| 参数化 model/dataset/output | 候选可隔离、可复现 | 路径硬编码导致覆盖基线或拿错文件 |

#### 23.5.3 为什么 Rec 校准图应是文字行

Rec 输入是固定高度的单行文字，激活分布来自字符笔画、右侧 padding、短行和长行。如果使用整页图片直接校准 Rec，图像内容分布与真实输入不一致。v4 初始工程使用 Model Zoo 20 张集合完成可运行验证；进一步做业务精度时，应使用真实 Det crop 或人工单行图建立 Rec 专用校准集。

### 23.6 `convert_models.sh` 为什么新增

官方示例可以分别手工执行 Det 和 Rec，但手工操作容易出现：环境不同、路径不同、只转了一个模型、失败后继续、覆盖错误文件。因此新增统一入口：

~~~bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate toolkit2
~~~

随后按固定顺序转换 Det 和 Rec。三个 shell 选项分别表示：

- `-e`：某条命令失败后停止。
- `-u`：使用未定义变量时停止。
- `pipefail`：管道中前面命令失败也返回失败。

统一入口没有改变模型计算，只是把转换过程变成可重复、可失败停止的工程流程。

### 23.7 C++ Runtime：官方通用 IO 为什么要改

#### 23.7.1 上游通用路径

许多 RKNN 示例使用：

~~~cpp
rknn_inputs_set(ctx, input_num, inputs);
rknn_run(ctx, nullptr);
rknn_outputs_get(ctx, output_num, outputs, nullptr);
rknn_outputs_release(ctx, output_num, outputs);
~~~

当前 RV1106 实板上，这条路径曾出现 `context config invalid`。模型能初始化并查询属性，错误发生在普通 IO 配置阶段，因此定位到 Runtime IO 契约，而不是 DB 阈值或字典。

#### 23.7.2 本工程 native 零拷贝路径

改为：

~~~cpp
rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, ...);
rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, ...);

input_mem = rknn_create_mem(ctx, native_input_attr.size_with_stride);
output_mem = rknn_create_mem(ctx, native_output_attr.size_with_stride);

rknn_set_io_mem(ctx, input_mem, &native_input_attr);
rknn_set_io_mem(ctx, output_mem, &native_output_attr);
rknn_run(ctx, nullptr);
~~~

这项改动改变的是“数据怎样交给 Runtime”，没有改变模型网络结构。

### 23.8 `copy_to_input()`：为什么不能一次 `memcpy`

native tensor 可能存在对齐：

~~~text
有效宽度 width = 320
native w_stride 可能大于 320
有效行字节 = width × channels
目标行跨度 = w_stride × channels
~~~

若使用：

~~~cpp
memcpy(dst, src, width * height * channels);
~~~

第二行会紧跟第一行有效数据写入，而 Runtime 按 native stride 读取，导致行错位。工程新增 `copy_to_input()`：

1. 检查源行跨度和目标容量。
2. 清零整个 input memory，包括 padding。
3. 对每一行分别计算源地址和目标地址。
4. 每行只复制有效像素字节。
5. 保留目标行末对齐区域为 0。

这也是为什么输入 buffer 不能只按 `width*height*channels` 分配，而要使用 native attr 的 `size_with_stride`。

### 23.9 `get_output_float()`：为什么输出不能按文件名猜

v4 Det 的 RKNN 是 INT8 构建，Runtime 输出为 INT8；v4 Rec 同样是 INT8 构建，但 Runtime 输出为 FP16。因此文件名 `_i8.rknn` 只说明主要量化构建路径，不表示每个输出都是 INT8。

工程增加统一读取函数：

| Runtime dtype | 转换方式 |
|---|---|
| FLOAT32 | 直接复制 float |
| FLOAT16 | half 位模式转换为 float |
| INT8/UINT8 + affine | `(q - zero_point) * scale` |
| DFP | 根据 fractional length 恢复 |
| 未知类型 | 返回错误，不把字节强行解释为 float |

Det 实际输出属性示例：

~~~text
[1,480,480,1], INT8, zp=-128, scale=0.003922
~~~

Rec 实际输出属性示例：

~~~text
[1,40,6625], FP16, 530000 bytes
~~~

### 23.10 Rec 输入为什么从 float 改为 uint8

上游示例可能在 C++ 中执行：

~~~text
(pixel - 127.5) / 127.5
~~~

本工程又在 `rknn.config()` 中写入同一 mean/std。如果两边都做，输入会被重复归一化。修复方法是：

- C++ 保留 uint8 resize 结果。
- 由 RKNN 模型配置执行 mean/std。
- 右侧 padding 填 128。

为什么 padding 是 128：

~~~text
(128 - 127.5) / 127.5 ≈ 0.0039
~~~

它接近归一化后的中性值；uint8 0 会映射到 -1，更像强黑色信号。

### 23.11 Rec 概率处理为什么增加条件判断

板端曾观察到 Rec 分数超过 1。首先要区分输出是 logits、未精确归一化的非负分数，还是已经过 Softmax 的概率。

工程处理逻辑：

1. 每个时间步检查最小值、最大值和行和。
2. 如果存在负数，按 logits 做稳定 Softmax：先减行最大值，再 `exp` 和归一化。
3. 如果全部非负但行和明显不为 1，只按行和归一化。
4. 如果已经位于 `[0,1]` 且行和接近 1，直接交给 CTC，不再 Softmax。

不能无条件再次 Softmax，因为模型最终图已经包含 Softmax 时，二次 Softmax 会把高低概率差距压小。

### 23.12 哪些官方后处理逻辑保留

以下主逻辑继续沿用上游 PPOCR-System 思路：

- Det 概率图阈值化。
- 轮廓查找。
- mini box 和 box score。
- unclip 扩框。
- 四点顺时针排序。
- 透视裁剪和竖排旋转。
- Rec 每时间步 argmax。
- CTC blank 删除和相邻重复类别合并。
- 字典索引映射。

本工程主要改变这些算法所接收的数据来源：普通 outputs_get 的 float，改为 native output 按真实 dtype 解码后的 float。算法主体没有因为 RV1106 被替换成另一套 OCR 算法。

### 23.13 CMake、交叉编译和部署为何必须改

#### CMake

复制源码到 `src/ppocr_system` 后，上游示例原来的多层相对路径不再成立。本工程让 CMake 明确定位：

- `vendor/rknn_model_zoo/3rdparty`。
- Model Zoo utils。
- ARMHF OpenCV。
- RV1106 RKNN Runtime。
- RGA。
- 工程自己的 RKNN 和测试图。

#### 交叉编译

虚拟机是 x86_64/glibc，板端是 armv7l/Buildroot/uClibc。新增 `build_rv1106.sh` 固定使用：

~~~text
arm-rockchip830-linux-uclibcgnueabihf
~~~

主机 GCC、AArch64 工具链或 ARM glibc 工具链即使能生成文件，也可能因指令集、动态加载器或 libc ABI 不匹配而无法启动。

#### 部署

新增 `deploy_and_run.sh` 固定：检查 ADB、推送完整安装目录、设置 `LD_LIBRARY_PATH`、运行、保存日志、回拉结果图。部署到 `/tmp/ppocr_rv1106` 是因为当时 `/userdata` 剩余空间不足；代价是板子重启后要重新部署。

### 23.14 怎样自己查看官方差异

先找到 vendor 中对应上游文件，再与工程文件比较。不同 Model Zoo 版本目录层级可能略有差异，可以先执行：

~~~bash
cd /home/lubancat/PPOCR_RV1106

find vendor/rknn_model_zoo/examples/PPOCR -name 'ppocr_system.cc' -o -name 'CMakeLists.txt'
find src/ppocr_system -type f | sort
~~~

确定文件后执行：

~~~bash
diff -u \
  vendor/rknn_model_zoo/examples/PPOCR/PPOCR-System/cpp/rknpu2/ppocr_system.cc \
  src/ppocr_system/rknpu2/ppocr_system.cc \
  > logs/official_vs_rv1106_ppocr_system.diff || true
~~~

阅读 diff 时按功能分组，不要只看加减行数：

1. Context 字段和 native attr。
2. `rknn_create_mem/rknn_set_io_mem`。
3. stride-aware input copy。
4. FP16/INT8 输出恢复。
5. Rec uint8 resize/padding。
6. 条件概率归一化。
7. 释放路径和失败清理。

`diff` 返回 1通常表示文件存在差异，不代表命令执行失败，因此示例使用 `|| true` 保留输出。

### 23.15 怎样证明改动没有把模型改坏

不能只靠“程序能启动”。本工程采用分层证据：

| 层级 | 检查内容 | 当前证据 |
|---|---|---|
| 文件身份 | ONNX/RKNN SHA256 | `models/SHA256SUMS` |
| ONNX 结构 | opset、节点、输入输出、checker | `inspect_onnx_rknn.py` 输出 |
| RKNN 构建 | config/load/build/export 返回值 | 转换日志 |
| 设备分配 | CPU/NPU 节点和内部 fallback | Det verbose、Rec Network Layer Table |
| Runtime 契约 | shape、layout、dtype、scale/zp | `logs/rv1106_run.log` |
| 数值解释 | Det 反量化、Rec FP16、行概率 | C++ 输出检查和条件归一化 |
| 任务结果 | 框、文字、分数和结果图 | 16 行样例输出与 `output/rv1106_test_result.jpg` |

### 23.16 学习时建议做的五个练习

#### 练习 1：画出来源边界

在纸上画三列：PaddleOCR、RKNN Model Zoo、本工程。把模型、转换器、Runtime、C++、脚本分别放到对应列。

#### 练习 2：只读 Det 官方转换器

找到 `config/load_onnx/build/export_rknn` 四步，记录 Det mean/std、target、quant 和 dataset 从哪里来。

#### 练习 3：逐行读 Rec 转换器

解释每个返回值为什么必须检查，并尝试把 `do_quantization=False` 的失败日志与代码对应。

#### 练习 4：对比 C++ IO 路径

在 diff 中分别标记普通 IO、native attr、create mem、set IO mem、run、release。画出内存的创建和释放生命周期。

#### 练习 5：从 Runtime attr 推导字节数

Rec 输出 `[1,40,6625]` FP16：

~~~text
1 × 40 × 6625 × 2 bytes = 530000 bytes
~~~

如果日志大小不是这个值，应先检查 shape、dtype 或 stride，再看 CTC。

### 23.17 本章结论

本工程没有把 PaddleOCR 模型任意改成另一套网络。Det 继续调用 Rockchip 官方转换器并显式传入 RV1106/INT8；Rec 新增专用转换脚本以固化正确的 target、归一化、量化和返回值检查；C++ 主要把通用 RKNN IO 改成当前 RV1106 实测可用的 native 零拷贝，并正确处理 stride、INT8/FP16 输出、Rec uint8 输入和条件概率归一化。DB、裁剪、CTC 和字典映射的算法主链保持上游思路。vendor 原件保留，因此每个变化都能用 diff 和日志复核。

## 24. PP-OCRv4 从 ONNX 转 RKNN：可以逐条执行的完整教程

> 本章补充日期：2026-10-31。第 5 章解释转换参数，第 14、23 章解释相对官方代码的变化；本章专门给出一条从环境检查、模型检查、Det/Rec 转换、日志保存、产物审计到回滚的完整操作链。命令默认在 Ubuntu 20.04 虚拟机执行。

### 24.1 先确认本工程从哪一步开始

本工程当前保存的是已经准备好的两个 ONNX：

~~~text
models/onnx/ppocrv4_det.onnx
models/onnx/ppocrv4_rec.onnx
~~~

因此本章实际执行的是：

~~~text
PP-OCRv4 ONNX
→ RKNN Toolkit2 读取和优化
→ INT8 PTQ 校准
→ 编译 RV1106 执行图
→ 导出 .rknn
~~~

本章不是在板端运行 Paddle，也不是重新训练模型。Paddle inference 模型→ONNX 属于更上游的导出阶段；当前 v4 工程已经冻结 ONNX 作为可复现输入，不应在没有记录来源、配置和数值对比时重新导出并覆盖它。

### 24.2 转换前文件清单

进入工程：

~~~bash
cd /home/lubancat/PPOCR_RV1106
pwd
~~~

期望输出：

~~~text
/home/lubancat/PPOCR_RV1106
~~~

检查必要文件：

~~~bash
test -f models/onnx/ppocrv4_det.onnx
test -f models/onnx/ppocrv4_rec.onnx
test -f scripts/convert_models.sh
test -f scripts/convert_rec_rv1106.py
test -f vendor/rknn_model_zoo/examples/PPOCR/PPOCR-Det/python/convert.py
test -f vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt
echo "conversion inputs exist"
~~~

如果某个 `test -f` 失败，shell 会停止在那一行。先修正文件，不要把空路径交给转换器。

检查 ONNX 当前身份：

~~~bash
sha256sum models/onnx/ppocrv4_det.onnx \
          models/onnx/ppocrv4_rec.onnx
~~~

当前冻结值：

~~~text
55a2092c7068bc8e9c79e4564958beaacdd66e802fd2eeffbba7604003dbb5c2  ppocrv4_det.onnx
773fc4e13c25f4847fb82aa2112dc673355c13c0b32b066ae2fb082478810916  ppocrv4_rec.onnx
~~~

哈希不一致时先确认是否主动换过模型。不要继续转换后再猜输入模型是哪一个。

### 24.3 激活正确的 Toolkit2 环境

~~~bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate toolkit2

which python
python --version
python -m pip show rknn-toolkit2 onnx numpy | \
  grep -E '^(Name|Version):'
~~~

本工程实测环境：

~~~text
Python 3.8.20
rknn-toolkit2 2.3.2（包信息 commit bd980be9）
ONNX 1.16.1
NumPy 1.24.4
~~~

如果 `import rknn` 失败，先确认 `which python` 是否位于 `.../envs/toolkit2/bin/python`。不要在 Paddle 3.x 导出环境里临时安装 Toolkit2，也不要为了一个 import 错误升级整套 NumPy/ONNX。

快速导入检查：

~~~bash
python - <<'PY'
from rknn.api import RKNN
import onnx
import numpy
print('RKNN API import: OK')
print('onnx:', onnx.__version__)
print('numpy:', numpy.__version__)
PY
~~~

### 24.4 检查校准清单

当前清单：

~~~text
vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt
~~~

查看前几行：

~~~bash
sed -n '1,10p' \
  vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt
~~~

统计行数并检查引用图片：

~~~bash
wc -l vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt
~~~

RKNN dataset 文件通常每行写一组模型输入文件路径。单输入图像模型每行是一张图片。路径是相对还是绝对，要结合转换脚本的当前工作目录判断。

本工程运行官方 Det `convert.py` 前会先进入它所在目录，因为该官方脚本内部的 `DATASET_PATH` 是相对于脚本目录设计的。如果直接在工程根目录调用同一个 Python 文件，而脚本仍使用相对校准路径，可能出现“清单文件找不到”或“清单中的图片找不到”。

### 24.5 转换前保护已经验证的 RKNN

当前基线模型：

~~~text
models/rknn/ppocrv4_det_rv1106_i8.rknn
models/rknn/ppocrv4_rec_rv1106_i8.rknn
~~~

在重新转换前建立候选/备份目录：

~~~bash
stamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p "models/rknn/backups/$stamp" logs

for model in \
  models/rknn/ppocrv4_det_rv1106_i8.rknn \
  models/rknn/ppocrv4_rec_rv1106_i8.rknn; do
  if [ -f "$model" ]; then
    cp -a "$model" "models/rknn/backups/$stamp/"
  fi
done

sha256sum models/rknn/backups/$stamp/*.rknn 2>/dev/null || true
~~~

更安全的工程习惯是让实验输出进入 `models/rknn/candidates/<实验名>/`，完成板端回归后再晋级。`convert_models.sh` 会写正式路径，学习时第一次重跑前应先完成上述备份。

### 24.6 一键转换入口到底执行什么

运行：

~~~bash
cd /home/lubancat/PPOCR_RV1106
scripts/convert_models.sh 2>&1 | tee logs/convert_all_new.log
status=${PIPESTATUS[0]}
echo "convert exit code: $status"
test "$status" -eq 0
~~~

为什么读取 `PIPESTATUS[0]`：命令通过管道交给 `tee` 后，普通 `$?` 更接近最后一个 `tee` 的状态；`PIPESTATUS[0]` 才是转换脚本本身的退出状态。

当前 `convert_models.sh` 的核心结构如下：

~~~bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"

if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
fi

conda activate toolkit2
mkdir -p "$ROOT/models/rknn"

DET_DIR="$ROOT/vendor/rknn_model_zoo/examples/PPOCR/PPOCR-Det/python"

(
  cd "$DET_DIR"
  python convert.py \
    "$ROOT/models/onnx/ppocrv4_det.onnx" \
    rv1106 \
    i8 \
    "$ROOT/models/rknn/ppocrv4_det_rv1106_i8.rknn"
)

python "$ROOT/scripts/convert_rec_rv1106.py" \
  --model "$ROOT/models/onnx/ppocrv4_rec.onnx" \
  --dataset "$ROOT/vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt" \
  --output "$ROOT/models/rknn/ppocrv4_rec_rv1106_i8.rknn"
~~~

这段脚本没有实现神经网络算子，它负责把环境、工作目录、模型路径、平台、量化类型和输出路径固定下来。

### 24.7 Det 转换逐步解释

#### 24.7.1 实际命令

~~~bash
ROOT=/home/lubancat/PPOCR_RV1106
DET_DIR="$ROOT/vendor/rknn_model_zoo/examples/PPOCR/PPOCR-Det/python"

cd "$DET_DIR"

python convert.py \
  "$ROOT/models/onnx/ppocrv4_det.onnx" \
  rv1106 \
  i8 \
  "$ROOT/models/rknn/ppocrv4_det_rv1106_i8.rknn" \
  2>&1 | tee "$ROOT/logs/convert_det_manual.log"
~~~

#### 24.7.2 四个位置参数

| 位置 | 值 | 作用 |
|---:|---|---|
| 1 | `ppocrv4_det.onnx` | 输入 ONNX |
| 2 | `rv1106` | 目标平台 |
| 3 | `i8` | 开启 INT8 量化 |
| 4 | `ppocrv4_det_rv1106_i8.rknn` | 输出文件 |

#### 24.7.3 官方 Det `convert.py` 的核心逻辑

~~~python
rknn = RKNN(verbose=False)

rknn.config(
    mean_values=[[123.675, 116.28, 103.53]],
    std_values=[[58.395, 57.12, 57.375]],
    target_platform=platform,
)

ret = rknn.load_onnx(model=model_path)
ret = rknn.build(
    do_quantization=do_quant,
    dataset=DATASET_PATH,
)
ret = rknn.export_rknn(output_path)
rknn.release()
~~~

这份上游 Det 转换器没有在 vendor 中被本工程直接修改；本工程通过命令行把 `platform` 设为 `rv1106`，把 `do_quant` 对应的类型设为 `i8`。

#### 24.7.4 每个 RKNN API 的含义

| API | 输入 | 输出/状态变化 | 失败时优先检查 |
|---|---|---|---|
| `RKNN()` | verbose 选项 | 创建转换对象 | Toolkit2 是否正确安装 |
| `config()` | target、mean/std、量化设置 | 保存预处理和平台配置 | target 拼写、参数版本兼容 |
| `load_onnx()` | ONNX 文件 | 解析 ONNX 图 | 文件、opset、shape、算子、外部权重 |
| `build()` | quant 开关、dataset | 优化、量化、融合、设备规划 | 校准路径、数据、算子、RV1106 限制 |
| `export_rknn()` | 输出路径 | 写出 RKNN 文件 | 目录权限、磁盘空间、前一步是否成功 |
| `release()` | 无 | 释放转换对象 | 应放在成功或异常清理路径 |

### 24.8 为什么另有 Det verbose 转换

官方 Det 脚本创建 `RKNN(verbose=False)`，短日志不足以审计完整 CPU/NPU 表。本工程新增：

~~~text
scripts/convert_det_rv1106_verbose.py
~~~

运行：

~~~bash
cd /home/lubancat/PPOCR_RV1106
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate toolkit2

python scripts/convert_det_rv1106_verbose.py \
  2>&1 | tee logs/convert_det_verbose_new.log

status=${PIPESTATUS[0]}
echo "det verbose exit code: $status"
test "$status" -eq 0
~~~

默认候选输出：

~~~text
models/rknn/candidates/ppocrv4_det_rv1106_i8_verbose.rknn
~~~

它保持官方 Det mean/std、RV1106、INT8 和校准集，只改变日志详细度、返回值检查和输出隔离。当前已有日志结果：169 个 NPU 节点、1 个 CPU 输入边界、内部 CPU fallback 0。

### 24.9 Rec 转换逐步解释

#### 24.9.1 手工运行命令

~~~bash
cd /home/lubancat/PPOCR_RV1106
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate toolkit2

python scripts/convert_rec_rv1106.py \
  --model models/onnx/ppocrv4_rec.onnx \
  --dataset vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt \
  --output models/rknn/ppocrv4_rec_rv1106_i8.rknn \
  2>&1 | tee logs/convert_rec_manual.log

status=${PIPESTATUS[0]}
echo "rec convert exit code: $status"
test "$status" -eq 0
~~~

#### 24.9.2 工程 Rec 转换脚本完整逻辑

~~~python
#!/usr/bin/env python3
import argparse
from pathlib import Path
from rknn.api import RKNN


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default=str(root / 'models/onnx/ppocrv4_rec.onnx'),
    )
    parser.add_argument(
        '--dataset',
        default=str(
            root / 'vendor/rknn_model_zoo/'
                   'datasets/PPOCR/imgs/dataset_20.txt'
        ),
    )
    parser.add_argument(
        '--output',
        default=str(
            root / 'models/rknn/'
                   'ppocrv4_rec_rv1106_i8.rknn'
        ),
    )
    args = parser.parse_args()

    rknn = RKNN(verbose=True)

    ret = rknn.config(
        target_platform='rv1106',
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
    )
    if ret != 0:
        raise SystemExit('RKNN config failed: %s' % ret)

    ret = rknn.load_onnx(model=args.model)
    if ret != 0:
        raise SystemExit('ONNX load failed: %s' % ret)

    ret = rknn.build(
        do_quantization=True,
        dataset=args.dataset,
    )
    if ret != 0:
        raise SystemExit('RKNN build failed: %s' % ret)

    ret = rknn.export_rknn(args.output)
    if ret != 0:
        raise SystemExit('RKNN export failed: %s' % ret)

    rknn.release()
    print('Conversion complete')


if __name__ == '__main__':
    main()
~~~

#### 24.9.3 为什么脚本不直接照抄 Det 参数

| 项目 | Det | Rec |
|---|---|---|
| 网络任务 | 整页文字区域检测 | 单行文字识别 |
| 输入 | `[1,3,480,480]` ONNX | `[1,3,48,320]` ONNX |
| mean | 123.675/116.28/103.53 | 127.5/127.5/127.5 |
| std | 58.395/57.12/57.375 | 127.5/127.5/127.5 |
| Runtime 输出 | `[1,480,480,1]` INT8 | `[1,40,6625]` FP16 |
| 后处理 | DB | CTC |

两个模型的输入分布和输出语义不同，因此不能用同一个配置文件只改 ONNX 路径。

#### 24.9.4 为什么每一步都检查 `ret`

如果 `load_onnx()` 失败后仍调用 `build()`，后续错误会掩盖真正根因。如果 `build()` 失败但目录里残留上一次 RKNN，操作者可能误以为新转换成功。当前脚本在每一步失败时立即退出，确保日志中的成功对应这次运行。

### 24.10 转换完成后先检查文件，不要立刻上板

~~~bash
cd /home/lubancat/PPOCR_RV1106

ls -lh \
  models/rknn/ppocrv4_det_rv1106_i8.rknn \
  models/rknn/ppocrv4_rec_rv1106_i8.rknn

file \
  models/rknn/ppocrv4_det_rv1106_i8.rknn \
  models/rknn/ppocrv4_rec_rv1106_i8.rknn

sha256sum \
  models/rknn/ppocrv4_det_rv1106_i8.rknn \
  models/rknn/ppocrv4_rec_rv1106_i8.rknn
~~~

已经验证的基线哈希为：

~~~text
Det RKNN:
9fda16291222154d0fcf12ef20e82482719ce825b3965bf667f031a9285bf690

Rec RKNN:
8082ac32d95d96346c50e5166aa5304a22b6231165433ffca96d1f1a99ef0938
~~~

重新转换后 SHA 不一定逐字节相同。哈希变化不自动等于错误，也不自动等于正确；它表示产生了一个新候选，必须重新审计和上板回归。

### 24.11 运行 ONNX/RKNN 日志审计

切换到含 ONNX Python 包的只读审计环境：

~~~bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ppocr_latest

cd /home/lubancat/PPOCR_RV1106

python scripts/inspect_onnx_rknn.py \
  --model models/onnx/ppocrv4_det.onnx \
  --model models/onnx/ppocrv4_rec.onnx \
  --log logs/convert_det_verbose_new.log \
  --log logs/convert_rec_manual.log \
  --show-nodes Shape Slice ReduceMean Pow Sqrt MatMul Softmax Resize
~~~

重点看：

1. ONNX IR/opset、输入输出和动态维度。
2. `rknn.build` 是否真正完成。
3. 是否出现完整 Network Layer Information Table。
4. CPU 节点是输入输出边界还是内部计算。
5. warning 对应什么 tensor。
6. error-looking 行之后是否仍有 build/export 成功。
7. 输出模型路径是否是本次候选。

### 24.12 当前设备分配应怎样理解

#### Det verbose 候选

~~~text
NPU：169
CPU：1 个 InputOperator 边界
内部 CPU fallback：0
~~~

#### Rec

~~~text
NPU：125
CPU：7
  ├─ InputOperator：1
  ├─ OutputOperator：1
  └─ 内部 exNorm：5
内部 CPU fallback：5
~~~

Rec 的 5 个 `exNorm` 是 LayerNorm 融合节点。模型仍由 NPU 执行为主，但准确表述应是混合执行。

### 24.13 转换日志的成功判据

一次转换至少同时满足：

- `config/load/build/export` 返回 0。
- 日志没有未处理的 Python traceback。
- 输出文件存在且非空。
- 日志末尾能定位到本次输出路径。
- verbose 构建能看到设备表，或明确记录为什么没有。
- 模型身份、配置、校准集和输出 SHA 被保存。

不能使用这些错误判据：

- “目录里有 `.rknn`，所以这次成功。”可能是旧文件。
- “日志有 `E`，所以必然失败。”v4 Rec 有 fallback 相关 error-level 诊断，但后续成功导出。
- “文件名有 i8，所以所有输出都是 INT8。”Rec 输出实际是 FP16。
- “build 成功，所以所有节点都在 NPU。”必须读取 Target 表。

### 24.14 常见转换错误与定位顺序

#### 错误 1：`No module named rknn`

~~~bash
which python
conda env list
conda activate toolkit2
python -c "from rknn.api import RKNN; print('OK')"
~~~

#### 错误 2：找不到 dataset 或图片

检查：

~~~bash
pwd
ls -l vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt
sed -n '1,5p' vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt
~~~

Det 官方脚本还要检查其当前工作目录，因为内部 `DATASET_PATH` 是相对路径。

#### 错误 3：`load_onnx failed`

优先检查：模型文件是否正确、是否为空、ONNX checker、opset、输入输出 shape、外部权重、失败节点名。不要先调 DB 阈值。

#### 错误 4：RV1106 非量化构建失败

实测日志：

~~~text
Current target_platform(rv1106) not support do_quantization = False!
Build model failed!
~~~

本工程 Rec 使用 `do_quantization=True`。不能忽略返回值继续导出。

#### 错误 5：`Unkown op target: 0`

按时间顺序继续检查 build 返回值、设备表、Mini RKNN export、正式 export 和板端运行。当前 v4 Rec 的 5 条信息与 5 个 CPU `exNorm` 对应，但新模型不能直接照搬这个结论。

#### 错误 6：qtype 默认 FP16

定位对应 tensor，检查是否处于 attention、Softmax、Reshape/Transpose 或 Resize 常量边界。只要设备表、数值、任务精度和延迟通过，可以记录后接受；不要为了消除 warning 盲目插 Cast。

### 24.15 新 RKNN 怎样晋级成正式模型

转换成功后按顺序执行：

1. 保存 ONNX、脚本、校准清单、日志和 RKNN SHA。
2. 运行设备分配审计。
3. 执行 `scripts/build_rv1106.sh`，生成新的安装包。
4. 连接开发板并查询 Runtime attr。
5. 固定样例冒烟测试。
6. 多张业务图片检查 Det 框、crop 和 Rec 文字。
7. 有标注集时计算检测 P/R/F1、识别 CER/整行准确率。
8. 测量分阶段延迟和稳定性。
9. 全部通过后更新 `models/SHA256SUMS`、文档和回滚包。

在此之前，新文件只是候选，不应覆盖“已验证基线”的结论。

### 24.16 转换失败怎样恢复基线

查找备份：

~~~bash
find models/rknn/backups -maxdepth 2 -type f -name '*.rknn' -printf '%p %s bytes\n'
~~~

确认备份哈希后恢复：

~~~bash
cp -a models/rknn/backups/<时间戳>/ppocrv4_det_rv1106_i8.rknn \
      models/rknn/ppocrv4_det_rv1106_i8.rknn

cp -a models/rknn/backups/<时间戳>/ppocrv4_rec_rv1106_i8.rknn \
      models/rknn/ppocrv4_rec_rv1106_i8.rknn

sha256sum -c models/SHA256SUMS
~~~

恢复必须把 Det、Rec、字典、C++ 预处理和参数视为一个组合。不能只恢复 Rec 模型，却继续使用另一个类别数或字典。

### 24.17 从转换到板端的最短复现路径

已经理解上述细节后，可以使用：

~~~bash
cd /home/lubancat/PPOCR_RV1106

# 1. 检查输入
sha256sum models/onnx/*.onnx

# 2. 转 RKNN
scripts/convert_models.sh 2>&1 | tee logs/convert_all_new.log
test "${PIPESTATUS[0]}" -eq 0

# 3. 检查输出
ls -lh models/rknn/*.rknn
sha256sum models/rknn/*.rknn

# 4. 编译板端程序
scripts/build_rv1106.sh 2>&1 | tee logs/build_new.log
test "${PIPESTATUS[0]}" -eq 0

# 5. ADB 连板并运行
adb devices
scripts/deploy_and_run.sh data/test.jpg

# 6. 看证据
sed -n '1,220p' logs/rv1106_run.log
ls -lh output/rv1106_test_result.jpg
~~~

### 24.18 本章总结

完成本章后，应能独立回答四个问题：输入 ONNX 和校准集在哪里；Det 与 Rec 为什么使用不同转换入口；怎样从构建日志判断量化、CPU fallback 与设备分配；新模型为什么必须经过哈希、日志、实板和回归图片四层验证后才能替换正式模型。

---

## 25. PP-OCRv4 在 RV1106 上的速度与精度实测（2026-10-31）

本章不是根据模型宣传页估算，而是对当前工程里的两个正式 RKNN、当前 C++ Runtime 和连接的 RV1106 实物板进行的实测。为了让数字能够复查，本章同时记录测试对象的 SHA256、计时代码插入位置、预热次数、正式运行次数、温度、统计方法、人工真值、错误案例和原始日志位置。

### 25.1 先看结论，但不要脱离测试条件

速度样例为 `data/test.jpg`，分辨率 500×500，Det 得到 16 个框，Rec 串行执行 16 次。5 次预热后正式运行 30 次：

| 指标 | 均值 | P50 | P90 | P95 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| 纯 OCR System：Det + 裁剪 + 16 次 Rec | 1646.978 ms | 1646.905 ms | 1669.107 ms | 1671.961 ms | 1617.263 ms | 1681.616 ms |
| 完整程序：模型初始化 + 读图 + OCR + 绘图 + 释放 | 1766.382 ms | 1759.711 ms | 1797.904 ms | 1819.864 ms | 1728.238 ms | 1829.646 ms |
| Det 整段 | 168.214 ms | 164.624 ms | 181.935 ms | 183.719 ms | 163.789 ms | 196.111 ms |
| Det 的 `rknn_run()` | 93.216 ms | 92.907 ms | 94.311 ms | 94.442 ms | 92.792 ms | 96.093 ms |
| 单个 Rec 整段，共 480 个 crop | 86.983 ms | 86.075 ms | 88.488 ms | 97.297 ms | 84.608 ms | 121.623 ms |
| 单个 Rec 的 `rknn_run()`，共 480 次 | 57.356 ms | 57.124 ms | 58.084 ms | 59.221 ms | 57.012 ms | 63.652 ms |

按均值换算：

- 只算 OCR System，这张 16 框图片约为 `1000 / 1646.978 = 0.607 image/s`。
- 算完整一次性进程约为 `0.566 image/s`；服务化常驻进程不会每张图都重新初始化模型，因此不应拿这个数直接代表常驻吞吐。
- Det 整段约为 5.945 image/s；Det 单次 NPU 执行约为 10.728 run/s。
- Rec 整段约为 11.496 crop/s；Rec 单次 NPU 执行约为 17.435 run/s。
- 16 次 Rec 合计均值 1391.731 ms，占 OCR System 均值约 84.50%；Det 占约 10.21%，裁剪占约 4.31%。当前端到端速度的第一瓶颈是**逐框串行识别**，不是 Det。
- 板温从预热前 43.415°C 上升到 30 轮完成后的 49.950°C，温升 6.535°C；30 次均成功，未观察到运行错误。

精度采用工程内两张图片的人工真值，共 19 个视觉文本行。严格口径是 Unicode NFKC、删除空白、保留标点：

| 范围 | 输出模式 | 精确匹配行 | 严格 CER | `1-CER` 字符准确率 |
|---|---|---:|---:|---:|
| `test.jpg`，16 行 | 当前阈值后的端到端输出 | 12/16 = 75.00% | 3.9409% | 96.0591% |
| `test1.jpg`，3 行 | Rec 原始结果，不做 0.5 过滤 | 0/3 | 16.6667% | 83.3333% |
| `test1.jpg`，3 行 | 当前阈值后的端到端输出 | 0/3 | 22.2222% | 77.7778% |
| 两图合计，19 行 | Rec 原始结果 | 12/19 = 63.1579% | 4.9774% | 95.0226% |
| 两图合计，19 行 | 当前阈值后的端到端输出 | 12/19 = 63.1579% | 5.4299% | 94.5701% |

`test1.jpg` 的两个长句只缺结尾标点，因此严格整行匹配为 0；如果删除标点和符号，只比较文字内容，端到端为 2/3 精确行，内容 CER 为 12.5%。两图合计的内容精确行为 14/19，内容 CER 为 5.5556%。报告严格值和内容值的目的，是把“文字本身识别正确”和“标点也完全一致”分开，而不是挑一个更好看的数字。

> 重要边界：19 行是用于检查转换后模型是否明显退化的小样本，不是官方 benchmark。两张图的 Det 框数都与人工数出的视觉行数一致，共 19/19，但这只是**框数量一致性代理**。没有逐点 polygon 真值，不能计算标准 Det precision、recall、Hmean，也不能宣称 Det 准确率 100%。

### 25.2 被测对象必须锁定，否则数字不能复现

本轮使用的模型与程序如下：

| 文件 | SHA256 |
|---|---|
| `models/onnx/ppocrv4_det.onnx` | `55a2092c7068bc8e9c79e4564958beaacdd66e802fd2eeffbba7604003dbb5c2` |
| `models/onnx/ppocrv4_rec.onnx` | `773fc4e13c25f4847fb82aa2112dc673355c13c0b32b066ae2fb082478810916` |
| `models/rknn/ppocrv4_det_rv1106_i8.rknn` | `9fda16291222154d0fcf12ef20e82482719ce825b3965bf667f031a9285bf690` |
| `models/rknn/ppocrv4_rec_rv1106_i8.rknn` | `8082ac32d95d96346c50e5166aa5304a22b6231165433ffca96d1f1a99ef0938` |
| 30 轮实测及最终交付的带计时 `rknn_ppocr_system_demo` | `0ce45e03c0e9a5e4a3a9dcb57debc172acc2fb29c1bd49d65466a05d2531ad0f` |

平台信息：

| 项目 | 实测值 |
|---|---|
| Toolkit2 | 2.3.2，位于 conda 环境 `toolkit2` |
| 板端内核 | Linux 5.10.160，armv7l |
| 板端总内存 | 185556 kB，约 181 MiB |
| 测试程序 | ARMhf/uClibc Release 构建 |
| Det RKNN 输入 | NHWC INT8，`1×480×480×3` |
| Det RKNN 输出 | NHWC INT8，`1×480×480×1` |
| Rec RKNN 输入 | NHWC INT8，`1×48×320×3` |
| Rec RKNN 输出 | FP16，时间步 40，字符类别 6625 |
| 测试图片 | `data/test.jpg`，500×500，Det 16 框 |
| 运行次数 | 5 次预热，不入统计；30 次正式运行 |

为什么要记录哈希：同名文件可能已经被重新量化或覆盖。只记“PP-OCRv4 INT8”无法判断两次测试是不是同一个模型；哈希一致才能把结果绑定到同一份字节内容。

### 25.3 “NPU 推理”在本工程中到底指什么

Det 和每个 Rec crop 都通过 RKNN Runtime 调用 `rknn_run()`，计时日志中的 `npu_ms` 就包围这一调用，因此模型主体确实在 RV1106 RKNN/NPU 路径上执行，不是拿 ONNX Runtime 或 Paddle CPU 在板端冒充。

但完整 OCR 不是“所有步骤全在 NPU”：

1. JPEG 解码、缩放/填充、内存拷贝在 CPU/RGA 辅助路径。
2. Det 的输出读取、反量化和 DB 后处理在 CPU。
3. 文字框裁剪、透视/旋转处理在 CPU/OpenCV。
4. Rec 输出读取、必要的行归一化、CTC 解码和字典映射在 CPU。
5. v4 Rec 的 RKNN 构建信息还显示 5 个内部 `exNorm` 节点由 CPU 执行；其余主体节点在 NPU。准确说法是“RKNN Runtime 管理、NPU 为主、带少量 CPU fallback 的混合执行”。

因此 `npu_ms` 不能等同于端到端耗时。以 Det 均值为例，NPU 是 93.216 ms，但 Det 整段是 168.214 ms；Rec 单框 NPU 是 57.356 ms，包含 IO 和解码后是 86.983 ms。

### 25.4 为什么新增计时，而不是在 ADB 外面掐秒表

如果在虚拟机上用 `time adb shell ...`，结果还会混入 ADB 建连、USB 传输、shell 启动和终端输出时间。为了区分真正的模型与后处理开销，工程在板端进程内部用 `std::chrono::steady_clock` 计时。

`steady_clock` 的理由：它是单调时钟，不受系统时间校准影响，适合计算时间差。计时只在环境变量 `PPOCR_BENCH` 存在且不为 `0` 时打印；正常部署不设置该变量，程序功能与输出格式不受影响。

新增的日志层级：

~~~text
BENCH_DET      Det 预处理 / rknn_run / 输出读取 / DB 后处理 / 总计
BENCH_REC      每个 crop 的预处理 / rknn_run / 输出读取 / 归一化+CTC / 总计
BENCH_SYSTEM   Det + 全部裁剪 + 全部 Rec + 循环杂项
BENCH_MAIN     两模型初始化 / 读图 / System / 绘制和写图 / 释放前总计
BENCH_PROGRAM  从 main 进入到模型释放、内存释放完成的进程总计
~~~

打印示例：

~~~text
BENCH_DET pre_ms=44.668 npu_ms=92.732 output_ms=5.067 post_ms=21.789 total_ms=164.265 boxes=16
BENCH_REC call=1 pre_ms=0.958 npu_ms=57.049 output_ms=7.569 post_ms=21.673 total_ms=87.256 score=0.709973
BENCH_SYSTEM det_boxes=16 accepted=16 rec_calls=16 det_ms=164.265 crop_ms=69.659 rec_ms=1367.836 other_ms=12.886 total_ms=1614.646
BENCH_MAIN det_init_ms=42.816 rec_init_ms=60.033 read_ms=14.933 inference_ms=1614.774 draw_write_ms=20.393 before_release_ms=1752.978
BENCH_PROGRAM total_ms=1756.155
~~~

### 25.5 官方/原工程代码具体改了哪里，为什么改

这次没有改 ONNX 网络、RKNN 转换参数、量化集、字典、DB 阈值和 CTC 算法；只给板端 C++ 加了**可关闭的观测代码**。主要改动在：

- `src/ppocr_system/rknpu2/ppocr_system.cc`
- `src/ppocr_system/main.cc`

`ppocr_system.cc` 的变化：

1. 加入 `<chrono>`，提供毫秒级单调时间函数。
2. 加入 `PPOCR_BENCH` 环境变量开关。
3. 在 Det 的预处理结束、`rknn_run()` 前后、输出读取前后和 DB 后处理结束处取时间戳。
4. 在每次 Rec 的预处理、`rknn_run()`、输出读取和归一化/CTC 后处理处取时间戳。
5. 在 `inference_ppocr_system_model()` 入口清零累计结构体，累计所有裁剪与识别时间，避免跨图片污染。
6. 输出 `det_boxes`、`accepted` 和 `rec_calls`，因为总耗时必须和文字框数量一起解释。
7. 为核对 `test1.jpg` 被 `TEXT_SCORE=0.5` 过滤的竖排标题，曾临时给 `BENCH_REC` 增加 `text=%s`，捕获到原始结果 `斗`、分数 0.457527。该诊断结果保存在 `test1_with_all_rec_text.log`。完成定位后又移除了 `text=%s` 并重新构建，因为打印长度不固定的识别文本会给 benchmark 引入额外终端 IO；最终源码、最终二进制和 30 轮测速使用相同的纯数值日志格式。

`main.cc` 的变化：

1. 分别包围 Det 模型初始化、Rec 模型初始化和 `read_image()`。
2. 包围 `inference_ppocr_system_model()`，用于和内部 `BENCH_SYSTEM` 相互校验。
3. 包围绘框与 `write_image()`。
4. 在释放前打印 `BENCH_MAIN`，释放完成后打印 `BENCH_PROGRAM`。

为什么不把日志无条件打开：Rec 每个框都会打印一行，正式产品中会增加串口/终端 IO，并污染业务日志。环境变量让同一二进制兼顾验证和正常运行。

原始源码已备份到：

~~~text
/home/lubancat/PPOCR_RV1106/backups/20261031_v4_benchmark/src/main.cc
/home/lubancat/PPOCR_RV1106/backups/20261031_v4_benchmark/src/ppocr_system.cc
/home/lubancat/PPOCR_RV1106/backups/20261031_v4_benchmark/install/rknn_ppocr_system_demo.before_bench
~~~

### 25.6 本轮实际遇到的编译错误，以及为什么这样修

首次给 `main.cc` 加绘图计时时，代码把以下变量声明在推理失败的 `goto out` 之后：

~~~cpp
const double draw_start_ms = bench_now_ms();
const double draw_write_ms = bench_now_ms() - draw_start_ms;
~~~

交叉编译器报错：

~~~text
error: jump to label 'out' ... crosses initialization of 'const double draw_write_ms'
error: jump to label 'out' ... crosses initialization of 'const double draw_start_ms'
~~~

原因不是 RKNN 或模型算子不兼容，而是标准 C++ 控制流规则：`goto out` 不能跳过带初始化的局部变量后进入它们仍在作用域内的位置。修复方法是在函数前部、任何可能 `goto` 之前声明普通变量：

~~~cpp
double draw_start_ms = 0.0;
double draw_write_ms = 0.0;
~~~

到原来的位置再赋值：

~~~cpp
draw_start_ms = bench_now_ms();
draw_write_ms = bench_now_ms() - draw_start_ms;
~~~

这样所有跳转路径都已经完成变量初始化，统一释放结构保持不变。修复后的构建日志为：

~~~text
/home/lubancat/PPOCR_RV1106/logs/build_v4_benchmark_main_20261031.log
BUILD_MAIN_EXIT=0
~~~

编译器仍提示字符串常量转换为 `char*` 的上游 warning，但不是本轮新增错误，链接和安装均成功。

### 25.7 速度测试怎样复现

先编译并部署：

~~~bash
cd /home/lubancat/PPOCR_RV1106
conda activate ppocr_latest
scripts/build_rv1106.sh
scripts/deploy_and_run.sh
~~~

单次查看分段日志：

~~~bash
sudo adb shell '
  cd /tmp/ppocr_rv1106 &&
  export LD_LIBRARY_PATH=/tmp/ppocr_rv1106/lib &&
  export PPOCR_BENCH=1 &&
  ./rknn_ppocr_system_demo \
    model/ppocrv4_det_i8.rknn \
    model/ppocrv4_rec_i8.rknn \
    model/test.jpg
'
~~~

测试时先做 5 次预热。预热用于让文件缓存、模型 Runtime 和工作频率进入较稳定状态，但不写入统计文件。随后执行 30 次，每次都启动新进程；这同时能观察模型初始化的分布。如果目标是常驻服务，应另外在同一进程中循环图片，避免重复初始化。

原始日志：

~~~text
/home/lubancat/PPOCR_RV1106/logs/v4_benchmark_20261031/test_jpg_30runs_raw.log
~~~

统计脚本：

~~~bash
python tools/analyze_v4_benchmark.py \
  logs/v4_benchmark_20261031/test_jpg_30runs_raw.log \
  --json logs/v4_benchmark_20261031/test_jpg_30runs_stats.json
~~~

脚本对每种字段输出：样本数、均值、总体标准差、最小值、P50、P90、P95 和最大值。30 次 `BENCH_DET/SYSTEM/MAIN/PROGRAM` 各有 30 条；每次 16 个 crop，所以 `BENCH_REC` 应有 `30×16=480` 条。如果数量不一致，应先查程序是否中途失败，不能直接计算均值。

### 25.8 更细的耗时结构

Det 30 次：

| 分段 | 均值 | P50 | P95 | 解释 |
|---|---:|---:|---:|---|
| `pre_ms` | 46.437 ms | 44.689 ms | 56.459 ms | 500×500 图像到 480×480 模型输入；日志提示原图宽度未按 4/16 对齐，发生 CPU convert |
| `npu_ms` | 93.216 ms | 92.907 ms | 94.442 ms | `rknn_run()` |
| `output_ms` | 5.015 ms | 5.014 ms | 5.077 ms | 输出读取/整理 |
| `post_ms` | 23.536 ms | 21.806 ms | 35.425 ms | DB 后处理、轮廓、框过滤 |
| `total_ms` | 168.214 ms | 164.624 ms | 183.719 ms | 上述分段合计附近 |

Rec 共 480 个 crop：

| 分段 | 均值 | P50 | P95 | 解释 |
|---|---:|---:|---:|---|
| `pre_ms` | 0.747 ms | 0.677 ms | 0.963 ms | 单个文字 crop 缩放/填充与输入写入 |
| `npu_ms` | 57.356 ms | 57.124 ms | 59.221 ms | 单个 crop 的 `rknn_run()` |
| `output_ms` | 6.763 ms | 6.068 ms | 7.622 ms | 读取 FP16 输出并转为后处理数据 |
| `post_ms` | 22.110 ms | 21.014 ms | 31.435 ms | 概率归一化与 CTC/字典解码 |
| `total_ms` | 86.983 ms | 86.075 ms | 97.297 ms | 单个 crop 总时间 |

完整进程的其他均值：Det 初始化 42.231 ms、Rec 初始化 37.531 ms、读图 15.631 ms、绘制加写图 20.552 ms。初始化最大值明显高于中位数，因此只报告单次结果会非常容易误判。

### 25.9 精度真值是怎样建立的

工程原本只有图片，没有官方 label。不能把程序预测复制一份再称为真值，因此本轮实际打开两张原图，逐行人工抄录，然后把每个视觉文本行与 Rec 输出对齐。

真值与预测文件：

~~~text
accuracy/v4_manual_gt_and_predictions.json
~~~

评测脚本：

~~~text
tools/evaluate_v4_accuracy.py
~~~

复现命令：

~~~bash
python tools/evaluate_v4_accuracy.py \
  accuracy/v4_manual_gt_and_predictions.json \
  --json accuracy/v4_accuracy_results.json
~~~

CER 的定义：

~~~text
CER = Levenshtein 编辑距离（替换 + 删除 + 插入）/ 真值字符数
字符准确率 = 1 - CER
~~~

严格模式先执行 Unicode NFKC，再删除空白，保留标点和大小写。内容模式在此基础上删除 Unicode 标点与符号。由于删除标点会同时缩短分母，内容 CER 不保证一定比严格 CER 小；它只是回答另一个问题。

### 25.10 两张图具体错在哪里

`test.jpg` 的 16 行中，12 行在严格归一化后完全一致。主要错误：

| 真值 | 板端结果 | 类型 |
|---|---|---|
| `纯臻营养护发素` | `纯营养护发素` | 漏掉低频/细笔画字“臻” |
| `【品名】：纯臻营养护发素` | `【品名】：纯营养护发素` | 同一词再次漏“臻” |
| `【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚` | `【主要成分】：硬脂醇、麦β-葡` | 多个前后字符漏识别 |
| `糖、椰油酰胺丙基甜菜碱、泛醌` | `糖、椰油酰胺丙基甜菜、泛配` | 漏“碱”，把“醌”识别成“配” |

空格差异如 `OEM ODM`/`ODMOEM` 在严格归一化时已经删除空白，因此没有被当成错误。

`test1.jpg` 是竖排书法式图片：

- 真值 `奋斗`，Rec 原始结果只有 `斗`，分数 0.457527；当前 `TEXT_SCORE=0.5` 将整行过滤，因此端到端不输出该行。
- 真值 `梅花香自苦寒来。`，结果为 `梅花香自苦寒来`，文字正确但句号缺失。
- 真值 `宝剑锋从磨砺出，`，结果为 `宝剑锋从磨砺出`，文字正确但逗号缺失。

这说明把阈值从 0.5 降到 0.45 只会让不完整的 `斗` 出现在结果里，并不会自动恢复 `奋斗`。阈值是“是否输出”的开关，不是纠错器。要修复该案例，应先检查竖排框方向、裁剪内容和识别模型对竖排/书法字体的覆盖，再做阈值 A/B 测试。

### 25.11 精度数字为什么不能写成“检测准确率 100%”

标准 Det 评测需要每张图的真值 polygon，并用 IoU 匹配预测框，统计 TP、FP、FN，随后计算 precision、recall 和 Hmean。当前只有人工数出的视觉文本行和预测框数量：`test.jpg` 为 16 对 16，`test1.jpg` 为 3 对 3。

相同数量并不保证相同位置：完全错误的框也可能碰巧数量一样；一个真值行被拆成两个框、另一个真值行漏掉时，总数仍可能相同。因此文档只写“框数量一致性 19/19”，不写 Det precision/recall。

要得到正式 Det 指标，应至少：

1. 用 PaddleOCR 标注格式为足量业务图逐点画 polygon。
2. 固定 train/validation/test 划分，当前调阈值时只能看 validation，最终只在 test 上报告一次。
3. 使用与 PaddleOCR 官方一致的匹配和 ignore-region 规则。
4. 同时保存原始预测框、阈值、模型哈希与评测脚本版本。
5. 分场景报告小字、竖排、倾斜、反光、低照、模糊和长文本，不只报一个总数。

### 25.12 从结果看，下一步优化应先做什么

按收益优先级排序：

1. **把程序改为模型常驻。**当前完整程序每张图都重新初始化 Det/Rec，均值多约 80 ms，并引入长尾。服务循环应初始化一次，多图复用，退出时再释放。
2. **减少逐框串行 Rec。**16 框的 Rec 合计占 84.5%。需要研究模型/Runtime 是否可做小 batch、双缓冲或流水线；RV1106 内存很小，必须实测峰值内存，不能盲目加 batch。
3. **优化 Rec CPU 后处理。**单框后处理均值 22.110 ms，16 次约 354 ms。检查概率输出是否已是 softmax、能否减少重复归一化，以及 CTC 循环和字典访问是否可缩减；任何删除归一化的改动都必须用 CER 回归验证。
4. **避免 Det 输入的 CPU 对齐转换。**日志明确显示 `src width is not 4/16-aligned, convert image use cpu`。相机链路可直接提供对齐 buffer，或在上游生成 480×480/对齐输入，但必须保持 resize、padding 和坐标回映一致。
5. **针对竖排和生僻词建业务校准/验证集。**当前错误集中在“臻、鲸蜡、燕麦、碱、醌”和竖排标题。量化校准集应覆盖实际文字 crop 分布；若要进一步提高识别精度，通常需要更合适的 Rec 模型/字典或业务微调，而不是只调 Det 阈值。
6. **阈值搜索必须在验证集上做。**Det `threshold`、`box_threshold`、`unclip_ratio` 与 Rec `TEXT_SCORE` 应网格搜索并同时看 precision/recall/CER/延迟；不能为了当前两张图手调后宣称全局更好。

本章没有声称上述优化已经产生新速度或新精度。它们是由本次 profile 推导出的下一轮实验方向，只有再次运行同一套 benchmark 并给出对照表后，才能写成“已优化”。

### 25.13 原始证据和文件清单

虚拟机工程内：

~~~text
logs/v4_benchmark_smoke_20261031.log
logs/v4_benchmark_20261031/test_jpg_30runs_raw.log
logs/v4_benchmark_20261031/test_jpg_30runs_stats.json
logs/v4_benchmark_20261031/test1_raw.log
logs/v4_benchmark_20261031/test1_with_all_rec_text.log
logs/v4_benchmark_20261031/environment.txt
logs/build_v4_benchmark_20261031.log
logs/build_v4_benchmark_main_20261031.log
logs/build_v4_benchmark_text_20261031.log
logs/build_v4_benchmark_final_20261031.log
logs/v4_benchmark_final_smoke_20261031.log
tools/analyze_v4_benchmark.py
tools/evaluate_v4_accuracy.py
accuracy/v4_manual_gt_and_predictions.json
accuracy/v4_accuracy_results.json
~~~

本地交付包也保存了同名脚本、人工真值和结果，便于在不连接板卡时学习统计逻辑。

### 25.14 怎样检查测试有没有被“做漂亮”

复核时至少做以下检查：

1. `record_counts` 是否是 Det/System/Main/Program 各 30、Rec 480。
2. 原始日志中每轮是否都有 `RUN_BEGIN` 和 `RUN_END`，中间是否出现错误。
3. 测试前后温度是否被记录，是否发生高温降频。
4. 模型 SHA256 是否与表格一致。
5. 精度 JSON 中真值是否真的来自原图，而不是复制预测。
6. 被阈值过滤的行是否仍在 raw Rec 指标中保留，避免“只评估输出正确的行”。
7. 是否同时报告均值、分位数和最大值；只报最好的一次没有代表性。
8. 是否把“每个 crop 的 Rec”误写成“每张图的 Rec”。本图有 16 个 crop，二者差 16 倍左右。

### 25.15 本章最终结论

当前 PP-OCRv4 RKNN 已在 RV1106 上实际调用 NPU 路径完成 Det 和 Rec。对 500×500、16 框样例，OCR System 均值 1646.978 ms，完整进程均值 1766.382 ms；Rec 串行循环是最大瓶颈。两张人工标注小样本的端到端严格 CER 为 5.4299%，字符准确率为 94.5701%，但样本只有 19 行，不能外推为正式业务精度。后续优化应优先做模型常驻、Rec 流水/批处理可行性、CPU 后处理优化与真实业务标注集，而不是用单次最好耗时或框数量一致性代替完整结论。

v4 转 RKNN 的核心不是一句 `python convert.py`，而是一条受版本和数据契约约束的链：确认冻结 ONNX和环境，检查校准清单，Det 复用官方转换器并传入 `rv1106/i8`，Rec 使用工程专用脚本写入正确 mean/std 和 INT8 校准，保存完整返回值与日志，核对产物和设备表，最后经过交叉编译与实板回归才能晋级。模型权重没有被手工修改；本工程增加的是可重复的转换入口、RV1106 平台参数、Rec 配置、失败保护、设备审计和板端 Runtime 适配。
