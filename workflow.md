# YOLO 26 TensorRT 推理实验实现流程

> 实验平台：RTX 5090 | CUDA 13.3 | Ubuntu 24.04 (WSL)

---

## 目录

- [第一步：配置运行环境](#第一步配置运行环境)
- [第二步：模型权重下载与 ONNX 转换](#第二步模型权重下载与-onnx-转换)
- [第三步：编译 TensorRT 计算引擎](#第三步编译-tensorrt-计算引擎)
- [第四步：加载推理程序并执行](#第四步加载推理程序并执行)
- [附录：常见问题排查](#附录常见问题排查)

---

## 第一步：配置运行环境

### 1.1 前置条件

确保系统已安装以下基础组件（安装步骤省略）：

- NVIDIA 显卡驱动（支持 CUDA 13.x）
- CUDA Toolkit 13.3
- Ubuntu 24.04（或 WSL2 环境）

### 1.2 安装 Miniconda3

```Shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### 1.3 配置国内镜像源（可选）

为加速依赖包下载，建议配置清华镜像源：

```Shell
vim ~/.condarc
```

```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

更新缓存：

```Shell
conda clean -i 
```

### 1.4 创建虚拟环境

```Shell
# 创建独立虚拟环境（Python 3.12）
conda create -n yolo-onnx python=3.12 -y

# 激活环境（后续所有操作必须在该环境内执行）
conda activate yolo-onnx
```

### 1.5 安装 Python 依赖

```Shell
pip install torch torchvision torchaudio
pip install ultralytics onnx onnxsim opencv-python pycuda
pip install nvidia-modelopt[onnx] onnxruntime-gpu onnx-graphsurgeon protobuf 
```

### 1.6 安装 TensorRT

前往 NVIDIA 开发者官网注册并下载对应版本的 TensorRT：

> **下载地址**：<https://developer.nvidia.com/tensorrt/download/11x>
> **推荐版本**：[TensorRT 11.1.0 GA for Ubuntu 24.04 and CUDA 13.0 to 13.3 DEB local repo Package](https://developer.download.nvidia.com/compute/tensorrt/11.1.0/local_installers/nv-tensorrt-local-repo-ubuntu2404-11.1.0-cuda-13.3_1.0-1_amd64.deb)

下载完成后执行以下命令安装：

```Shell
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2x04-11.x.x-cuda-x.x_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2x04-11.x.x-cuda-x.x/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

**安装完成后重启终端**，确保环境变量生效。

### 1.7 验证环境

```Shell
# 验证 CUDA
nvcc --version

# 验证 TensorRT
trtexec --help

# 验证 Python 依赖
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

---

## 第二步：模型权重下载与 ONNX 转换

### 2.1 准备工作

确保已激活虚拟环境：

```Shell
conda activate yolo-onnx
```

### 2.2 导出 ONNX 模型

创建 `onnx_export.py` 脚本：

```Python
from ultralytics import YOLO

model = YOLO("yolo26x.pt")

# 导出静态 shape 的 ONNX，适配 TensorRT 最优性能
onnx_path = model.export(
    format="onnx",
    imgsz=640,
    opset=13,
    dynamic=False,
    half=False,
    simplify=True
)
print(f"ONNX 模型已导出: {onnx_path}")
```

执行导出：

```Shell
python onnx_export.py
```

> **参数说明**：
> - `imgsz=640`：输入图像尺寸
> - `opset=13`：ONNX 算子集版本
> - `dynamic=False`：禁用动态 shape，提升 TensorRT 性能
> - `half=False`：先导出 FP32 模型，后续再转换为 FP16

---

## 第三步：编译 TensorRT 计算引擎

### 3.1 FP16 量化

将 ONNX 模型转换为 FP16 精度以加速推理：

```Shell
python -m modelopt.onnx.quantization.autocast \
    --onnx_path yolo26x.onnx \
    --output_path yolo26x_fp16.onnx \
    --low_precision_type fp16
```

### 3.2 编译引擎

```Shell
trtexec --onnx=yolo26x_fp16.onnx \
        --saveEngine=yolo26x_fp16.engine \
        --memPoolSize=workspace:4096 \
        --skipInference
```

> **参数说明**：
> - `--memPoolSize=workspace:4096`：设置工作空间大小为 4GB
> - `--skipInference`：跳过推理阶段，仅编译引擎

### 3.3 验证引擎

编译完成后，检查是否生成了 `yolo26x_fp16.engine` 文件：

```Shell
ls -la yolo26x_fp16.engine
```

---

## 第四步：加载推理程序并执行

### 4.1 创建推理脚本

创建 `inference.py`：

```Python
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse


class YOLORunnerTRT:
    def __init__(self, engine_path, input_size=640, num_classes=80, conf_thres=0.25, iou_thres=0.45):
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 加载 Engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 分配显存与内存
        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        self.binding_names = []

        for binding in self.engine:
            self.binding_names.append(binding)
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

        # 设置张量地址（TensorRT 11.x API）
        input_idx = 0
        output_idx = 0
        for binding in self.binding_names:
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.context.set_tensor_address(binding, int(self.inputs[input_idx]["device"]))
                input_idx += 1
            else:
                self.context.set_tensor_address(binding, int(self.outputs[output_idx]["device"]))
                output_idx += 1

    # 前处理：letterbox 缩放 + 归一化 + 维度转换
    def preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        input_h = input_w = self.input_size

        scale = min(input_h / h, input_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        dy, dx = (input_h - new_h) // 2, (input_w - new_w) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.expand_dims(blob, axis=0))
        return blob, scale, (dy, dx)

    # TensorRT 推理
    def infer(self, blob):
        np.copyto(self.inputs[0]["host"], blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])

    # 后处理：置信度过滤 + 坐标转换 + NMS + 原图坐标还原
    def postprocess(self, output, scale, pad, orig_shape):
        preds = output[0].T
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        max_scores = 1 / (1 + np.exp(-max_scores))
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores > self.conf_thres
        boxes_xywh = boxes_xywh[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh → xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), scores.tolist(), self.conf_thres, self.iou_thres
        )

        dy, dx = pad
        h, w = orig_shape
        results = []
        for i in indices:
            i = i.item() if isinstance(i, np.ndarray) else i
            x1 = max(0, min((boxes_xyxy[i, 0] - dx) / scale, w))
            y1 = max(0, min((boxes_xyxy[i, 1] - dy) / scale, h))
            x2 = max(0, min((boxes_xyxy[i, 2] - dx) / scale, w))
            y2 = max(0, min((boxes_xyxy[i, 3] - dy) / scale, h))

            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_id": int(class_ids[i]),
                "score": float(scores[i])
            })
        return results

    def predict(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片文件: {img_path}")
        blob, scale, pad = self.preprocess(img)
        output = self.infer(blob)
        return img, self.postprocess(output, scale, pad, img.shape[:2])

    def draw_results(self, img, results):
        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            class_id = res["class_id"]
            score = res["score"]

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_id}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), (0, 255, 0), -1)
            cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return img


# 测试运行
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO TensorRT 推理脚本")
    parser.add_argument("--engine", type=str, default="yolo26x_fp16.engine", help="TensorRT引擎路径")
    parser.add_argument("--image", type=str, default="test.jpg", help="推理图片路径")
    parser.add_argument("--output", type=str, default="output.jpg", help="输出图片路径")
    args = parser.parse_args()

    runner = YOLORunnerTRT(args.engine)
    img, results = runner.predict(args.image)

    print(f"检测到 {len(results)} 个目标：")
    for res in results:
        print(f"class_id={res['class_id']}, score={res['score']:.3f}, bbox={res['bbox']}")

    img_with_boxes = runner.draw_results(img, results)
    cv2.imwrite(args.output, img_with_boxes)
    print(f"推理结果已保存到 {args.output}")
```

### 4.2 执行推理

```Shell
python inference.py --engine /path/to/your/engine  # 刚刚编译的计算引擎路径 
                    --image input.jpg              # 输入图片路径
                    --output result.jpg            # 输出图片路径
```

### 4.3 预期输出

运行完成后，终端应显示类似以下信息：

```Shell
# 运行完成后，终端应有类似提示
检测到 3 个目标：
class_id=0, score=0.923, bbox=[373.5, 153.875, 600.0, 463.0]
class_id=2, score=0.886, bbox=[311.25, 122.1875, 420.0, 317.5]
class_id=5, score=0.754, bbox=[42.78125, 1.484375, 197.125, 181.5]
推理结果已保存到 result.jpg
```

---

## 附录：常见问题排查

### Q1：`libnvinfer_plugin.so.11: cannot open shared object file`

**原因**：TensorRT 库路径未添加到环境变量

**解决方案**：

```Shell
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

或添加到 `~/.bashrc` 中持久化配置。

### Q2：`execute_async_v3(): incompatible function arguments`

**原因**：TensorRT 版本差异导致 API 调用方式不同

**解决方案**：确保使用 TensorRT 11.x 兼容的 API，参考本文提供的 `inference.py` 脚本。

### Q3：图片读取失败 `NoneType object has no attribute 'shape'`

**原因**：图片路径错误或文件不存在

**解决方案**：检查图片路径是否正确，确保文件存在且格式支持（如 `.jpg`, `.png`）。

### Q4：`trtexec` 命令找不到

**原因**：TensorRT 的 `bin` 目录未添加到 `PATH`

**解决方案**：

```Shell
export PATH=/usr/lib/x86_64-linux-gnu/tensorrt/bin:$PATH
```

---

## 项目文件结构

```
YOLO_ONNX/
├── yolo26x.pt                    # YOLO 26x 模型权重
├── yolo26x.onnx                  # FP32 ONNX 模型
├── yolo26x_fp16.onnx             # FP16 ONNX 模型
├── yolo26x_fp16.engine           # TensorRT 计算引擎
├── onnx_export.py                # ONNX 导出脚本
├── inference.py                  # TensorRT 推理脚本
├── test.jpg                      # 测试图片
├── output.jpg                    # 推理结果图片
└── workflow.md                   # 实验流程文档
```