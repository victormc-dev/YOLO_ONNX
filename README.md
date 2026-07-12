# YOLO 26 TensorRT 推理实验

基于 TensorRT 11.x 的 YOLO 26 模型推理实现。

## 项目结构

```
YOLO_ONNX/
├── onnx_export.py       # ONNX 模型导出脚本
├── inference.py         # TensorRT 推理脚本
├── fp16.py              # FP16 量化脚本
├── workflow.md          # 详细实验流程文档
└── .gitignore           # Git 忽略配置
```

## 快速开始

请参考 [workflow.md](workflow.md) 获取详细的环境配置和使用指南。

## 环境要求

- RTX 5090（或兼容的 NVIDIA GPU）
- CUDA 13.3
- TensorRT 11.1.0
- Ubuntu 24.04 / WSL2
- Python 3.12

## 主要功能

- YOLO 26 模型导出为 ONNX 格式
- FP16 量化优化
- TensorRT 引擎编译
- 高性能推理（包含前处理和后处理）
- 检测结果可视化

## 许可证

MIT License