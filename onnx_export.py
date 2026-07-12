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