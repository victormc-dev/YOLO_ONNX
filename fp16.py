import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_fp16_engine(onnx_path, engine_path, workspace_gb=4):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 设置显存工作空间
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 
        workspace_gb << 30
    )

    # 开启 FP16 精度（TensorRT 11.x API 依然支持，只是trtexec命令移除了参数）
    config.set_flag(trt.BuilderFlag.FP16)

    # 解析ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ ONNX解析失败，错误如下：")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    # 构建并序列化引擎
    print("🔨 开始构建FP16引擎，请稍候...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ 引擎构建失败")
        return False

    # 保存到文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"✅ FP16引擎已保存: {engine_path}")
    print(f"文件大小: {os.path.getsize(engine_path)/1024/1024:.2f} MB")
    return True

# 执行构建
if __name__ == "__main__":
    build_fp16_engine(
        onnx_path="yolo26x.onnx",
        engine_path="yolo26x_fp16.engine",
        workspace_gb=4
    )