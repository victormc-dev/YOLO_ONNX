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