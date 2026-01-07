#!/usr/bin/env python3
"""
OCR Python 示例

展示如何使用 Python 调用 OCR C API
"""

import sys
from pathlib import Path
from ocr_wrapper import OcrEngine, OcrConfig, OcrBackend, OcrPrecision


def example_simple(det_path: str, rec_path: str, charset_path: str, image_path: str):
    """示例 1: 简单使用 (推荐)"""
    print("\n=== 示例 1: 简单使用 ===")
    
    # 使用上下文管理器自动释放资源
    with OcrEngine(det_path, rec_path, charset_path) as engine:
        results = engine.recognize_file(image_path)
        
        print(f"识别结果数量: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"[{i}] 文本: {result.text}")
            print(f"    置信度: {result.confidence * 100:.2f}%")
            print(f"    位置: ({result.bbox.x}, {result.bbox.y}, "
                  f"{result.bbox.width}, {result.bbox.height})")


def example_with_config(det_path: str, rec_path: str, charset_path: str, image_path: str):
    """示例 2: 使用自定义配置"""
    print("\n=== 示例 2: 自定义配置 ===")
    
    # 创建自定义配置
    config = OcrConfig(
        backend=OcrBackend.CPU,
        thread_count=8,
        det_max_side_len=1280,
        min_result_confidence=0.6,
        enable_parallel=True,
    )
    
    with OcrEngine(det_path, rec_path, charset_path, config) as engine:
        results = engine.recognize_file(image_path)
        
        print(f"识别结果数量: {len(results)}")
        for result in results:
            print(f"  {result.text} ({result.confidence * 100:.2f}%)")


def example_gpu(det_path: str, rec_path: str, charset_path: str, image_path: str):
    """示例 3: GPU 加速"""
    print("\n=== 示例 3: GPU 加速 ===")
    
    try:
        # 尝试使用 GPU 配置
        config = OcrConfig.gpu()
        config.thread_count = 8
        
        with OcrEngine(det_path, rec_path, charset_path, config) as engine:
            results = engine.recognize_file(image_path)
            print(f"GPU 模式识别成功，结果数量: {len(results)}")
            
    except RuntimeError as e:
        print(f"GPU 不可用: {e}")
        print("回退到 CPU 模式...")
        
        config = OcrConfig.default()
        with OcrEngine(det_path, rec_path, charset_path, config) as engine:
            results = engine.recognize_file(image_path)
            print(f"CPU 模式识别成功，结果数量: {len(results)}")


def example_fast_mode(det_path: str, rec_path: str, charset_path: str, image_path: str):
    """示例 4: 快速模式"""
    print("\n=== 示例 4: 快速模式 ===")
    
    config = OcrConfig.fast()
    
    with OcrEngine(det_path, rec_path, charset_path, config) as engine:
        results = engine.recognize_file(image_path)
        
        print(f"快速模式识别结果: {len(results)}")
        for result in results:
            print(f"  {result.text}")


def example_batch_processing(
    det_path: str,
    rec_path: str,
    charset_path: str,
    image_paths: list
):
    """示例 5: 批量处理"""
    print("\n=== 示例 5: 批量处理 ===")
    
    # 创建一次引擎，处理多张图片
    with OcrEngine(det_path, rec_path, charset_path) as engine:
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n处理图片 {i}: {image_path}")
            
            results = engine.recognize_file(image_path)
            
            for result in results:
                print(f"  {result.text}")


def example_with_pil(det_path: str, rec_path: str, charset_path: str, image_path: str):
    """示例 6: 配合 PIL 使用"""
    print("\n=== 示例 6: 配合 PIL 使用 ===")
    
    try:
        from PIL import Image
    except ImportError:
        print("请安装 Pillow: pip install Pillow")
        return
    
    # 使用 PIL 加载图片
    img = Image.open(image_path)
    
    # 转换为 RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    rgb_data = img.tobytes()
    
    # 使用 RGB 数据识别
    with OcrEngine(det_path, rec_path, charset_path) as engine:
        results = engine.recognize_rgb(rgb_data, width, height)
        
        print(f"从 PIL 图像识别结果: {len(results)}")
        for result in results:
            print(f"  {result.text} ({result.confidence * 100:.1f}%)")


def example_with_opencv(det_path: str, rec_path: str, charset_path: str, image_path: str):
    """示例 7: 配合 OpenCV 使用"""
    print("\n=== 示例 7: 配合 OpenCV 使用 ===")
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("请安装 OpenCV: pip install opencv-python")
        return
    
    # 使用 OpenCV 读取图片
    img = cv2.imread(image_path)
    
    # OpenCV 默认是 BGR，转换为 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width = img_rgb.shape[:2]
    rgb_data = img_rgb.tobytes()
    
    # 识别
    with OcrEngine(det_path, rec_path, charset_path) as engine:
        results = engine.recognize_rgb(rgb_data, width, height)
        
        print(f"从 OpenCV 图像识别结果: {len(results)}")
        
        # 在图像上绘制结果
        for result in results:
            x, y, w, h = result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height
            
            # 绘制矩形框
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制文本
            cv2.putText(
                img,
                result.text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # 保存结果
        output_path = "output_opencv.jpg"
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")


def main():
    print("OCR Python 示例")
    print(f"版本: {OcrEngine.get_version()}")
    
    if len(sys.argv) < 5:
        print("\n用法: python example.py <det_model> <rec_model> <charset> <image> [image2 ...]")
        print("\n示例:")
        print("  python example.py ../../models/det.mnn ../../models/rec.mnn "
              "../../models/keys.txt ../../res/1.png")
        sys.exit(1)
    
    det_path = sys.argv[1]
    rec_path = sys.argv[2]
    charset_path = sys.argv[3]
    image_path = sys.argv[4]
    
    # 检查文件是否存在
    for path, name in [
        (det_path, "检测模型"),
        (rec_path, "识别模型"),
        (charset_path, "字符集"),
        (image_path, "图片"),
    ]:
        if not Path(path).exists():
            print(f"错误: {name}文件不存在: {path}")
            sys.exit(1)
    
    # 运行示例
    try:
        example_simple(det_path, rec_path, charset_path, image_path)
        example_with_config(det_path, rec_path, charset_path, image_path)
        example_fast_mode(det_path, rec_path, charset_path, image_path)
        
        # 批量处理
        if len(sys.argv) > 5:
            image_paths = sys.argv[4:]
            example_batch_processing(det_path, rec_path, charset_path, image_paths)
        
        # 可选示例（需要额外库）
        example_with_pil(det_path, rec_path, charset_path, image_path)
        example_with_opencv(det_path, rec_path, charset_path, image_path)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
