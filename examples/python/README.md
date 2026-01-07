# Python 示例

使用 Python 调用 OCR C API 的示例代码。

## 文件说明

- `ocr_wrapper.py` - Python 封装类，使用 ctypes 调用 C API
- `example.py` - 完整示例，展示多种使用场景
- `requirements.txt` - Python 依赖（可选）

## 依赖安装

基础功能不需要额外依赖，仅需 Python 3.7+

可选依赖（用于某些示例）：

```bash
pip install -r requirements.txt
```

或者：

```bash
pip install Pillow opencv-python
```

## 快速开始

### 1. 编译 C 库

```bash
cd ../..  # 回到 ocr_capi 目录
cargo build --release
```

### 2. 运行示例

```bash
python example.py \
    ../../models/PP-OCRv5_mobile_det_fp16.mnn \
    ../../models/PP-OCRv5_mobile_rec_fp16.mnn \
    ../../models/ppocr_keys_v5.txt \
    ../../res/1.png
```

## 使用方法

### 基础用法

```python
from ocr_wrapper import OcrEngine

# 创建引擎
with OcrEngine(det_model, rec_model, charset) as engine:
    # 识别图片
    results = engine.recognize_file("test.jpg")
    
    # 遍历结果
    for result in results:
        print(f"文本: {result.text}")
        print(f"置信度: {result.confidence}")
        print(f"位置: {result.bbox.x}, {result.bbox.y}")
```

### 自定义配置

```python
from ocr_wrapper import OcrEngine, OcrConfig, OcrBackend

# 创建配置
config = OcrConfig(
    backend=OcrBackend.CPU,
    thread_count=8,
    det_max_side_len=1280,
    min_result_confidence=0.7,
)

# 使用配置创建引擎
engine = OcrEngine(det_model, rec_model, charset, config)
```

### 预设配置

```python
# 快速模式
config = OcrConfig.fast()

# GPU 模式
config = OcrConfig.gpu()

# 默认模式
config = OcrConfig.default()
```

### 批量处理

```python
# 复用引擎处理多张图片
with OcrEngine(det_model, rec_model, charset) as engine:
    for image_path in image_list:
        results = engine.recognize_file(image_path)
        # 处理结果...
```

### 配合 PIL 使用

```python
from PIL import Image

img = Image.open("test.jpg")
img = img.convert('RGB')

width, height = img.size
rgb_data = img.tobytes()

with OcrEngine(det_model, rec_model, charset) as engine:
    results = engine.recognize_rgb(rgb_data, width, height)
```

### 配合 OpenCV 使用

```python
import cv2

img = cv2.imread("test.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width = img_rgb.shape[:2]
rgb_data = img_rgb.tobytes()

with OcrEngine(det_model, rec_model, charset) as engine:
    results = engine.recognize_rgb(rgb_data, width, height)
    
    # 在图像上绘制结果
    for result in results:
        x, y, w, h = result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, result.text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

## API 参考

### OcrEngine

主要的 OCR 引擎类。

**构造函数**

```python
OcrEngine(
    det_model_path: str,
    rec_model_path: str,
    charset_path: str,
    config: Optional[OcrConfig] = None,
    lib_path: Optional[str] = None,
)
```

**方法**

- `recognize_file(image_path: str) -> List[OcrResult]` - 识别图片文件
- `recognize_rgb(rgb_data: bytes, width: int, height: int) -> List[OcrResult]` - 识别 RGB 数据
- `recognize_rgba(rgba_data: bytes, width: int, height: int) -> List[OcrResult]` - 识别 RGBA 数据
- `get_version() -> str` - 获取库版本（静态方法）

### OcrConfig

引擎配置类。

**构造函数参数**

- `backend: int` - 后端类型（CPU/Metal/OpenCL/Vulkan）
- `thread_count: int` - 线程数，默认 4
- `precision: int` - 精度模式（Normal/Low/High）
- `det_max_side_len: int` - 检测图像最大边长，默认 960
- `det_box_threshold: float` - 检测框阈值，默认 0.5
- `det_score_threshold: float` - 检测分数阈值，默认 0.3
- `rec_min_score: float` - 识别最小置信度，默认 0.3
- `min_result_confidence: float` - 最终结果最小置信度，默认 0.5
- `enable_parallel: bool` - 是否启用并行处理，默认 True

**预设配置方法**

- `OcrConfig.default()` - 默认配置
- `OcrConfig.fast()` - 快速模式
- `OcrConfig.gpu()` - GPU 模式

### OcrResult

识别结果数据类。

**属性**

- `text: str` - 识别的文本
- `confidence: float` - 置信度 (0.0-1.0)
- `bbox: BBox` - 文本框位置

### BBox

文本框位置数据类。

**属性**

- `x: int` - 左上角 x 坐标
- `y: int` - 左上角 y 坐标
- `width: int` - 宽度
- `height: int` - 高度

## 性能优化建议

1. **复用引擎**: 创建引擎开销较大，应该复用同一个引擎实例
2. **使用上下文管理器**: 自动管理资源释放
3. **GPU 加速**: 在支持的平台使用 GPU 模式
4. **调整线程数**: 根据 CPU 核心数调整 `thread_count`
5. **调整图像尺寸**: 通过 `det_max_side_len` 平衡速度和精度

## 故障排除

### 找不到动态库

确保已经编译了 C 库：

```bash
cd ../..
cargo build --release
```

动态库位于 `../../target/release/` 目录下。

### 模型加载失败

检查模型文件路径是否正确，文件是否存在。

### GPU 不可用

如果 GPU 模式初始化失败，会抛出异常。可以捕获异常并回退到 CPU 模式。

## 注意事项

- 引擎对象不是线程安全的，多线程使用时每个线程应创建独立的引擎
- 使用完毕后务必释放引擎（使用 `with` 语句或手动调用 `__del__`）
- RGB/RGBA 数据应该是连续的字节序列
