# OCR C API

基于 `ocr-rs` 库的 C 语言接口，提供简单易用的 OCR 功能。

## 特性

- **三层 API 设计**：从底层到高层，满足不同需求
- **零重复初始化**：模型只需加载一次，可重复使用
- **高性能**：原生 Rust 实现，支持多线程和 GPU 加速
- **跨平台**：支持 macOS、Linux、Windows

## API 层级

### 1. 便捷 API（推荐）

最简单的使用方式，直接传入图片路径：

```c
// 创建引擎（只需一次）
OcrEngineHandle* engine = ocr_engine_create(
    "det.mnn", "rec.mnn", "keys.txt", NULL);

// 识别图片
OcrResultList result = ocr_engine_recognize_file(engine, "test.jpg");

// 遍历结果
for (size_t i = 0; i < result.count; i++) {
    printf("文本: %s, 置信度: %.2f%%\n",
           result.items[i].text,
           result.items[i].confidence * 100);
    printf("位置: (%d, %d, %u, %u)\n",
           result.items[i].bbox.x,
           result.items[i].bbox.y,
           result.items[i].bbox.width,
           result.items[i].bbox.height);
}

// 释放资源
ocr_result_list_free(&result);
ocr_engine_destroy(engine);
```

### 2. 普通 API

接收 RGB/RGBA 原始数据：

```c
// 从图片库获取 RGB 数据
unsigned char* rgb_data = load_image_as_rgb("test.jpg", &width, &height);

// 识别
OcrResultList result = ocr_engine_recognize_rgb(engine, rgb_data, width, height);

// 或者 RGBA 数据
OcrResultList result = ocr_engine_recognize_rgba(engine, rgba_data, width, height);
```

### 3. 底层 API

分别控制检测和识别模型：

```c
// 创建检测模型
DetModelHandle* det = ocr_det_model_create("det.mnn", NULL);

// 检测文本区域
DetResultList det_result = ocr_det_model_detect(det, rgb_data, width, height);

// 创建识别模型
RecModelHandle* rec = ocr_rec_model_create("rec.mnn", "keys.txt", NULL);

// 对每个检测到的区域进行识别
for (size_t i = 0; i < det_result.count; i++) {
    // 裁剪区域...
    RecResult rec_result = ocr_rec_model_recognize(rec, cropped_rgb, w, h);
    printf("文本: %s\n", rec_result.text);
    ocr_rec_result_free(&rec_result);
}

// 释放资源
ocr_det_result_free(&det_result);
ocr_det_model_destroy(det);
ocr_rec_model_destroy(rec);
```

## 配置选项

```c
// 默认配置
OcrConfig config = ocr_config_default();

// 快速模式（牺牲一些精度换取速度）
OcrConfig config = ocr_config_fast();

// GPU 模式
OcrConfig config = ocr_config_gpu();

// 自定义配置
OcrConfig config = ocr_config_default();
config.backend = OCR_BACKEND_METAL;  // macOS GPU
config.thread_count = 8;
config.det_max_side_len = 1280;
config.min_result_confidence = 0.7;
```

## 编译

### 编译库

```bash
cd ocr_capi
cargo build --release
```

生成的动态库位于 `target/release/`:
- macOS: `libocr_capi.dylib`
- Linux: `libocr_capi.so`
- Windows: `ocr_capi.dll`

### 编译示例

```bash
# macOS/Linux
gcc -o example examples/example.c -L../target/release -locr_capi -Iinclude

# 运行
./example models/det.mnn models/rec.mnn models/keys.txt test.jpg
```

## 内存管理

所有返回的指针都需要手动释放：

| 函数 | 释放函数 |
|------|----------|
| `ocr_engine_create` | `ocr_engine_destroy` |
| `ocr_det_model_create` | `ocr_det_model_destroy` |
| `ocr_rec_model_create` | `ocr_rec_model_destroy` |
| `ocr_engine_recognize_*` | `ocr_result_list_free` |
| `ocr_det_model_detect` | `ocr_det_result_free` |
| `ocr_rec_model_recognize` | `ocr_rec_result_free` |
| `ocr_get_last_error` | `ocr_free_string` |

## 错误处理

```c
OcrEngineHandle* engine = ocr_engine_create(...);
if (!engine) {
    char* error = ocr_get_last_error();
    fprintf(stderr, "创建引擎失败: %s\n", error);
    ocr_free_string(error);
    return -1;
}
```

## 线程安全

- 每个 Handle（引擎/模型）在单线程内使用是安全的
- 多线程使用时，建议每个线程创建独立的 Handle
- 或者使用外部锁保护共享的 Handle

## 性能建议

1. **复用引擎**：创建一次 `OcrEngineHandle`，多次调用识别函数
2. **批量处理**：多张图片使用同一个引擎
3. **GPU 加速**：在支持的平台使用 `ocr_config_gpu()`
4. **调整参数**：根据实际需求调整 `det_max_side_len` 和 `min_result_confidence`

## 完整 API 参考

参见 [include/ocr_capi.h](include/ocr_capi.h)
