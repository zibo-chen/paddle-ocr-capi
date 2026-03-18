# OCR Swift iOS 示例

基于 `ocr-capi` 静态库的 iOS Swift 示例应用，演示如何在 iOS 上集成 OCR 功能。

## 功能

- 从相册选择图片进行 OCR 识别
- 支持 Metal GPU 加速
- 显示识别结果（文本、置信度、位置）
- 显示推理耗时
- **旋转矫正**：可选启用方向矫正，自动处理旋转的图片 (0°/90°/180°/270°)
- 方向检测徽章：未启用矫正时显示检测到的旋转角度供参考

## 前置条件

1. **静态库**：确保 `libocr_capi-ios-device.a` 已编译并放置在 `paddle-ocr-capi/examples/swift/` 目录下

   ```bash
   # 编译 iOS aarch64 静态库
   cargo build --release --target aarch64-apple-ios -p paddle-ocr-capi
   cp target/aarch64-apple-ios/release/libocr_capi.a \
      paddle-ocr-capi/examples/swift/libocr_capi-ios-device.a
   ```

2. **必须模型文件**：需要将以下模型文件添加到 Xcode 项目的 Bundle Resources 中：
   - `ch_PP-OCRv4_det_infer.mnn` — 文本检测模型
   - `ch_PP-OCRv4_rec_infer.mnn` — 文本识别模型
   - `ppocr_keys_v4.txt` — 字符集文件

3. **可选模型文件（旋转矫正）**：如需启用旋转矫正功能，还需添加：
   - `PP-LCNet_x1_0_doc_ori.mnn` — 文档方向分类模型（支持 0°/90°/180°/270°）

   模型文件位于项目根目录 `models/` 文件夹下。

## 设置步骤

### 1. 打开 Xcode 项目

```
open ocr-rs-capi-example.xcodeproj
```

### 2. 添加模型文件到 Bundle

1. 在 Xcode 中，右键点击 `ocr-rs-capi-example` 文件夹
2. 选择 **Add Files to "ocr-rs-capi-example"**
3. 选择以下文件（从项目根目录的 `models/` 文件夹中）：
   - `ch_PP-OCRv4_det_infer.mnn`
   - `ch_PP-OCRv4_rec_infer.mnn`
   - `ppocr_keys_v4.txt`
   - （可选）`PP-LCNet_x1_0_doc_ori.mnn`
4. 确保勾选 **"Copy items if needed"** 和 **"Add to targets: ocr-rs-capi-example"**

> **提示**：如果 Bundle 中存在 `PP-LCNet_x1_0_doc_ori.mnn`，界面上会自动出现"旋转矫正"开关。

### 3. 构建并运行

选择一台 iOS 真机设备（不支持模拟器，因为静态库是 arm64 架构），点击 Run。

## 项目结构

```
ocr-rs-capi-example/
├── OcrBridging-Header.h          # C-Swift 桥接头文件
├── OcrEngine.swift               # C API 的 Swift 封装 (OcrEngine + OcrOriModel)
├── ContentView.swift             # 主界面
├── ocr_rs_capi_exampleApp.swift  # App 入口
└── Assets.xcassets/              # 资源文件
```

## Xcode 项目配置说明

项目已预配置以下关键设置：

| 设置项 | 值 |
|-------|---|
| `SWIFT_OBJC_BRIDGING_HEADER` | `ocr-rs-capi-example/OcrBridging-Header.h` |
| `HEADER_SEARCH_PATHS` | `$(PROJECT_DIR)/../../../include` |
| `LIBRARY_SEARCH_PATHS` | `$(PROJECT_DIR)/..` |
| `OTHER_LDFLAGS` | `-locr_capi-ios-device -lc++ -framework Metal -framework CoreML -framework Accelerate` |

## Swift API 说明

### 普通引擎（无旋转矫正）

```swift
let engine = try OcrEngine(
    detModel: "ch_PP-OCRv4_det_infer",
    recModel: "ch_PP-OCRv4_rec_infer",
    charset: "ppocr_keys_v4",
    useGPU: true
)
let results = engine.recognize(image: uiImage)
```

### 带旋转矫正的引擎

```swift
let engine = try OcrEngine(
    detModel: "ch_PP-OCRv4_det_infer",
    recModel: "ch_PP-OCRv4_rec_infer",
    charset: "ppocr_keys_v4",
    oriModel: "PP-LCNet_x1_0_doc_ori",  // 自动矫正旋转
    useGPU: true
)
let results = engine.recognize(image: uiImage)  // 旋转图片也能正确识别
```

### 单独使用方向分类模型

```swift
let oriModel = try OcrOriModel(modelName: "PP-LCNet_x1_0_doc_ori")
let result = oriModel.classify(image: uiImage)
// result.angle: 0, 90, 180, 或 270
// result.confidence: 置信度
print("图片旋转了 \(result.angle)°，置信度 \(result.confidence * 100)%")
```

## 使用其他语言模型

修改 `ContentView.swift` 中的 `reloadEngine()` 方法，替换模型文件名即可：

```swift
// 英文模型
try OcrEngine(
    detModel: "PP-OCRv5_mobile_det",
    recModel: "en_PP-OCRv5_mobile_rec_infer",
    charset: "ppocr_keys_en",
    useGPU: true
)
```

可用模型请参考项目根目录 `models/` 文件夹。


## 前置条件

1. **静态库**：确保 `libocr_capi-ios-device.a` 已编译并放置在 `paddle-ocr-capi/examples/swift/` 目录下

   ```bash
   # 编译 iOS aarch64 静态库
   cargo build --release --target aarch64-apple-ios -p paddle-ocr-capi
   cp target/aarch64-apple-ios/release/libocr_capi.a \
      paddle-ocr-capi/examples/swift/libocr_capi-ios-device.a
   ```

2. **模型文件**：需要将以下模型文件添加到 Xcode 项目的 Bundle Resources 中：
   - `ch_PP-OCRv4_det_infer.mnn` — 文本检测模型
   - `ch_PP-OCRv4_rec_infer.mnn` — 文本识别模型
   - `ppocr_keys_v4.txt` — 字符集文件

   模型文件位于项目根目录 `models/` 文件夹下。

## 设置步骤

### 1. 打开 Xcode 项目

```
open ocr-rs-capi-example.xcodeproj
```

### 2. 添加模型文件到 Bundle

1. 在 Xcode 中，右键点击 `ocr-rs-capi-example` 文件夹
2. 选择 **Add Files to "ocr-rs-capi-example"**
3. 选择以下文件（从项目根目录的 `models/` 文件夹中）：
   - `ch_PP-OCRv4_det_infer.mnn`
   - `ch_PP-OCRv4_rec_infer.mnn`
   - `ppocr_keys_v4.txt`
4. 确保勾选 **"Copy items if needed"** 和 **"Add to targets: ocr-rs-capi-example"**

### 3. 构建并运行

选择一台 iOS 真机设备（不支持模拟器，因为静态库是 arm64 架构），点击 Run。

## 项目结构

```
ocr-rs-capi-example/
├── OcrBridging-Header.h   # C-Swift 桥接头文件
├── OcrEngine.swift         # C API 的 Swift 封装
├── ContentView.swift       # 主界面
├── ocr_rs_capi_exampleApp.swift  # App 入口
└── Assets.xcassets/        # 资源文件
```

## Xcode 项目配置说明

项目已预配置以下关键设置：

| 设置项 | 值 |
|-------|---|
| `SWIFT_OBJC_BRIDGING_HEADER` | `ocr-rs-capi-example/OcrBridging-Header.h` |
| `HEADER_SEARCH_PATHS` | `$(PROJECT_DIR)/../../../include` |
| `LIBRARY_SEARCH_PATHS` | `$(PROJECT_DIR)/..` |
| `OTHER_LDFLAGS` | `-locr_capi-ios-device -lc++ -framework Metal -framework CoreML -framework Accelerate` |

## 使用其他语言模型

修改 `ContentView.swift` 中的 `initEngine()` 方法，替换模型文件名即可：

```swift
// 英文模型
try OcrEngine(
    detModel: "PP-OCRv5_mobile_det",
    recModel: "en_PP-OCRv5_mobile_rec_infer",
    charset: "ppocr_keys_en",
    useGPU: true
)
```

可用模型请参考项目根目录 `models/` 文件夹。
